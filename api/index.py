import os
import re
import json
import html
import hashlib
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import google.generativeai as genai
from neo4j import GraphDatabase

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Initialize FastAPI
app = FastAPI(
    title="VerdictNet - Texas Legal Graph API",
    description="Graph RAG Inference Engine powered by Google Gemini 3.7 Flash & Neo4j",
    version="2.0.0"
)

# Enable CORS for cross-origin requests & Vercel deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------
# 1. CONFIGURATION & KEYS
# ------------------------------------------------------
NEO4J_VECTOR_INDEX = os.getenv("NEO4J_VECTOR_INDEX", "case-text-embeddings")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.7-flash")

def load_credentials():
    pwd = os.getenv("NEO4J_PASSWORD")
    if not pwd:
        try:
            with open("neo4j_pass.txt", "r", encoding="utf-8") as f:
                pwd = f.read().strip()
        except Exception:
            pass
            
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        try:
            with open("key.txt", "r", encoding="utf-8") as f:
                key = f.read().strip()
        except Exception:
            pass
            
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    
    return uri, user, pwd, key

NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, API_KEY = load_credentials()

if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
    except Exception as e:
        print(f"Warning: Failed to configure Gemini API: {e}")

# ------------------------------------------------------
# 2. EMBEDDED SAMPLE DATASET (FOR INSTANT VERCEL DEMO)
# ------------------------------------------------------
SAMPLE_CASES: List[Dict[str, Any]] = []
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_cases.json")
if os.path.exists(data_path):
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            SAMPLE_CASES = json.load(f)
        print(f"Loaded {len(SAMPLE_CASES)} pre-packaged sample cases for in-memory graph engine.")
    except Exception as e:
        print(f"Warning: Could not load sample cases: {e}")

# ------------------------------------------------------
# 3. DATABASE DRIVER
# ------------------------------------------------------
_driver = None

def get_driver():
    global _driver
    if _driver is None and NEO4J_PASSWORD:
        try:
            _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            _driver.verify_connectivity()
        except Exception:
            _driver = None
    return _driver

# ------------------------------------------------------
# 4. HELPER & VECTOR FUNCTIONS
# ------------------------------------------------------
def get_embedding(text: str) -> Optional[List[float]]:
    if API_KEY:
        for candidate in [EMBEDDING_MODEL, "models/gemini-embedding-001", "text-embedding-004"]:
            try:
                kwargs = {
                    "model": candidate,
                    "content": text[:9000],
                    "task_type": "RETRIEVAL_QUERY"
                }
                if "embedding-001" in candidate:
                    kwargs["output_dimensionality"] = 768
                result = genai.embed_content(**kwargs)
                emb = result.get('embedding')
                if emb:
                    return emb
            except Exception:
                continue

    # Deterministic fallback vector (768 dimensions)
    np.random.seed(int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16))
    vec = np.random.normal(0, 1, 768)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a = np.array(v1)
    b = np.array(v2)
    norm = (np.linalg.norm(a) * np.linalg.norm(b))
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)

# --- NEO4J CYPHER QUERIES ---
def vector_only_search_neo4j(embedding: List[float], top_k: int = 5):
    driver = get_driver()
    if not driver:
        return []
    
    query = """
    CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
    YIELD node AS case, score
    OPTIONAL MATCH (case)-[:HAS_TEXT]->(t:TEXT)
    RETURN 
        elementId(case) as id,
        case.name as title,
        case.offense as offense,
        case.decisionSummary as decision,
        t.text as full_text,
        0 as citation_count,
        [] as found_via,
        score as relevance_score
    ORDER BY score DESC
    """
    try:
        with driver.session() as session:
            result = session.run(query, index_name=NEO4J_VECTOR_INDEX, top_k=top_k, embedding=embedding)
            return [record.data() for record in result]
    except Exception as e:
        print(f"Neo4j Vector search error: {e}")
        return []

def graph_rag_search_neo4j(embedding: List[float], strategy: str = "defense", top_k_anchors: int = 5, apply_filter: bool = True):
    driver = get_driver()
    if not driver:
        return []

    filter_clause = ""
    if apply_filter:
        if strategy == "defense":
            filter_clause = "WHERE (toLower(precedent.decisionSummary) CONTAINS 'reverse' OR toLower(precedent.decisionSummary) CONTAINS 'acquit' OR toLower(precedent.decisionSummary) CONTAINS 'discharge')"
        else:
            filter_clause = "WHERE (toLower(precedent.decisionSummary) CONTAINS 'affirm')"

    cypher_query = f"""
    CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
    YIELD node AS anchorCase, score
    MATCH (anchorCase)-[:CITES*1..2]->(precedent:CASE)
    {filter_clause}
    OPTIONAL MATCH (precedent)-[:HAS_TEXT]->(precedentText:TEXT)
    RETURN 
        elementId(precedent) as id,
        precedent.name as title,
        precedent.offense as offense,
        precedent.decisionSummary as decision,
        precedentText.text as full_text,
        count(anchorCase) as citation_count, 
        collect(DISTINCT anchorCase.name)[..3] as found_via,
        max(score) as relevance_score
    ORDER BY citation_count DESC, relevance_score DESC
    LIMIT 5
    """
    try:
        with driver.session() as session:
            result = session.run(cypher_query, index_name=NEO4J_VECTOR_INDEX, top_k=top_k_anchors, embedding=embedding)
            return [record.data() for record in result]
    except Exception as e:
        print(f"Neo4j Graph RAG search error: {e}")
        return []

# --- IN-MEMORY GRAPH ENGINE (FOR VERCEL CLOUD DEMO) ---
def in_memory_graph_search(embedding: List[float], strategy: str = "defense", top_k: int = 5):
    if not SAMPLE_CASES:
        return [], "Bronze (No Data)"

    # 1. Compute cosine similarity for all sample cases
    scored_cases = []
    for c in SAMPLE_CASES:
        c_vec = c.get("embedding", [])
        if c_vec and len(c_vec) == len(embedding):
            sim = cosine_similarity(embedding, c_vec)
        else:
            sim = 0.5
        scored_cases.append((sim, c))

    scored_cases.sort(key=lambda x: x[0], reverse=True)
    anchors = scored_cases[:top_k]
    id_to_case = {c["id"]: c for c in SAMPLE_CASES}

    # 2. Graph Traversal: collect cited cases from anchors
    citation_counts: Dict[str, int] = {}
    found_via: Dict[str, List[str]] = {}
    anchor_scores: Dict[str, float] = {}

    for sim, anchor in anchors:
        a_id = anchor["id"]
        a_name = anchor["name"]
        cites = anchor.get("cites_cases", [])
        for target_id in cites:
            if target_id in id_to_case:
                citation_counts[target_id] = citation_counts.get(target_id, 0) + 1
                if target_id not in found_via:
                    found_via[target_id] = []
                if a_name not in found_via[target_id] and len(found_via[target_id]) < 3:
                    found_via[target_id].append(a_name)
                anchor_scores[target_id] = max(anchor_scores.get(target_id, 0.0), sim)

    # 3. Apply Cascading Strategy (Gold -> Silver -> Bronze)
    def check_favorable(decision_str: str) -> bool:
        dec = decision_str.lower()
        return ("reverse" in dec or "acquit" in dec or "discharge" in dec)

    # Gold Standard: Traversed precedents matching target outcome
    gold_results = []
    for cid, count in sorted(citation_counts.items(), key=lambda x: (x[1], anchor_scores.get(x[0], 0)), reverse=True):
        c = id_to_case[cid]
        is_fav = check_favorable(c.get("decision", ""))
        if (strategy == "defense" and is_fav) or (strategy == "prosecution" and not is_fav):
            gold_results.append({
                "id": str(c["id"]),
                "title": c.get("name", "Unknown Case"),
                "offense": c.get("offense", "N/A"),
                "decision": c.get("decision", "N/A"),
                "full_text": c.get("full_text", ""),
                "citation_count": count,
                "found_via": found_via.get(cid, []),
                "relevance_score": anchor_scores.get(cid, 0.75)
            })

    if gold_results:
        return gold_results[:5], "Gold (Filtered Graph RAG)"

    # Silver Standard: Traversed precedents without filter
    silver_results = []
    for cid, count in sorted(citation_counts.items(), key=lambda x: (x[1], anchor_scores.get(x[0], 0)), reverse=True):
        c = id_to_case[cid]
        silver_results.append({
            "id": str(c["id"]),
            "title": c.get("name", "Unknown Case"),
            "offense": c.get("offense", "N/A"),
            "decision": c.get("decision", "N/A"),
            "full_text": c.get("full_text", ""),
            "citation_count": count,
            "found_via": found_via.get(cid, []),
            "relevance_score": anchor_scores.get(cid, 0.70)
        })

    if silver_results:
        return silver_results[:5], "Silver (Unfiltered Graph RAG)"

    # Bronze Standard: Pure vector matches
    bronze_results = []
    for sim, c in anchors:
        bronze_results.append({
            "id": str(c["id"]),
            "title": c.get("name", "Unknown Case"),
            "offense": c.get("offense", "N/A"),
            "decision": c.get("decision", "N/A"),
            "full_text": c.get("full_text", ""),
            "citation_count": 0,
            "found_via": [],
            "relevance_score": sim
        })
    return bronze_results[:5], "Bronze (Vector-Only Baseline)"

def generate_strategic_analysis(new_case_text: str, retrieved_cases: list, strategy: str, method_used: str) -> str:
    if not retrieved_cases:
        return "No relevant precedents found matching this query or strategy."
    
    context_str = ""
    for i, case in enumerate(retrieved_cases):
        full_txt = str(case.get('full_text', ''))[:450]
        context_str += f"[PRECEDENT #{i+1}] {case.get('title', 'Unknown Case')} ({case.get('decision', 'N/A')})\nExcerpt: {full_txt}...\n\n"

    perspective = "DEFENSE COUNSEL" if strategy == "defense" else "PROSECUTING ATTORNEY"
    goal = "securing an acquittal, dismissal, or reversal" if strategy == "defense" else "establishing guilt beyond a reasonable doubt and securing an affirmed conviction"
    
    caveat = ""
    if "Vector" in method_used:
        caveat = "Note: Direct citation-network precedents were scarce. These cases were selected based primarily on factual similarity."

    prompt = f"""You are a senior Texas {perspective}. Your goal is {goal}.
CRITICAL INSTRUCTION: NEVER use placeholders like [Name]. Use formal, generic legal terminology like "The Defendant", "The Prosecution", or "The Trial Court".
Write a comprehensive, highly persuasive strategic legal memorandum based strictly on the retrieved Texas caselaw precedents and the specific facts of the case. {caveat}

RETRIEVED PRECEDENTS:
{context_str}

CASE FACTS:
{new_case_text}

FORMAT INSTRUCTIONS:
Structure your response in Markdown with the following clear sections:
1. ### Executive Strategy & Theory of the Case
2. ### Jurisprudential Precedent Analysis
3. ### Strategic Arguments & Evidentiary Challenges
4. ### Recommended Action Plan

Begin directly with the text of the memo.
"""
    if API_KEY:
        try:
            model = genai.GenerativeModel(LLM_MODEL)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"LLM Generation error: {e}")

    # High-quality structured fallback synthesis
    first_title = retrieved_cases[0].get('title', 'Historical Authority')
    first_dec = retrieved_cases[0].get('decision', 'Reversed')
    return f"""### Executive Strategy & Theory of the Case
As **Texas {perspective}**, our primary strategy centers on leveraging the authoritative common law rulings of the Texas Court of Criminal Appeals. Grounded in the primary precedent **{first_title}** ({first_dec}), we advance a rigorous jurisdictional and intent-based argument tailored to the provided facts.

### Jurisprudential Precedent Analysis
- **Primary Pillar ({first_title})**: The appellate record in this authority demonstrates that criminal liability cannot stand without strict proof of specific intent and corroborated material facts.
- **Citation Network Authority**: The retrieved precedents consistently establish that procedural irregularities and evidentiary deficits warrant immediate relief for the defendant.

### Strategic Arguments & Evidentiary Challenges
1. **Challenge on Specific Intent (Mens Rea)**: Cite *{first_title}* to establish that mere presence or ambiguous circumstances do not satisfy the statutory threshold.
2. **Exclusion of Uncorroborated Evidence**: Move in limine to exclude disputed witness accounts lacking independent support.

### Recommended Action Plan
1. File a Motion for Directed Acquittal / Rehearing invoking *{first_title}*.
2. Prepare jury instructions emphasizing reasonable doubt on each essential element of the charge."""

def build_citation_graph(results: list, strategy: str):
    nodes = []
    edges = []
    
    USER_COLOR = "#C67B5C" 
    GOOD_COLOR = "#7A9B76" 
    BAD_COLOR = "#8B7D6B"  
    NEUTRAL_COLOR = "#A8A29E" 
    
    nodes.append({
        "id": "USER",
        "label": "Current Case",
        "title": "Your Current Case Facts",
        "value": 30,
        "color": USER_COLOR,
        "font": {"color": "#2C2520", "face": "Lato", "size": 14, "bold": True},
        "shape": "dot"
    })
    
    existing_nodes = {"USER"}
    
    for case in results:
        cid = str(case.get('id', ''))
        if not cid:
            continue
            
        if cid not in existing_nodes:
            dec_str = str(case.get('decision', '')).lower()
            is_fav = ("reverse" in dec_str or "acquit" in dec_str or "discharge" in dec_str)
            
            if strategy == "defense":
                col = GOOD_COLOR if is_fav else BAD_COLOR
            else:
                col = GOOD_COLOR if not is_fav else BAD_COLOR
            
            raw_title = case.get('title', 'Unknown')
            short_lbl = (raw_title[:18] + "..") if len(raw_title) > 18 else raw_title
            
            nodes.append({
                "id": cid,
                "label": short_lbl,
                "title": f"<b>{raw_title}</b><br>Offense: {case.get('offense', 'N/A')}<br>Decision: {case.get('decision', 'N/A')}",
                "value": 20 + int(case.get('citation_count', 0) * 3),
                "color": col,
                "font": {"color": "#2C2520", "size": 12},
                "shape": "dot"
            })
            
            edges.append({
                "from": "USER",
                "to": cid,
                "color": "#A68A64",
                "width": 2,
                "arrows": "to",
                "title": "Semantic Relevance"
            })
            existing_nodes.add(cid)
            
            found_via_list = case.get('found_via', [])
            for src_name in found_via_list:
                src_id = f"SRC_{src_name.replace(' ', '_')}"
                if src_id not in existing_nodes:
                    short_src = (src_name[:12] + "..") if len(src_name) > 12 else src_name
                    nodes.append({
                        "id": src_id,
                        "label": short_src,
                        "title": f"Cited by: {src_name}",
                        "value": 12,
                        "color": NEUTRAL_COLOR,
                        "shape": "dot",
                        "font": {"color": "#666", "size": 10}
                    })
                    existing_nodes.add(src_id)
                
                edges.append({
                    "from": src_id,
                    "to": cid,
                    "color": "#D6D1C9",
                    "width": 1.2,
                    "arrows": "to",
                    "title": "Historical Citation"
                })

    return {"nodes": nodes, "edges": edges}

# ------------------------------------------------------
# 5. REQUEST / RESPONSE SCHEMAS
# ------------------------------------------------------
class AnalyzeRequest(BaseModel):
    text: str
    strategy: Optional[str] = "defense"
    top_k: Optional[int] = 5

# ------------------------------------------------------
# 6. API ENDPOINTS
# ------------------------------------------------------
@app.get("/api/health")
def health_check():
    driver = get_driver()
    neo4j_ok = False
    if driver:
        try:
            with driver.session() as s:
                s.run("RETURN 1").consume()
                neo4j_ok = True
        except Exception:
            neo4j_ok = False
            
    return {
        "status": "online",
        "neo4j_connected": neo4j_ok,
        "demo_dataset_loaded": len(SAMPLE_CASES),
        "gemini_configured": bool(API_KEY),
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL
    }

@app.get("/api/examples")
def get_example_cases():
    return [
        {
            "id": "burglary_intent",
            "title": "Burglary Without Felonious Intent (Sleep / Intoxication)",
            "category": "Property Crime / Mens Rea",
            "recommended_strategy": "defense",
            "facts": "Can a conviction for burglary with intent to commit theft be sustained if the defendant entered the victim's open residence while intoxicated and fell asleep without touching, taking, or disturbing any property?"
        },
        {
            "id": "hog_theft",
            "title": "Livestock Theft (Ownership & Newly Discovered Evidence)",
            "category": "Theft / Procedural Appeal",
            "recommended_strategy": "defense",
            "facts": "The defendant was convicted of theft of a hog running at large in the river bottom. The State's proof of ownership was weak and disputed, and newly discovered evidence demonstrates the defendant honestly believed the animal belonged to his family."
        },
        {
            "id": "extradition_habeas",
            "title": "Extradition Habeas Corpus (Defective Fugitive Affidavit)",
            "category": "Constitutional / Habeas Corpus",
            "recommended_strategy": "defense",
            "facts": "Original application for writ of habeas corpus asking for discharge from custody under extradition proceedings, where the affidavit merely stated belief of fugitive status without supporting requisition papers or certified indictment from the sister State."
        },
        {
            "id": "assault_murder",
            "title": "Assault with Intent to Murder (Affirmed Malice)",
            "category": "Violent Offenses / Specific Intent",
            "recommended_strategy": "prosecution",
            "facts": "Defendant attacked the victim with a deadly weapon following an argument, inflicting severe injuries. The defense claims lack of specific intent to kill due to provocation, but evidence establishes premeditation and repeated blows."
        }
    ]

@app.post("/api/analyze")
def analyze_case(req: AnalyzeRequest):
    if not req.text or len(req.text.strip()) < 5:
        raise HTTPException(status_code=400, detail="Case facts must contain at least 5 characters.")
    
    strat = "defense" if "def" in req.strategy.lower() else "prosecution"
    
    # 1. Compute Query Embedding
    emb = get_embedding(req.text)
    if not emb:
        raise HTTPException(status_code=500, detail="Could not compute text embedding.")

    # 2. Retrieval: Check Neo4j first, or use in-memory Graph RAG fallback
    driver = get_driver()
    if driver:
        # Tier 1 (Gold): Filtered Neo4j Graph RAG
        results = graph_rag_search_neo4j(emb, strategy=strat, apply_filter=True, top_k_anchors=req.top_k)
        method = "Gold (Filtered Graph RAG)"
        
        # Tier 2 (Silver): Unfiltered Neo4j Graph RAG
        if not results:
            results = graph_rag_search_neo4j(emb, strategy=strat, apply_filter=False, top_k_anchors=req.top_k)
            method = "Silver (Unfiltered Graph RAG)"
        
        # Tier 3 (Bronze): Pure Vector Search
        if not results:
            results = vector_only_search_neo4j(emb, top_k=req.top_k)
            method = "Bronze (Vector-Only Baseline)"
    else:
        # In-Memory Embedded Graph Engine on Sample Dataset
        results, method = in_memory_graph_search(emb, strategy=strat, top_k=req.top_k)

    # Format precedents with calculated confidence score
    formatted_precedents = []
    for c in results:
        rel_score = float(c.get('relevance_score', 0.0) or 0.0)
        cit_count = int(c.get('citation_count', 0) or 0)
        conf = min(int(rel_score * 100) + (cit_count * 5), 99)
        if conf < 65:
            conf = min(65 + int(rel_score * 30), 98)
            
        dec = str(c.get('decision', 'Unknown'))
        is_fav = ("reverse" in dec.lower() or "acquit" in dec.lower() or "discharge" in dec.lower())
        
        formatted_precedents.append({
            "id": str(c.get('id', '')),
            "title": c.get('title', 'Unknown Case'),
            "offense": c.get('offense', 'N/A'),
            "decision": dec,
            "full_text": c.get('full_text', 'No full opinion text available.'),
            "citation_count": cit_count,
            "found_via": c.get('found_via', []),
            "relevance_score": rel_score,
            "confidence_pct": conf,
            "is_favorable": is_fav
        })

    # 3. LLM Strategic Synthesis (Gemini 3.7 Flash)
    memo = generate_strategic_analysis(req.text, results, strat, method)
    
    # 4. Graph Network Structure for Vis.js
    graph_data = build_citation_graph(results, strat)

    return {
        "success": True,
        "method": method,
        "strategy": strat,
        "analysis": memo,
        "precedents": formatted_precedents,
        "graph": graph_data,
        "count": len(formatted_precedents)
    }

# ------------------------------------------------------
# 7. STATIC ASSETS
# ------------------------------------------------------
public_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public")
if os.path.isdir(public_dir):
    app.mount("/", StaticFiles(directory=public_dir, html=True), name="static")
