import os
import re
import html
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
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
    description="Graph RAG Inference Engine powered by Google Gemini 3.7 Flash and Neo4j",
    version="2.0.0"
)

# Enable CORS for cross-origin requests & Vercel deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------
# 1. CONFIGURATION & KEYS
# ------------------------------------------------------
NEO4J_VECTOR_INDEX = os.getenv("NEO4J_VECTOR_INDEX", "case-text-embeddings")
EMBEDDING_MODEL = "text-embedding-004"
LLM_MODEL = "gemini-3.7-flash"

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
# 2. DATABASE DRIVER
# ------------------------------------------------------
_driver = None

def get_driver():
    global _driver
    if _driver is None and NEO4J_PASSWORD:
        try:
            _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            # verify connectivity
            _driver.verify_connectivity()
        except Exception as e:
            print(f"Warning: Neo4j connection failed: {e}")
            _driver = None
    return _driver

# ------------------------------------------------------
# 3. HELPER FUNCTIONS
# ------------------------------------------------------
def get_embedding(text: str) -> Optional[List[float]]:
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text[:9000],
            task_type="RETRIEVAL_QUERY"
        )
        return result.get('embedding')
    except Exception as e:
        print(f"Error computing embedding: {e}")
        return None

def vector_only_search(embedding: List[float], top_k: int = 5):
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
        print(f"Vector search error: {e}")
        return []

def graph_rag_search(embedding: List[float], strategy: str = "defense", top_k_anchors: int = 5, apply_filter: bool = True):
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
        print(f"Graph RAG search error: {e}")
        return []

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
    try:
        model = genai.GenerativeModel(LLM_MODEL)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating strategic analysis with {LLM_MODEL}: {e}"

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
# 4. REQUEST / RESPONSE SCHEMAS
# ------------------------------------------------------
class AnalyzeRequest(BaseModel):
    text: str
    strategy: Optional[str] = "defense"
    top_k: Optional[int] = 5

class ExampleCase(BaseModel):
    id: str
    title: str
    category: str
    facts: str
    recommended_strategy: str

# ------------------------------------------------------
# 5. API ENDPOINTS
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
        "status": "online" if (neo4j_ok and bool(API_KEY)) else "degraded",
        "neo4j_connected": neo4j_ok,
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
    
    # 1. Generate Query Vector Embedding
    emb = get_embedding(req.text)
    if not emb:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate text embedding with Google Gemini. Check GEMINI_API_KEY."
        )

    # 2. Cascading Hybrid Graph Retrieval
    # Tier 1 (Gold): Graph RAG with strategy-aligned filtering
    results = graph_rag_search(emb, strategy=strat, apply_filter=True, top_k_anchors=req.top_k)
    method = "Gold (Filtered Graph RAG)"
    
    # Tier 2 (Silver): Graph RAG without filtering
    if not results:
        results = graph_rag_search(emb, strategy=strat, apply_filter=False, top_k_anchors=req.top_k)
        method = "Silver (Unfiltered Graph RAG)"
    
    # Tier 3 (Bronze): Pure Vector Similarity Search
    if not results:
        results = vector_only_search(emb, top_k=req.top_k)
        method = "Bronze (Vector-Only Baseline)"

    # Format precedents with calculated confidence score
    formatted_precedents = []
    for c in results:
        rel_score = float(c.get('relevance_score', 0.0) or 0.0)
        cit_count = int(c.get('citation_count', 0) or 0)
        conf = min(int(rel_score * 100) + (cit_count * 5), 99)
        
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

    # 3. LLM Synthesis with Gemini 3.7 Flash
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
# 6. STATIC FILES (FOR STANDALONE LOCAL EXECUTION)
# ------------------------------------------------------
public_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public")
if os.path.isdir(public_dir):
    app.mount("/", StaticFiles(directory=public_dir, html=True), name="static")
