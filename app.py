import os
import html
import streamlit as st
import google.generativeai as genai
from neo4j import GraphDatabase
from streamlit_agraph import agraph, Node, Edge, Config

# ------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------
st.set_page_config(
    page_title="Texas Legal Graph",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------
# 2. CONFIG & KEYS
# ------------------------------------------------------
NEO4J_VECTOR_INDEX = "case-text-embeddings" 

def load_keys():
    pwd = os.getenv("NEO4J_PASSWORD")
    if not pwd:
        try:
            with open("neo4j_pass.txt", "r") as f: pwd = f.read().strip()
        except: pass
        
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        try:
            with open("key.txt", "r") as f: key = f.read().strip()
        except: pass
        
    if not pwd or not key:
        st.error("Missing credentials. Please check neo4j_pass.txt and key.txt")
        st.stop()
    return pwd, key

NEO4J_PASSWORD, API_KEY = load_keys()
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"

genai.configure(api_key=API_KEY)
EMBEDDING_MODEL = 'text-embedding-004'
LLM_MODEL = 'gemini-3.7-flash'

# ------------------------------------------------------
# 3. BACKEND LOGIC
# ------------------------------------------------------
@st.cache_resource
def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def get_embedding(text):
    try:
        result = genai.embed_content(model=EMBEDDING_MODEL, content=text, task_type="RETRIEVAL_QUERY")
        return result['embedding']
    except Exception as e:
        return None

# --- BRONZE STANDARD (VECTOR ONLY) ---
def vector_only_search(embedding, top_k=5):
    driver = get_driver()
    if not driver: return []
    
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
        return []

# --- GOLD & SILVER STANDARD (GRAPH) ---
def graph_rag_search(embedding, strategy="defense", top_k_anchors=5, apply_filter=True):
    driver = get_driver()
    if not driver: return []

    filter_clause = ""
    if apply_filter:
        if strategy == "defense":
            filter_clause = "WHERE (toLower(precedent.decisionSummary) CONTAINS 'reverse' OR toLower(precedent.decisionSummary) CONTAINS 'acquit')"
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
        return []

def generate_strategic_analysis(new_case_text, retrieved_cases, strategy, method_used):
    if not retrieved_cases: return "No relevant precedents found."
    
    context_str = ""
    for i, case in enumerate(retrieved_cases):
        context_str += f"[PRECEDENT #{i+1}] {case['title']} ({case['decision']})\nExcerpt: {str(case.get('full_text', ''))[:400]}...\n\n"

    perspective = "DEFENSE ATTORNEY" if strategy == "defense" else "PROSECUTOR"
    goal = "securing an acquittal" if strategy == "defense" else "securing a conviction"
    
    caveat = ""
    if "Vector" in method_used:
        caveat = "Note: Direct precedents matching the strategy were scarce. These cases were selected based on factual similarity."

    prompt = f"""You are a senior Texas {perspective}. Goal: {goal}.
    CRITICAL: NEVER use placeholders like [Name]. Use generic terms like "The Defendant".
    Write a concise strategic memo based on these precedents. {caveat}
    
    PRECEDENTS:
    {context_str}
    CASE FACTS: {new_case_text}
    
    Format neatly with Markdown. Start directly with the text.
    """
    try:
        model = genai.GenerativeModel(LLM_MODEL)
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Error: {e}"

# ------------------------------------------------------
# 4. UI STYLING & COMPONENTS
# ------------------------------------------------------

def load_luxury_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Lato:wght@300;400;600;700&display=swap');
        
        .stApp {
            background-color: #F2F0E9 !important;
            color: #2C2520 !important;
            font-family: 'Lato', sans-serif !important;
        }
        
        footer {visibility: hidden;}
        
        h1, h2, h3, h4, p, span, div, label {
            color: #2C2520 !important;
        }
        h1 { font-family: 'Playfair Display', serif !important; }
        
        section[data-testid="stSidebar"] {
            background-color: #E8E5DD !important;
            border-right: 1px solid #D4CFC4;
        }
        
        section[data-testid="stSidebar"] .stMarkdown p, 
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span {
            color: #2C2520 !important;
        }
        
        .stRadio label p {
            color: #2C2520 !important;
            font-weight: 600;
        }
        
        .stTextArea textarea {
            background-color: #FFFFFF !important;
            color: #2C2520 !important;
            border: 1px solid #D4CFC4 !important;
            border-radius: 12px !important;
        }
        
        .stButton > button {
            background-color: #2C2520 !important;
            color: #F2F0E9 !important;
            border: none !important;
            border-radius: 50px !important;
            padding: 0.6rem 2rem !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
            opacity: 1 !important;
        }
        .stButton > button p { color: #F2F0E9 !important; }
        .stButton > button:hover {
            background-color: #A68A64 !important;
            transform: translateY(-2px);
        }
        
        .paper-card {
            background-color: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        
        .prec-card {
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
            border-left: 5px solid #A68A64;
            box-shadow: 0 2px 8px rgba(0,0,0,0.03);
            border: 1px solid #EAEAEA;
        }
        
        .streamlit-expanderHeader {
            background-color: #FFFFFF !important;
            color: #2C2520 !important;
            border: 1px solid #D4CFC4;
            border-radius: 8px;
        }
        .streamlit-expanderHeader:hover {
            background-color: #F2F0E9 !important;
            color: #A68A64 !important;
        }
        details[open] > summary {
            background-color: #FFFFFF !important;
            color: #2C2520 !important;
        }
        .streamlit-expanderHeader p {
            color: #2C2520 !important;
            font-weight: 600;
        }
        .streamlit-expanderContent {
            background-color: #FAFAF8 !important;
            border: 1px solid #D4CFC4;
            border-top: none;
            color: #2C2520 !important;
        }
        
        iframe {
            background-color: #F2F0E9 !important;
        }
    </style>
    """, unsafe_allow_html=True)

def render_citation_graph(results, strategy):
    nodes = []
    edges = []
    
    BG = "#F2F0E9"
    USER = "#C67B5C" 
    GOOD = "#7A9B76" 
    BAD = "#8B7D6B"  
    NEUTRAL = "#A8A29E" 
    
    nodes.append(Node(id="USER", label="Current Case", title="Your Case", size=40, color=USER, font={'color':'#2C2520', 'face':'Lato'}, shape="dot"))
    existing_nodes = {"USER"}
    
    for i, case in enumerate(results):
        cid = str(case['id'])
        if cid not in existing_nodes:
            is_fav = ("reverse" in str(case['decision']).lower() or "acquit" in str(case['decision']).lower())
            col = GOOD if (strategy=="defense" and is_fav) or (strategy=="prosecution" and not is_fav) else BAD
            
            lbl = case['title'][:15] + "..."
            
            nodes.append(Node(
                id=cid, 
                label=lbl, 
                title=case['title'], 
                size=25, 
                color=col, 
                font={'color':'#2C2520', 'size':12}
            ))
            edges.append(Edge(source="USER", target=cid, color="#A68A64", width=2))
            existing_nodes.add(cid)
            
            found_via_list = case.get('found_via', [])
            for source_case_name in found_via_list:
                source_id = f"SRC_{source_case_name.replace(' ', '_')}"
                if source_id not in existing_nodes:
                    nodes.append(Node(
                        id=source_id, 
                        label=source_case_name[:10]+"..", 
                        title=source_case_name,
                        size=15, 
                        color=NEUTRAL, 
                        shape="dot", 
                        font={'color':'#666', 'size':10}
                    ))
                    existing_nodes.add(source_id)
                edges.append(Edge(source=source_id, target=cid, color="#D6D1C9", width=1, type="curved"))

    config = Config(
        width="100%", 
        height=350, 
        directed=True, 
        physics=True, 
        hierarchical=False,
        backgroundColor=BG,
        nodeHighlightBehavior=True, 
        highlightColor="#A68A64",
    )
    return nodes, edges, config

def main():
    load_luxury_css()
    
    if "results" not in st.session_state: st.session_state.results = None
    if "analysis" not in st.session_state: st.session_state.analysis = None
    if "search_method" not in st.session_state: st.session_state.search_method = None
    if "case_input" not in st.session_state: st.session_state.case_input = ""

    with st.sidebar:
        st.markdown("### Configuration")
        strategy = st.radio("Strategy", ["Defense", "Prosecution"])
        strat_key = "defense" if "Defense" in strategy else "prosecution"
        st.markdown("---")
        st.markdown("### System Status")
        if get_driver(): st.success("Database Connected")
        else: st.error("Offline")
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.caption("Texas Legal Graph v1.0")

    st.markdown("<h1 style='text-align: center;'>Texas Legal Graph</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-style: italic; color: #6B5D54;'>Clarity Through Citation Networks</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Case Details")
    c1, c2 = st.columns([3, 1])
    with c1:
        input_text = st.text_area("Facts", height=120, placeholder="Describe the incident...", label_visibility="collapsed", key="widget_input", value=st.session_state.case_input)
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze = st.button("ANALYZE", use_container_width=True)
        if st.button("CLEAR", use_container_width=True):
            st.session_state.results = None
            st.session_state.analysis = None
            st.session_state.search_method = None
            st.session_state.case_input = ""
            st.rerun()

    if analyze and input_text:
        st.session_state.case_input = input_text
        with st.spinner("Analyzing jurisprudence..."):
            emb = get_embedding(input_text)
            if emb:
                # --- STRATEGIA A CASCATA (Fallback Silenzioso) ---
                
                results = graph_rag_search(emb, strat_key, apply_filter=True)
                method = "Gold"
                
                if not results:
                    results = graph_rag_search(emb, strat_key, apply_filter=False)
                    method = "Silver"
                
                if not results:
                    results = vector_only_search(emb)
                    method = "Bronze"
                
                st.session_state.results = results
                st.session_state.search_method = method
                
                if results:
                    st.session_state.analysis = generate_strategic_analysis(input_text, results, strat_key, method)
            else:
                st.error("Embedding API Error.")

    if st.session_state.results:
        st.markdown("---")
        
        # RIMOSSO IL BLOCCO ST.INFO CHE MOSTRAVA IL FALLBACK
        # L'interfaccia passa direttamente ai risultati

        st.markdown("### Strategic Analysis")
        st.markdown(f"<div class='paper-card'>{st.session_state.analysis}</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_graph, col_cards = st.columns([1, 1], gap="large")
        
        with col_graph:
            st.markdown("### Citation Map")
            n, e, c = render_citation_graph(st.session_state.results, strat_key)
            agraph(n, e, c)
            
        with col_cards:
            st.markdown(f"### Precedents ({len(st.session_state.results)})")
            for case in st.session_state.results:
                is_fav = ("reverse" in str(case['decision']).lower() or "acquit" in str(case['decision']).lower())
                accent = "#7A9B76" if (strat_key=="defense" and is_fav) else "#C67B5C"
                
                conf = int(case['relevance_score'] * 100) + (case['citation_count'] * 5)
                conf = min(conf, 99)
                
                st.markdown(f"""
                <div class='prec-card' style='border-left-color: {accent};'>
                    <div style='display:flex; justify-content:space-between;'>
                        <div style='font-family: Playfair Display; font-size: 1.1em; font-weight: bold; color: #2C2520;'>{case['title']}</div>
                        <span style='background:#E8E5DD; padding:2px 8px; border-radius:10px; font-size:0.8em; font-weight:bold;'>{conf}% Match</span>
                    </div>
                    <div style='margin-top: 5px; font-size: 0.9em; color: #666;'>{case['decision']}</div>
                    <div style='margin-top: 10px; font-style: italic; font-size: 0.85em; color: #444; line-height: 1.4;'>"{str(case.get('full_text',''))[:120]}..."</div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander(f"Read Full Opinion ({case['title']})"):
                    full_text = html.escape(case.get('full_text', 'No text available'))
                    st.markdown(f"""
                    <div style='height: 300px; overflow-y: auto; padding: 15px; background-color: #FFFFFF; border: 1px solid #EAEAEA; border-radius: 5px; font-family: Georgia, serif; font-size: 0.95em; line-height: 1.6; color: #333;'>
                        {full_text}
                    </div>
                    """, unsafe_allow_html=True)

    elif st.session_state.results is not None and len(st.session_state.results) == 0:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background-color: white; padding: 40px; border-radius: 12px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.05); border: 1px solid #E8E5DD;'>
            <div style='font-size: 50px;'>📭</div>
            <h3 style='color: #2C2520; margin: 15px 0; font-family: "Playfair Display", serif;'>No Relevant Cases Found</h3>
            <p style='color: #6B5D54; font-family: Lato; font-size: 1.1em;'>
                Even after expanding the search to semantic similarity, no cases were found matching this description.<br>
                <em>Tip: Try providing more specific legal keywords or factual details.</em>
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()