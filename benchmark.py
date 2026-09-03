import os
import sys
import random
import time
from neo4j import GraphDatabase
from tqdm import tqdm
import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Import unified credentials and embedding logic from api.index
from api.index import load_credentials, get_embedding

NEO4J_URI, NEO4J_USER, PWD, API_KEY = load_credentials()
NEO4J_VECTOR_INDEX = os.getenv("NEO4J_VECTOR_INDEX", "case-text-embeddings")

if not PWD or not API_KEY:
    print("Errore: credenziali mancanti (NEO4J_PASSWORD o GEMINI_API_KEY/GOOGLE_API_KEY).")
    exit(1)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, PWD))

def get_test_set(sample_size=40):
    """
    Recupera i casi che citano "Precedenti Autorevoli" (Authority Cases).
    Nel dataset benchmark curato da 50 casi, la condizione di in-degree >= 2 individua
    44 casi candidati (88% del corpus), offrendo una base statisticamente solida
    e selettiva senza dover allentare la soglia a >= 1.
    """
    print("--- Generazione Dataset di Test ---")
    
    # Query pulita senza commenti // per evitare errori di parsing
    query = """
        MATCH (c:CASE)-[:CITES]->(cited:CASE)
        WHERE (c)-[:HAS_TEXT]->() 
          AND c.offense IS NOT NULL 
          AND cited.offense IS NOT NULL
        
        WITH c, cited
        WHERE count{(cited)<-[:CITES]-()} >= 2
        
        WITH c, collect(DISTINCT cited) as actual_citations
        WHERE size(actual_citations) > 0
        
        MATCH (c)-[:HAS_TEXT]->(t:TEXT)
        RETURN 
            elementId(c) as id, 
            t.text as text, 
            c.offense as source_offense,
            [x in actual_citations | {id: elementId(x), offense: x.offense}] as ground_truth
        LIMIT 2000
    """
    with driver.session() as session:
        data = session.run(query).data()
    
    print(f"Casi candidati trovati: {len(data)}")
    
    if len(data) > sample_size:
        return random.sample(data, sample_size)
    return data

def retrieve_cases(embedding, original_id, use_graph=True):
    """
    Esegue la ricerca e restituisce ID e OFFENSE dei casi trovati.
    """
    index_name = NEO4J_VECTOR_INDEX
    
    if use_graph:
        # GRAPH RAG (Vettori su CASE -> Traversata)
        query = f"""
        CALL db.index.vector.queryNodes($index_name, 50, $embedding) 
        YIELD node AS anchorCase, score
        WHERE elementId(anchorCase) <> $original_id
        
        MATCH (anchorCase)-[:CITES*1..2]->(precedent:CASE)
        WHERE precedent.offense IS NOT NULL
        
        RETURN elementId(precedent) as id, precedent.offense as offense, sum(score) as score
        ORDER BY score DESC LIMIT 10
        """
    else:
        # VECTOR ONLY (Baseline)
        query = f"""
        CALL db.index.vector.queryNodes($index_name, 10, $embedding)
        YIELD node AS precedent, score
        WHERE elementId(precedent) <> $original_id AND precedent.offense IS NOT NULL
        
        RETURN elementId(precedent) as id, precedent.offense as offense, score
        ORDER BY score DESC LIMIT 10
        """
        
    with driver.session() as session:
        results = session.run(query, index_name=index_name, embedding=embedding, original_id=original_id)
        return [{"id": r["id"], "offense": r["offense"]} for r in results]

def calculate_score(ground_truth_list, retrieved_list):
    """
    Calcola lo score per un singolo caso.
    - 1.0 punti: Trovato l'ID esatto (Hard Match)
    - 0.5 punti: ID diverso, ma stesso Offense (Soft Match)
    - 0.0 punti: Nessuna corrispondenza
    """
    # Set di ID veri e Offense veri
    true_ids = {item['id'] for item in ground_truth_list}
    
    # Normalizziamo le stringhe offense (lowercase, strip) per il confronto
    true_offenses = {str(item['offense']).strip().lower() for item in ground_truth_list if item['offense']}
    
    max_points = 0.0
    
    for found in retrieved_list:
        found_id = found['id']
        found_offense = str(found['offense']).strip().lower() if found['offense'] else ""
        
        if found_id in true_ids:
            return 1.0 # JACKPOT: Trovato esattamente il caso citato!
        
        if found_offense in true_offenses:
            max_points = 0.5 # SOFT MATCH: Caso analogo trovato
            
    return max_points

def main():
    test_cases = get_test_set(40) # Testiamo su 40 casi rappresentativi
    print(f"Test Set: {len(test_cases)} casi pronti.\n")
    
    vector_scores = []
    graph_scores = []
    
    print("--- STARTING EVALUATION ---")
    
    for i, case in enumerate(tqdm(test_cases)):
        input_text = case['text'][:4000] # Simuliamo input utente
        emb = get_embedding(input_text)
        if not emb: continue
        
        # 1. Test Vector
        res_v = retrieve_cases(emb, case['id'], use_graph=False)
        score_v = calculate_score(case['ground_truth'], res_v)
        vector_scores.append(score_v)
        
        # 2. Test Graph
        res_g = retrieve_cases(emb, case['id'], use_graph=True)
        score_g = calculate_score(case['ground_truth'], res_g)
        graph_scores.append(score_g)

    # --- CALCOLO FINALE ---
    avg_vector = np.mean(vector_scores) * 100
    avg_graph = np.mean(graph_scores) * 100
    
    print("\n" + "="*50)
    print(f"RISULTATI FINALI (Soft Match Scoring)")
    print("="*50)
    print(f"Criterio: 1.0 pt per ID Esatto | 0.5 pt per Stesso Reato")
    print("-" * 50)
    print(f"🔹 Vector Search Score:  {avg_vector:.2f} / 100")
    print(f"🔸 Graph RAG Score:      {avg_graph:.2f} / 100")
    print("-" * 50)
    
    delta = avg_graph - avg_vector
    if delta > 0:
        print(f"VITTORIA: Il Grafo migliora la qualità del {delta:.2f} punti!")
    else:
        print(f"Pareggio o Sconfitta ({delta:.2f})")

if __name__ == "__main__":
    main()