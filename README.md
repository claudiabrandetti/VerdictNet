# Gavel&Graph - AI Frontiers: LLM - Group 9
## Stare Decisis AI: Graph RAG for Texas Criminal Law

Project developed for the **AI Frontiers: LLM** course.

## Authors
* **Claudia Brandetti**
* **Denise Di Franza**
* **Anastasia Farinaro**
* **Daniele Sarcina**

## Link to the presentation
https://aifrontiersgroup6.my.canva.site/llm

> **Bridging the Semantic Gap in Legal Research with Hybrid Graph Retrieval-Augmented Generation.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Neo4j](https://img.shields.io/badge/Neo4j-Graph%20Database-orange)
![Gemini Extraction](https://img.shields.io/badge/Extraction-Gemini%202.5-4285F4)
![Gemini RAG](https://img.shields.io/badge/RAG-Gemini%202.0%20Flash-8E75B2)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)

An intelligent legal assistant that combines semantic search with legal citation network analysis.

## Overview

**Stare Decisis AI** is an intelligent legal assistant designed to navigate the complexity of the US Common Law system. Unlike traditional keyword-based search engines, which fail to capture legal nuance, this project utilizes a **Graph RAG (Retrieval-Augmented Generation)** architecture.

By transforming **7791 raw legal documents** (Texas Criminal Reports) into a Knowledge Graph, the system can identify relevant precedents based on **semantic similarity** via `text-embedding-004` and **legal authority** (Citation Network Analysis), generating strategic defense memos in real-time.

----

## Key Features

* **Hybrid Retrieval Engine:** Combines **Vector Search** (to find similar facts using `text-embedding-004`) with **Graph Traversal** (to find authoritative precedents cited by those cases).
* **Citation Network Analysis:** Traverses the graph (Depth 1-2) to uncover the "legal pillars" behind a case, identifying winning strategies (e.g., *Reversed* judgements).
* **Strategic Filtering:** Allows users to filter results based on the desired outcome (e.g., *Defense/Acquittal* vs. *Prosecution/Conviction*).
* **LLM-Driven Extraction (Gemini 2.5):** Utilizes the advanced reasoning of **Google Gemini 2.5** to extract structured metadata (Offense, Punishment, Decision, Conviction) from unstructured 19th-20th century texts with high precision.
* **Interactive Visualization:** Features a dynamic graph visualization to explore the connections between the user's case and historical precedents.

----

## Architecture

The system operates in two main phases: **The Builder** (Batch Processing with Gemini 2.5) and **The Assistant** (Inference with Gemini 2.0).

```mermaid
graph TD
    subgraph Data_Pipeline
        A[Raw JSON Data] -->|Parser| B(Data Cleaning)
        B -->|High-Fidelity Extraction| C{Gemini 2.5}
        C -->|Metadata| D[Neo4j Graph]
        C -->|Text| E[Vector Embeddings]
        E -->|text-embedding-004| D
    end
    subgraph Inference_Engine
        User[User Query] -->|text-embedding-004| Search[Hybrid Search]
        Search -->|Vector Sim| Anchor[Anchor Cases]
        Anchor -->|Graph Traversal| Prec[Precedents]
        Prec -->|Context Injection| Gen[Gemini 2.0 Flash]
        Gen --> Output[Strategic Memo]
    end
````
### LLM Engineering
We implemented a multi-model architecture to leverage the best strengths of each LLM version:

* **High-Fidelity Extraction (Gemini 2.5):** We used **Gemini 2.5** for Zero-Shot extraction of structured data from 19th-century legal texts. Its superior reasoning capabilities ensured accurate parsing of complex schema fields like `Offense`, `Punishment`, and `Decision`.
* **Grounded Generation (Gemini 2.0 Flash):** For the RAG phase, we utilized **Gemini 2.0 Flash** due to its speed and long-context window. By grounding the generation strictly in the retrieved graph context, we minimized legal hallucinations.
-----

##  Tech Stack

* **Language:** Python 3.10+
* **Database:** Neo4j (Graph DB + Vector Index)
* **AI Models:**
      * **Extraction:** Google Gemini 2.5
      * **RAG Engine:** Google Gemini 2.0 Flash
      * **Embeddings:** `text-embedding-004`
* **Frontend:** Streamlit & Streamlit-Agraph
* **Data Source:** [Case.law](https://case.law) (Harvard Caselaw Access Project)

-----

##  Project Structure

```bash
├── data_pipeline/
│   ├── case_law_crawler_tx.py       # Downloads raw cases
│   ├── json_to_csv_parser_tx.py     # Converts JSON to CSV
│   ├── llm_extraction_tx.py         # Extracts metadata via Gemini 2.5
│   └── create_embeddings_final.py   # Generates Vectors (text-embedding-004)
├── app.py                           # Main App (RAG Engine with Gemini 2.0 Flash)
├── requirements.txt                 # Dependencies
└── README.md                        # Documentation
```

-----

## Quick Start

### 1. Prerequisites

* Python 3.10+
* A running **Neo4j Database** (Desktop or AuraDB).
* A **Google Gemini API Key**.

### 2. Installation

Clone the repository:

```bash
git clone [https://github.com/denise-df/Gavel-Graph.git](https://github.com/denise-df/Gavel-Graph.git)
cd Gavel-Graph
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3\. Configuration

Create a `key.txt` file (or set environment variables) for your API keys:

```text
# Inside key.txt
YOUR_GEMINI_API_KEY
```

Create a `neo4j_pass.txt` for your database password.

### 4\. Running the App

Once the database is populated (see `data_pipeline` scripts), launch the interface:

```bash
streamlit run app.py
```
![Immagine WhatsApp 2025-11-30 ore 13 25 56_4684df20](https://github.com/user-attachments/assets/70cf8971-3a13-4e2e-ba61-6703fda7e7ea)
![Immagine WhatsApp 2025-11-30 ore 13 29 35_995768eb](https://github.com/user-attachments/assets/65264338-fc82-46f1-a209-40fd8010808f)


## Case Study: The "Hybrid" Advantage

To demonstrate the power of **Graph RAG**, we tested a complex query:

> **Query:** *"Can a conviction for burglary be sustained if the defendant entered the house but fell asleep without stealing anything?"*

| Method | Result | Why? |
| :--- | :--- | :--- |
| **Vector Search (Standard RAG)** | **Irrelevant** | Retrieved cases about *victims* sleeping during a robbery. The model matched the keywords "sleep" and "burglary" but missed the legal context of *intent*. |
| **Stare Decisis AI (Graph RAG)** | **Found Precedent** | Identified *McLemore v. State*, a key precedent where a conviction was **reversed** because the defendant was intoxicated/asleep, failing to prove "intent to commit theft." |

**Why it worked:** The system found a semantically similar case (Anchor) and traversed the citation graph to find the authoritative ruling that the Anchor relied upon.

-----

## Evaluation & Results

We conducted an ablation study comparing **Vector-Only Search** vs. **Graph RAG**.

* **Vector Search:** Efficient at finding textually similar descriptions but often retrieves legally irrelevant cases (e.g., finding *Burglary* cases with "sleeping victims" instead of "sleeping defendants").
* **Graph RAG:** Significantly improves relevance by retrieving the *authoritative cases* cited by the vector matches, successfully identifying procedural defenses (e.g., *McLemore v. State*) that standard search misses.

-----

*Disclaimer: This tool is for academic and educational purposes only and does not constitute professional legal advice.*
