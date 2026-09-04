# Grounded Citation RAG (EU AI Act Compliance Assistant)

**Live Demo:** [https://citation-rag.streamlit.app/](https://citation-rag.streamlit.app/)

## Overview
Grounded Citation RAG is an AI-powered legal compliance assistant tailored for querying the EU AI Act. It utilizes an advanced agentic workflow to ensure high-fidelity, zero-hallucination responses by grounding every answer with an exact, word-for-word citation from the source legal document.

## Architecture & Workflow

```mermaid
flowchart TD
    %% Styling Classes
    classDef start_end fill:#f96,stroke:#333,stroke-width:2px;
    classDef function fill:#bbf,stroke:#333,stroke-width:1px;
    classDef node fill:#bfb,stroke:#333,stroke-width:2px;
    classDef conditional fill:#fdd,stroke:#333,stroke-width:2px;
    classDef storage fill:#eee,stroke:#333,stroke-dasharray: 5 5;

    Start([User Inputs Query via Streamlit app.py]):::start_end
    
    subgraph LangGraph State Machine ["LangGraph Workflow (graph.py)"]
        
        %% Step 1: Validation Node
        N1["1. validate_query_node()"]:::node
        Start --> N1
        F1[["invoke(Gemini 3.8)"]]:::function
        N1 -.->|Checks if query is AI/Law related| F1
        
        %% Step 2: Route after validate
        C1{"2. route_after_validate()"}:::conditional
        N1 --> C1
        C1 -->|INVALID: route to end| End1([Bypass: Return 'Off-Topic' Error]):::start_end
        
        %% Step 3: Retrieval Node
        N2["3. retrieve_node()"]:::node
        C1 -->|VALID: route to retrieve| N2
        F2[["retrieve_docs() in nodes.py"]]:::function
        N2 -.-> F2
        F2 -.->|Semantic Search| FAISS[(FAISS Index)]:::storage
        F2 -.->|Keyword Search| BM25[(BM25 Retriever)]:::storage
        
        %% Step 4: Generation Node
        N3["4. generate_node()"]:::node
        N2 --> N3
        F3[["invoke(Gemini 3.8) w/ AnswerWithCitation Schema"]]:::function
        N3 -.->|Sends Context & Prompt| F3
        
        %% Step 5: Grade Node
        N4["5. grade_node()"]:::node
        N3 --> N4
        F4[["grade_citation() in nodes.py"]]:::function
        N4 -.-> F4
        F4 -.->|Regex string match of citation inside chunks| CheckValid{Is exact match?}
        
        %% Step 6: Route after grade
        C2{"6. route_after_grade()"}:::conditional
        N4 --> C2
        
        %% Feedback loop
        C2 -->|"is_valid == False & retries < 3<br>(Route to generate)"| N3
        C2 -->|"is_valid == True OR max retries<br>(Route to end)"| End2([Return Final Generation State]):::start_end
    end
    
    subgraph Frontend & Observability ["app.py"]
        End2 --> UI["Render Reasoning, Answer & Citation in UI"]:::function
        UI --> Audit["Save query & metrics to audit_trail.json"]:::storage
    end
```

## How It Works
The system is built on a **LangGraph Agentic Workflow** and follows a strict, self-correcting pipeline:
1. **Hybrid Retrieval:** Scans legal texts using both Semantic Search (FAISS) and Keyword Search (BM25) simultaneously for highly relevant context gathering.
2. **Strict Generation:** Utilizes the Google Gemini API to analyze the retrieved chunks and formulate an answer strictly based on the text.
3. **Autonomous Grader (Self-Correction):** A validation node ensures the generated citation is a perfect word-for-word match to the original PDF. If a hallucination or mismatch is detected, the graph routes backward, prompting the AI to correct itself (up to a defined retry limit) before finalizing the output.

## Why It Is Developed
Large Language Models (LLMs) are prone to hallucinations, which is unacceptable in legal, medical, and compliance domains where precision is critical. This project was developed to demonstrate a robust architecture that strictly binds the AI to the provided documents, ensuring trustworthy, verifiable, and exact answers without making up information.

## Where It Can Be Used
- **Legal & Compliance:** Analyzing complex regulations (like the EU AI Act), contracts, and corporate policies.
- **Healthcare & Medicine:** Querying medical guidelines, clinical trial protocols, and patient records.
- **Finance & Audit:** Reviewing financial reports, regulatory filings, and tax laws.
- **Enterprise Knowledge Base:** Internal document search where accuracy and source verification are paramount.

## Future Scopes
- **Multi-Document Support:** Expanding the RAG capabilities to handle vast libraries of interconnected documents across various domains.
- **Advanced Graph Capabilities:** Introducing multi-agent workflows for summarization, contradictory clause detection, and comparative analysis.
- **Dynamic Data Ingestion:** Allowing users to upload their own custom PDFs/documents directly via the UI for on-the-fly grounded RAG processing.
- **Extended Evaluation Metrics:** Integrating LLM-as-a-judge metrics for assessing the fluency, helpfulness, and comprehensiveness of the answers beyond exact citation matching.
