import streamlit as st
import os
import requests
import json
from datetime import datetime

# Page config
st.set_page_config(page_title="Grounded Citation RAG")

# Custom CSS for a designer dark blue flare background
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at center, rgba(15, 50, 120, 0.4) 0%, #0e1117 80%) !important;
    }
</style>
""", unsafe_allow_html=True)

# UI Layout
st.title("Grounded Citation RAG System")
st.markdown("""
**Welcome to the EU AI Act Compliance Assistant!**

This system is powered by an advanced LangGraph Agentic Workflow designed for zero-hallucination legal compliance:

1. **HTML Scraping & Smart Chunking:** Data is pulled directly from live HTML and grouped by section headers.
2. **Hybrid Retrieval:** It scans the legal text using both Semantic Search (FAISS) and Keyword Search (BM25).
3. **Query Decomposition & Memory:** It remembers your previous questions and breaks down complex questions into sub-queries.
4. **Autonomous Grader:** A validation node ensures the citation is a perfect word-for-word match.
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "reasoning" in message and "citation" in message:
            with st.expander("View Agent Reasoning"):
                st.write(message["reasoning"])
            st.info(f"**Exact Citation:** {message['citation']}")

# Chat input
query = st.chat_input("Ask a question about the EU AI Act (e.g. 'What are the penalties?')")

if query:
    # 1. Display User Question
    st.chat_message("user").write(query)
    
    # 2. Display Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing history, decomposing query, and validating citations..."):
            
            api_url = os.environ.get("API_URL", "http://backend:8000/chat")
            
            # Prepare payload with chat history
            payload = {
                "question": query,
                "chat_history": st.session_state.messages
            }
            
            try:
                response = requests.post(api_url, json=payload)
                response.raise_for_status()
                final_result = response.json()
            except Exception as e:
                st.error(f"Backend API Error: {str(e)}")
                st.stop()
            
            # Extract the final answer and citation
            generation = final_result.get("generation", {})
            reasoning = generation.get("reasoning", "No reasoning provided.")
            answer = generation.get("answer", "No answer generated.")
            citation = generation.get("citation", "No citation provided.")
            
            # --- AUDIT TRAIL LOGGING ---
            audit_log = {
                "query": query,
                "reasoning": reasoning,
                "answer": answer,
                "citation": citation,
                "retrieved_chunks_count": len(final_result.get("documents", [])),
                "hallucination_retries": final_result.get("revision_count", 0),
                "is_valid": final_result.get("is_valid", False),
                "timestamp": datetime.now().isoformat()
            }
            
            audit_file = "audit_trail.json"
            if os.path.exists(audit_file):
                with open(audit_file, "r") as f:
                    try:
                        logs = json.load(f)
                    except json.JSONDecodeError:
                        logs = []
            else:
                logs = []
                
            logs.append(audit_log)
            with open(audit_file, "w") as f:
                json.dump(logs, f, indent=4)
            # ---------------------------
            
            # Render the reasoning in an expander
            with st.expander("View Agent Reasoning"):
                st.write(reasoning)
                
            # Render the final answer in a standard text block
            st.write(answer)
            
            # Render the citation elegantly
            st.info(f"**Exact Citation:** {citation}")
            
    # Add to session state
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "reasoning": reasoning,
        "citation": citation
    })
