import streamlit as st
from graph import app as langgraph_app

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

1. Hybrid Retrieval: It scans the legal text using both Semantic Search (FAISS) and Keyword Search (BM25) simultaneously.  
2. Strict Generation: The Gemini API analyzes the chunks and answers your question based only on the text.  
3. Autonomous Grader: A validation node ensures the citation is a perfect word-for-word match to the raw PDF. If it detects a hallucination, it routes the graph backwards and forces the AI to correct itself!
""")

# Chat input
query = st.chat_input("Ask a question about the EU AI Act (e.g. 'What is the purpose of this Regulation?')")

if query:
    # 1. Display User Question
    st.chat_message("user").write(query)
    
    # 2. Display Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Searching database and validating citations (this may take a few seconds)..."):
            inputs = {"question": query}
            
            # Run the compiled LangGraph application
            final_result = langgraph_app.invoke(inputs)
            
            # Extract the final answer and citation
            generation = final_result.get("generation", {})
            answer = generation.get("answer", "No answer generated.")
            citation = generation.get("citation", "No citation provided.")
            
            # Render the final answer in a standard text block
            st.write(answer)
            
            # Render the citation elegantly
            st.info(f"**Exact Citation:** {citation}")
