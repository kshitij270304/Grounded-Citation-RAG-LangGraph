import streamlit as st
from graph import app as langgraph_app

# Page config
st.set_page_config(page_title="Grounded Citation RAG", page_icon="⚖️")

# UI Layout
st.title("Grounded Citation RAG System")
st.markdown("""
**Welcome to the EU AI Act Compliance Assistant!**

This system uses a **LangGraph agentic workflow** to answer your questions. It features an autonomous **self-correcting anti-hallucination loop**: 
1. It retrieves documents using FAISS.
2. It generates an answer using Gemini 3.6 Flash.
3. A grader verifies if the exact quote was used. If not, it loops back and forces the AI to correct itself before showing you the answer!
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
