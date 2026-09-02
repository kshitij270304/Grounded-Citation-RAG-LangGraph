import os
from typing import List
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

FAISS_INDEX_PATH = "faiss_index"

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

def retrieve_docs(query: str) -> List[Document]:
    """
    Takes a user query, queries the FAISS index + BM25, and returns the top 3 most relevant text chunks
    using Hybrid Search (EnsembleRetriever).
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    # allow_dangerous_deserialization=True is required in newer LangChain versions to load local FAISS index
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # 1. FAISS Retriever (Semantic Search)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 2. BM25 Retriever (Keyword Search)
    # Extract the raw documents already saved inside the FAISS docstore
    docs = list(vectorstore.docstore._dict.values())
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 3
    
    # 3. Ensemble Retriever (Hybrid Search)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]  # Give equal weight to keyword and semantic search
    )
    
    # Get hybrid results (Ensemble returns a ranked list, we slice to top 3)
    results = ensemble_retriever.invoke(query)
    return results[:3]

class AnswerWithCitation(BaseModel):
    answer: str = Field(description="The answer to the user's question.")
    citation: str = Field(description="The exact, word-for-word sentence used to form the answer. If the answer is not in the text, output 'Data not available'.")

def generate_answer(query: str, docs: List[Document]) -> AnswerWithCitation:
    """
    Uses LangChain's .with_structured_output() to force the LLM to return a Pydantic model.
    """
    # Initialize the LLM (using gemini-1.5-flash for speed/cost, but any compatible model works)
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
    structured_llm = llm.with_structured_output(AnswerWithCitation)

    context = "\n\n".join([doc.page_content for doc in docs])

    system_prompt = (
        "You are a strict compliance assistant. Answer the question using ONLY the provided context. "
        "You must extract the exact, word-for-word sentence you used to form your answer and place it in the citation field. "
        "If the answer is not in the text, output 'Data not available'."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context:\n{context}\n\nQuestion: {query}")
    ])

    chain = prompt | structured_llm
    result = chain.invoke({"context": context, "query": query})
    return result

def grade_citation(llm_output: AnswerWithCitation, docs: List[Document]) -> bool:
    """
    Anti-hallucination engine: takes the LLM's JSON output and checks if the string inside citation 
    actually exists within the raw text of the retrieved chunks.
    """
    import re
    retrieved_text = " ".join([doc.page_content for doc in docs])
    
    # Strip all non-alphanumeric characters (keep only letters and numbers)
    # This prevents PDF encoding errors (like '') and punctuation mismatches from causing false failures
    clean_text = re.sub(r'[^a-zA-Z0-9]', '', retrieved_text).lower()
    clean_citation = re.sub(r'[^a-zA-Z0-9]', '', llm_output.citation).lower()
    
    # Check if citation exists in the clean text
    if clean_citation in clean_text:
        return True
    
    # If the LLM correctly identified that the data is not available, consider the check passed
    if llm_output.citation == 'Data not available':
        return True
        
    return False

if __name__ == "__main__":
    # Quick test of the core functions
    test_query = "What is the purpose of this Regulation?"
    print(f"Testing with query: '{test_query}'\n")
    
    # 1. Retrieve
    print("1. Retrieving docs...")
    retrieved_docs = retrieve_docs(test_query)
    for i, doc in enumerate(retrieved_docs):
        print(f"  Chunk {i+1} length: {len(doc.page_content)}")
    
    # 2. Generate
    print("\n2. Generating answer...")
    answer_output = generate_answer(test_query, retrieved_docs)
    print(f"  Answer: {answer_output.answer}")
    print(f"  Citation: {answer_output.citation}")
    
    # 3. Grade
    print("\n3. Grading citation...")
    is_valid = grade_citation(answer_output, retrieved_docs)
    print(f"  Is citation valid? {is_valid}")
