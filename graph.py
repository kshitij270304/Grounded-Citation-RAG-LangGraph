import os
from typing import List, Dict, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from nodes import retrieve_faiss_docs, retrieve_bm25_docs, interleave_docs, AnswerWithCitation, grade_citation

load_dotenv()

# 1. Define the State
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    chat_history: List[Dict]
    sub_queries: List[str]
    faiss_docs: List[Document]
    bm25_docs: List[Document]
    documents: List[Document]
    generation: dict
    revision_count: int
    is_valid: bool

# 2. Build the Nodes

class SubQueries(BaseModel):
    queries: List[str] = Field(description="List of search queries")

def validate_query_node(state: GraphState):
    question = state["question"]
    print(f"\n---VALIDATE QUERY (PRE-TOOL GUARDRAIL)---")
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a pre-retrieval guardrail. Reply EXACTLY with 'VALID' if the user's query is about AI, regulations, NIST, EU AI Act, risk management, or compliance. Reply EXACTLY with 'INVALID' if it is off-topic (e.g. recipes, coding, casual chat)."),
        ("human", "Query: {query}")
    ])
    chain = prompt | llm
    
    try:
        res = chain.invoke({"query": question}).content.strip().upper()
        is_valid = "INVALID" not in res
    except Exception as e:
        if "429" in str(e) or "Quota" in str(e) or "RateLimit" in str(e):
            return {"generation": {"reasoning": "Google Gemini API rate limit exceeded during pre-tool validation.", "answer": "API Error: You have hit the Gemini free-tier rate limit (15 requests per minute). Please wait 60 seconds and try again.", "citation": "Data not available"}}
        is_valid = True # Failsafe
                
    if not is_valid:
        print("[GUARDRAIL TRIGGERED] Query is off-topic. Blocking retrieval.")
        return {"generation": {"reasoning": "The pre-tool guardrail detected an off-topic query. Retrieval was bypassed to save tokens.", "answer": "I can only answer questions related to the EU AI Act and NIST AI RMF.", "citation": "Data not available"}}
    else:
        print("[GUARDRAIL PASSED] Query is on-topic. Proceeding to retrieval.")
        return {}

def route_after_validate(state: GraphState):
    if state.get("generation"):
        return "end"
    return "decompose"

def decompose_node(state: GraphState):
    question = state["question"]
    chat_history = state.get("chat_history", [])
    print(f"\n---DECOMPOSE QUERY---")
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
    structured_llm = llm.with_structured_output(SubQueries)
    
    history_text = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in chat_history[-4:]]) # Only use last 4 messages for context
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant. Analyze the user's latest question and the chat history. "
                   "If the question is complex, break it down into up to 3 simpler search queries. "
                   "If it refers to past context (like 'it' or 'they'), resolve the pronouns into clear search queries. "
                   "If the question is simple, just return it as a single search query."),
        ("human", "Chat History:\n{history}\n\nLatest Question: {query}")
    ])
    
    chain = prompt | structured_llm
    
    try:
        res = chain.invoke({"history": history_text, "query": question})
        sub_queries = res.queries
    except Exception as e:
        sub_queries = [question] # Failsafe
        
    print(f"Sub-queries generated: {sub_queries}")
    return {"sub_queries": sub_queries}

def dispatch_node(state: GraphState):
    print("\n---DISPATCH TO PARALLEL RETRIEVAL---")
    return {}

def faiss_node(state: GraphState):
    sub_queries = state.get("sub_queries", [state["question"]])
    print(f"\n---RETRIEVE FAISS DOCS---")
    faiss_docs = []
    for q in sub_queries:
        faiss_docs.extend(retrieve_faiss_docs(q))
    return {"faiss_docs": faiss_docs}

def bm25_node(state: GraphState):
    sub_queries = state.get("sub_queries", [state["question"]])
    print(f"\n---RETRIEVE BM25 DOCS---")
    bm25_docs = []
    for q in sub_queries:
        bm25_docs.extend(retrieve_bm25_docs(q))
    return {"bm25_docs": bm25_docs}

def merge_node(state: GraphState):
    print(f"\n---MERGE RETRIEVED DOCS---")
    faiss_docs = state.get("faiss_docs", [])
    bm25_docs = state.get("bm25_docs", [])
    documents = interleave_docs(faiss_docs, bm25_docs)
    
    # Initialize revision_count if not present
    revision_count = state.get("revision_count", 0)
    
    return {"documents": documents, "revision_count": revision_count}


def generate_node(state: GraphState):
    print("\n---GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]
    chat_history = state.get("chat_history", [])
    revision_count = state.get("revision_count", 0)
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
    structured_llm = llm.with_structured_output(AnswerWithCitation)
    
    context = "\n\n".join([doc.page_content for doc in documents])
    history_text = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in chat_history[-4:]])
    
    system_prompt = (
        "You are a strict compliance assistant. Answer the question using ONLY the provided context. "
        "You may use the chat history to understand what the user is asking about if they use pronouns like 'it'. "
        "You must extract the exact, word-for-word snippet or sentence you used to form your answer and place it in the citation field. "
        "If the answer cannot be reasonably inferred from the text, output 'Data not available'."
    )
    
    if revision_count > 0:
        print(f"Revision Count is {revision_count}. Injecting hallucination warning...")
        system_prompt += "\n\nWarning: Your previous citation was hallucinated. You must pull the exact text from the documents."
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Chat History:\n{history}\n\nContext:\n{context}\n\nQuestion: {query}")
    ])
    
    chain = prompt | structured_llm
    
    try:
        result = chain.invoke({"history": history_text, "context": context, "query": question})
    except Exception as e:
        if "429" in str(e) or "Quota" in str(e) or "RateLimit" in str(e):
            result = AnswerWithCitation(reasoning="Google Gemini API rate limit exceeded during generation.", answer="API Error: You have hit the Gemini free-tier rate limit (15 requests per minute). Please wait 60 seconds and try again.", citation="Data not available")
        else:
            result = AnswerWithCitation(reasoning=f"API Error: {str(e)}", answer="API Error: An unexpected error occurred.", citation="Data not available")
    
    print(f"Generated Reasoning: {result.reasoning}")
    print(f"Generated Answer: {result.answer}")
    print(f"Generated Citation: {result.citation}")
    
    generation_dict = {"reasoning": getattr(result, 'reasoning', ''), "answer": result.answer, "citation": result.citation}
    
    return {"generation": generation_dict}

def grade_node(state: GraphState):
    print("\n---GRADE CITATION---")
    documents = state["documents"]
    generation = state["generation"]
    revision_count = state.get("revision_count", 0)
    
    # Reconstruct the Pydantic model for our grader function
    llm_output = AnswerWithCitation(reasoning=generation.get("reasoning", ""), answer=generation["answer"], citation=generation["citation"])
    is_valid = grade_citation(llm_output, documents)
    
    if not is_valid:
        print("[FAILED] CITATION FAILED GRADING. Incrementing revision count.")
        return {"is_valid": False, "revision_count": revision_count + 1}
    else:
        print("[PASSED] CITATION PASSED GRADING.")
        return {"is_valid": True, "revision_count": revision_count}

# 3. Wire the Conditional Edge
def route_after_grade(state: GraphState):
    if state.get("is_valid"):
        print("---ROUTING: FINISHED---")
        return "end"
    elif state.get("revision_count", 0) >= 3:
        print("---ROUTING: MAX RETRIES REACHED. ENDING---")
        return "end"
    else:
        print("---ROUTING: RETRYING GENERATION---")
        return "generate"

# Build the Graph
workflow = StateGraph(GraphState)

workflow.add_node("validate", validate_query_node)
workflow.add_node("decompose", decompose_node)
workflow.add_node("dispatch", dispatch_node)
workflow.add_node("faiss_node", faiss_node)
workflow.add_node("bm25_node", bm25_node)
workflow.add_node("merge", merge_node)
workflow.add_node("generate", generate_node)
workflow.add_node("grade", grade_node)

# Set the flow
workflow.set_entry_point("validate")
workflow.add_conditional_edges(
    "validate",
    route_after_validate,
    {
        "decompose": "decompose",
        "end": END
    }
)
workflow.add_edge("decompose", "dispatch")

# Parallel fan-out
workflow.add_edge("dispatch", "faiss_node")
workflow.add_edge("dispatch", "bm25_node")

# Parallel fan-in
workflow.add_edge("faiss_node", "merge")
workflow.add_edge("bm25_node", "merge")

workflow.add_edge("merge", "generate")
workflow.add_edge("generate", "grade")
workflow.add_conditional_edges(
    "grade",
    route_after_grade,
    {
        "end": END,
        "generate": "generate"
    }
)

# Compile
app = workflow.compile()
