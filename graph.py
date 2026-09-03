import os
from typing import List, Dict, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from nodes import retrieve_docs, AnswerWithCitation, grade_citation

load_dotenv()

# 1. Define the State
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    documents: List[Document]
    generation: dict
    revision_count: int
    is_valid: bool

# 2. Build the Nodes
def retrieve_node(state: GraphState):
    question = state["question"]
    print(f"\n---RETRIEVE DOCS---")
    print(f"Question: {question}")
    
    documents = retrieve_docs(question)
    
    # Initialize revision_count if not present
    revision_count = state.get("revision_count", 0)
    
    return {"documents": documents, "question": question, "revision_count": revision_count}

def validate_query_node(state: GraphState):
    question = state["question"]
    print(f"\n---VALIDATE QUERY (PRE-TOOL GUARDRAIL)---")
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a pre-retrieval guardrail. Reply EXACTLY with 'VALID' if the user's query is about AI, regulations, NIST, EU AI Act, risk management, or compliance. Reply EXACTLY with 'INVALID' if it is off-topic (e.g. recipes, coding, casual chat)."),
        ("human", "Query: {query}")
    ])
    chain = prompt | llm
    
    import time
    is_valid = True
    for attempt in range(3):
        try:
            res = chain.invoke({"query": question}).content.strip().upper()
            is_valid = "INVALID" not in res
            break
        except Exception as e:
            if "429" in str(e) or "RateLimit" in str(e) or "503" in str(e):
                print(f"API busy or Rate Limit hit in validate. Sleeping 20s... (Attempt {attempt+1}/3)")
                time.sleep(20)
            else:
                break
                
    if not is_valid:
        print("[GUARDRAIL TRIGGERED] Query is off-topic. Blocking retrieval.")
        return {"generation": {"reasoning": "The pre-tool guardrail detected an off-topic query. Retrieval was bypassed to save tokens.", "answer": "I can only answer questions related to the EU AI Act and NIST AI RMF.", "citation": "Data not available"}}
    else:
        print("[GUARDRAIL PASSED] Query is on-topic. Proceeding to retrieval.")
        return {}

def route_after_validate(state: GraphState):
    if state.get("generation"):
        # Guardrail triggered and set a canned response
        return "end"
    return "retrieve"

def generate_node(state: GraphState):
    print("\n---GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]
    revision_count = state.get("revision_count", 0)
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
    structured_llm = llm.with_structured_output(AnswerWithCitation)
    
    context = "\n\n".join([doc.page_content for doc in documents])
    
    system_prompt = (
        "You are a strict compliance assistant. Answer the question using ONLY the provided context. "
        "You are allowed to make obvious logical inferences (such as recognizing that a phrase followed by an acronym in parentheses defines that acronym). "
        "You must extract the exact, word-for-word snippet or sentence you used to form your answer and place it in the citation field. "
        "If the answer cannot be reasonably inferred from the text, output 'Data not available'."
    )
    
    if revision_count > 0:
        print(f"Revision Count is {revision_count}. Injecting hallucination warning...")
        system_prompt += "\n\nWarning: Your previous citation was hallucinated. You must pull the exact text from the documents."
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context:\n{context}\n\nQuestion: {query}")
    ])
    
    chain = prompt | structured_llm
    
    import time
    result = None
    for attempt in range(4):
        try:
            result = chain.invoke({"context": context, "query": question})
            break
        except Exception as e:
            if "429" in str(e) or "RateLimit" in str(e) or "503" in str(e):
                print(f"API busy or Rate Limit hit in generate. Sleeping 30s... (Attempt {attempt+1}/4)")
                time.sleep(30)
            else:
                raise e
                
    if not result:
        # Failsafe if it completely fails 4 times
        result = AnswerWithCitation(reasoning="API completely overloaded after retries.", answer="API Error: Google Gemini is currently overloaded. Please try again later.", citation="Data not available")
    
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
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("grade", grade_node)

# Set the flow
workflow.set_entry_point("validate")
workflow.add_conditional_edges(
    "validate",
    route_after_validate,
    {
        "retrieve": "retrieve",
        "end": END
    }
)
workflow.add_edge("retrieve", "generate")
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

if __name__ == "__main__":
    # Test the LangGraph workflow
    inputs = {"question": "What does the acronym TFEU stand for?"}
    
    print("Starting LangGraph execution...\n")
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node outputs
            pass
            
    print("\nFinal Graph State:")
    # We don't need persistent checkpointer here just simple run
    
    # Another way to print final output
    final_result = app.invoke(inputs)
    print("\nFinal Generation:")
    print(final_result["generation"])
