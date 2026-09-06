from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from graph import app as langgraph_app

app = FastAPI(title="Grounded Citation RAG API", description="Backend API for querying the EU AI Act.")

class QueryRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(request: QueryRequest):
    inputs = {"question": request.question}
    try:
        # Our graph.py already contains the fast-fail 429 rate limit handling.
        # It will return the structured API Error message if limits are hit.
        final_result = langgraph_app.invoke(inputs)
        return final_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
