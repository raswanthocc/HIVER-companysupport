"""
FastAPI Backend Server for Multi-Brand AI Customer Support Agent.
Powered by Agentic RAG, TWCS ChromaDB Vector Database, and Gemini LLM.
"""

import os
import sys
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.agent import CompanySupportAgent
from evaluation.llm_judge import RealLLMJudge

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Multi-Brand Agentic RAG Support API",
    description="Backend API powering the multi-brand customer support agent",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Agent
print("Initializing Multi-Brand Support Agent...")
agent = CompanySupportAgent()
print("Agent ready!")
print("Initializing LLM Judge...")
judge = RealLLMJudge()
if judge.is_configured():
    print("LLM Judge is configured and active.")
else:
    print("LLM Judge is NOT configured (missing API key).")


class ChatRequest(BaseModel):
    text: str
    brand: Optional[str] = "All Brands"

@app.get("/api/brands")
def get_brands():
    """Return all 108 distinct brands indexed in ChromaDB."""
    brands = agent.get_available_brands()
    return {"brands": brands}

@app.post("/api/chat")
def process_chat(req: ChatRequest):
    """Process incoming customer query through Agentic RAG pipeline."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text query cannot be empty.")
        
    try:
        pred = agent.process_query(
            customer_text=req.text.strip(),
            selected_brand=req.brand or "All Brands"
        )
        
        # Evaluate reply dynamically using LLM Judge
        judge_eval = judge.evaluate_reply(
            customer_text=pred.customer_text,
            generated_reply=pred.generated_reply,
            predicted_intent=pred.predicted_intent,
            predicted_action=pred.escalation_action,
            predicted_reason=pred.escalation_reason
        ) if judge.is_configured() else None
        
        return {
            "status": "success",
            "query_id": pred.query_id,
            "customer_text": pred.customer_text,
            "selected_brand": req.brand,
            "predicted_intent": pred.predicted_intent,
            "intent_confidence": pred.intent_confidence,
            "escalation_action": pred.escalation_action,
            "escalation_reason": pred.escalation_reason,
            "escalation_explanation": getattr(pred, "escalation_explanation", ""),
            "risk_level": pred.risk_level,
            "generated_reply": pred.generated_reply,
            "retrieved_evidence": pred.retrieved_evidence,
            "latency_ms": pred.latency_ms,
            "grounding_status": getattr(pred, "grounding_status", "agentic_rag_llm_synthesized"),
            "judge_evaluation": judge_eval
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount Static Files
static_dir = os.path.join(BASE_DIR, "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def read_root():
    """Serve the Custom Web App index page."""
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Custom Web App API is running. Create static/index.html to view UI."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
