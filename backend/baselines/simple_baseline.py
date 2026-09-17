"""
Simple Baseline for CompanySupport.
- Intent: Standard word-level TF-IDF + Logistic Regression.
- Escalation: Basic naive keyword matching (no confidence gating, no distress/exhaustion detection).
- Reply: 1-Nearest Neighbor historical agent reply copied verbatim.
"""

from typing import List, Dict, Any
from src.models import AgentPrediction
from src.classifier import SimpleClassifier
from src.retriever import ResolutionRetriever

SIMPLE_ESCALATE_KEYWORDS = ["dm", "password", "Account ID", "broken", "refund"]

class SimpleBaselineAgent:
    def __init__(self, classifier: SimpleClassifier, retriever: ResolutionRetriever):
        self.classifier = classifier
        self.retriever = retriever

    def process_query(self, customer_text: str, query_id: str = "", **kwargs) -> AgentPrediction:
        text_lower = customer_text.lower()
        
        # 1. Simple intent
        intent_pred = self.classifier.predict(customer_text)

        # 2. Naive keyword escalation
        should_escalate = any(kw in text_lower for kw in SIMPLE_ESCALATE_KEYWORDS)
        if should_escalate:
            action = "escalate"
            reason = "keyword_match"
        else:
            action = "auto_handle"
            reason = "no_keyword_match"

        # 3. 1-NN retrieval
        retrieved = self.retriever.retrieve(customer_text, intent_pred.intent, top_k=1)
        if retrieved.historical_exemplars:
            reply = retrieved.historical_exemplars[0]["agent_text"]
        else:
            reply = "We are here to help. What seems to be the issue?"

        return AgentPrediction(
            query_id=query_id,
            customer_text=customer_text,
            predicted_intent=intent_pred.intent,
            intent_confidence=intent_pred.confidence,
            escalation_action=action,
            escalation_reason=reason,
            risk_level="medium" if should_escalate else "low",
            generated_reply=reply,
            retrieved_exemplars=[reply],
            signals=["simple_keyword_escalation"],
            latency_ms=1.5
        )
