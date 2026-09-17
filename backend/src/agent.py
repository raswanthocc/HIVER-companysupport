"""
End-to-End Multi-Brand Orchestrator for Support AI Agent.
"""

import time
from typing import Optional, List, Dict, Any

from src.models import CustomerQuery, AgentPrediction, IntentPrediction, EscalationDecision
from src.classifier import CalibratedIntentClassifier
from src.escalation import EscalationEngine
from src.retriever import ResolutionRetriever
from src.generator import GroundedResponseGenerator

class CompanySupportAgent:
    def __init__(
        self,
        classifier: Optional[CalibratedIntentClassifier] = None,
        escalation_engine: Optional[EscalationEngine] = None,
        retriever: Optional[ResolutionRetriever] = None,
        generator: Optional[GroundedResponseGenerator] = None
    ):
        self.classifier = classifier or CalibratedIntentClassifier()
        self.escalation_engine = escalation_engine or EscalationEngine()
        self.retriever = retriever or ResolutionRetriever()
        self.generator = generator or GroundedResponseGenerator()

    def get_available_brands(self) -> List[str]:
        return self.retriever.get_available_brands()

    def process_query(self, customer_text: str, query_id: str = "", selected_brand: str = "All Brands") -> AgentPrediction:
        t0 = time.perf_counter()

        # Step 1: Classify intent
        intent_pred = self.classifier.predict(customer_text, brand=selected_brand)

        # Step 2: Decide escalation and state reason (dynamic handoff rules)
        escalation = self.escalation_engine.evaluate(customer_text, intent_pred)

        # Step 3: Retrieve historical resolutions filtered by selected brand
        retrieved = self.retriever.retrieve(customer_text, brand=selected_brand, top_k=3)

        # Step 4: Generate grounded reply using Gemini LLM
        reply = self.generator.generate(customer_text, intent_pred, escalation, retrieved, selected_brand=selected_brand)

        latency = (time.perf_counter() - t0) * 1000.0

        exemplars = [e.get("agent_text", "") for e in retrieved.historical_exemplars[:2]]
        evidence = [dict(e) for e in retrieved.historical_exemplars[:3]]

        return AgentPrediction(
            query_id=query_id,
            customer_text=customer_text,
            predicted_intent=intent_pred.intent,
            intent_confidence=intent_pred.confidence,
            escalation_action=escalation.action,
            escalation_reason=escalation.reason,
            risk_level=escalation.risk_level,
            generated_reply=reply,
            retrieved_exemplars=exemplars,
            retrieved_evidence=evidence,
            signals=escalation.signals_detected,
            latency_ms=round(latency, 2),
            escalation_explanation=getattr(escalation, "explanation", ""),
            grounding_status=getattr(retrieved, "grounding_status", "agentic_rag_llm_synthesized")
        )
