"""
Trivial Baseline for CompanySupport.
- Intent: Always predicts majority class ("software_os").
- Escalation: Never escalates (auto_handle everything).
- Reply: Fixed canned template message.
"""

from src.models import AgentPrediction

class TrivialBaselineAgent:
    def __init__(self):
        self.majority_intent = "technical_issue"
        self.canned_reply = (
            "Thank you for reaching out to Enterprise Support. "
            "Please restart your device and visit https://company.com/support for more help."
        )

    def process_query(self, customer_text: str, query_id: str = "", **kwargs) -> AgentPrediction:
        return AgentPrediction(
            query_id=query_id,
            customer_text=customer_text,
            predicted_intent=self.majority_intent,
            intent_confidence=1.0,
            escalation_action="auto_handle",
            escalation_reason="default_rule",
            risk_level="low",
            generated_reply=self.canned_reply,
            retrieved_exemplars=[],
            signals=["trivial_baseline_default"],
            latency_ms=0.1
        )
