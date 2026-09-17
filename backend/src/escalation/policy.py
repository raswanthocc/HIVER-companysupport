"""
Multi-Signal Escalation Policy Engine for CompanySupport.
Evaluates multi-signal risk layers including:
- PII / credential exposure
- Hardware safety hazards & physical damage
- Financial disputes & billing
- Troubleshooting exhaustion ('already tried restarting')
- Customer frustration & churn signals
- Classifier confidence gating
- Explicit human-readable reason alignment
"""

import os
import re
from typing import List, Tuple, Dict, Any, Optional
from src.models import EscalationDecision, IntentPrediction

# Regular expressions for PII and sensitive customer identifiers
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")
RE_IMEI = re.compile(r"\b\d{15}\b")
RE_SERIAL = re.compile(r"\b[A-Z0-9]{10,12}\b")
RE_ORDER_NO = re.compile(r"\b(W\d{8,10}|[0-9]{10,12})\b", re.IGNORECASE)

class EscalationEngine:
    """
    Tiered multi-signal escalation policy engine.
    Ensures zero dangerous false auto-handles (FAHR minimization)
    and returns explicit machine-readable reasons and specific evidence-based explanations.
    """
    def __init__(self, confidence_threshold: float = 0.50):
        self.confidence_threshold = confidence_threshold
        self.handoff_rules = self._load_handoff_rules()

    def _load_handoff_rules(self) -> Dict[str, Any]:
        rules_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rules", "handoff_rules.json")
        if os.path.exists(rules_path):
            try:
                import json
                with open(rules_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def evaluate(
        self,
        customer_text: str,
        intent_pred: IntentPrediction,
        retrieval_similarity: Optional[float] = None
    ) -> EscalationDecision:
        text_lower = customer_text.lower()
        signals = []

        # Check Dynamic Handoff Rules from handoff_rules.json
        if self.handoff_rules and "handoff_triggers" in self.handoff_rules:
            for trigger in self.handoff_rules["handoff_triggers"]:
                keywords = trigger.get("keywords", [])
                if any(kw in text_lower for kw in keywords):
                    signals.append(f"handoff_rule_triggered_{trigger.get('category')}")
                    explanation = trigger.get("explanation", "Human handoff rule triggered.")
                    signals.append(explanation)
                    return EscalationDecision(
                        action="escalate",
                        reason=trigger.get("category", "human_handoff"),
                        confidence=0.99,
                        risk_level=trigger.get("risk_level", "HIGH").lower(),
                        signals_detected=signals,
                        explanation=explanation
                    )

        # Regex for raw credit card number (13 to 19 digits)
        has_cc_digits = bool(re.search(r"\b(?:\d[ -]*?){13,19}\b", customer_text))

        # Layer 1: PII & Credential Exposure / Account Security Detection
        has_pii = bool(RE_EMAIL.search(customer_text) or RE_IMEI.search(customer_text) or has_cc_digits)
        sec_patterns = [
            r"\baccount\s*id\b", r"\bpassword\w*\b", r"\bpasscode\w*\b", r"\block(?:ed|out)?\b",
            r"\b2fa\b", r"\btwo-factor\b", r"\bverification\s*code\b", r"\bsecurity\s*questions?\b",
            r"\bhack(?:ed)?\b", r"\bbreach\w*\b", r"\bunauthoriz\w*\b", r"\bstolen\b",
            r"\bcredential\w*\b", r"\bcloud\b", r"\bsign\s*in\b", r"\blogin\b",
            r"\btrusted\s*number\b", r"\bactivation\s*lock\b", r"\breset\s*password\b"
        ]
        has_creds = any(re.search(p, text_lower) for p in sec_patterns)
        account_security_intents = [
            "verification_identity", "unauthorized_fraudulent_transaction", "account_security"
        ]
        
        # Only escalate if actual PII is found, or it's a strict fraud/verification issue.
        if has_pii or intent_pred.intent in account_security_intents:
            signals.append("pii_or_credentials_detected")
            signals.append("Escalated because inquiry involves account credentials, authentication, or security compromise.")
            return EscalationDecision(
                action="escalate",
                reason="requires_pii_or_dm",
                confidence=0.98,
                risk_level="high",
                signals_detected=signals,
                explanation="Escalated because inquiry involves account credentials, authentication, or security compromise."
            )

        # Layer 2: Hardware Physical Safety Hazards & Defects
        hw_patterns = [
            r"\bcrack\w*\b", r"\bshatter\w*\b", r"\bswoll\w*\b", r"\bbulg\w*\b",
            r"\bbroken\b", r"\bbroke\b", r"\boverheat\w*\b", r"\bspark\w*\b", r"\bsmoke\b",
            r"\bcharging\s*port\b", r"\blightning\s*port\b", r"\bheadphone\s*jack\b",
            r"\bhardware\b", r"\bphysical\w*\b", r"\bwater\s*damage\b", r"\bliquid\s*damage\b",
            r"\bgenius\s*bar\b", r"\brepair\w*\b", r"\bdrop(?:ped)?\b", r"\bpopped\b",
            r"\bcrackl\w*\b", r"\bburn(?:ed)?\b", r"\bscarred\b", r"\bgot\s*fixed\b",
            r"\bbattery\s*expanding\b", r"\bscreen\s*(?:broken|lift|crack|shatter|unresponsive)\b"
        ]
        is_hardware_hazard = any(re.search(p, text_lower) for p in hw_patterns)
        if is_hardware_hazard or intent_pred.intent in ["product_service_issue", "hardware_defect"]:
            signals.append("physical_hardware_or_safety_hazard")
            signals.append("Escalated because customer reports possible physical device damage or safety hazard.")
            risk = "critical" if any(w in text_lower for w in ["swollen", "smoke", "expanding", "sparking", "burn"]) else "medium"
            return EscalationDecision(
                action="escalate",
                reason="hardware_safety_repair",
                confidence=0.95,
                risk_level=risk,
                signals_detected=signals,
                explanation="Escalated because customer reports possible physical device damage or safety hazard."
            )

        # Layer 3: Financial Disputes & Subscription Billing
        billing_patterns = [
            r"\bunauthorized\s*charge\b", r"\bdispute\b", r"\bfraud\w*\b", r"\bstolen\b"
        ]
        billing_intents = ["unauthorized_fraudulent_transaction"]
        is_billing_dispute = any(re.search(p, text_lower) for p in billing_patterns)
        if is_billing_dispute or intent_pred.intent in billing_intents:
            signals.append("financial_transaction_dispute")
            signals.append("Escalated because inquiry involves an unauthorized charge, refund dispute, or financial transaction.")
            return EscalationDecision(
                action="escalate",
                reason="financial_dispute",
                confidence=0.92,
                risk_level="medium",
                signals_detected=signals,
                explanation="Escalated because inquiry involves an unauthorized charge, refund dispute, or financial transaction."
            )

        # Layer 4: Repeated Troubleshooting Failure / Customer Exhaustion
        exhaustion_phrases = [
            "already tried", "already restarted", "already rebooted", "still not working",
            "nothing worked", "wasted hours", "same issue after restart", "restart my phone at least",
            "tried restarting", "done all the steps", "restarted multiple times", "nothing works",
            "tried resetting", "did all of that", "reset my settings", "still persists after restart"
        ]
        is_exhausted = any(p in text_lower for p in exhaustion_phrases)
        if is_exhausted:
            signals.append("troubleshooting_exhausted")
            signals.append("Escalated because customer reports repeated troubleshooting attempts with no resolution.")
            return EscalationDecision(
                action="escalate",
                reason="repeated_failure_exhausted",
                confidence=0.90,
                risk_level="medium",
                signals_detected=signals,
                explanation="Escalated because customer reports repeated troubleshooting attempts with no resolution."
            )

        # Layer 5: Severe Customer Frustration / Churn Signals
        churn_patterns = [
            "done with company", "switching to competitor", "worst company", "hate company",
            "useless support", "terrible service", "boycott", "never buying company again"
        ]
        is_high_churn = any(term in text_lower for term in churn_patterns)
        if is_high_churn:
            signals.append("high_churn_risk")
            signals.append("Escalated because customer exhibits severe frustration or urgent churn risk.")
            return EscalationDecision(
                action="escalate",
                reason="high_churn_frustration",
                confidence=0.88,
                risk_level="high",
                signals_detected=signals,
                explanation="Escalated because customer exhibits severe frustration or urgent churn risk."
            )

        # Layer 6: Classifier Confidence Safety Fallback Gate
        safety_critical_intents = [
            "product_service_issue", "account_access_login", "account_problem", "password_reset",
            "verification_identity", "unauthorized_fraudulent_transaction", "payment_billing_issue",
            "refund_request", "hardware_defect", "account_security", "billing_subscription"
        ]
        safety_threshold = 0.70 if intent_pred.intent in safety_critical_intents else self.confidence_threshold
        if intent_pred.confidence < safety_threshold:
            signals.append(f"low_classifier_confidence_{intent_pred.confidence:.2f}")
            signals.append("Escalated because classifier confidence is low on a safety-sensitive inquiry.")
            return EscalationDecision(
                action="escalate",
                reason="low_confidence_fallback",
                confidence=0.75,
                risk_level="low",
                signals_detected=signals,
                explanation="Escalated because classifier confidence is low on a safety-sensitive inquiry."
            )

        # Layer 7: General Feedback / Non-Actionable Routing
        if intent_pred.intent in ["complaint_poor_experience", "general_information", "general_feedback"]:
            signals.append("general_feedback_routing")
            return EscalationDecision(
                action="auto_handle",
                reason="feedback_portal_routing",
                confidence=0.90,
                risk_level="low",
                signals_detected=signals,
                explanation="Routed to official customer feedback portal."
            )

        # Layer 8: Autonomous Safe Auto-Handle (Software / Connectivity)
        signals.append(f"safe_to_autohandle_{intent_pred.intent}")
        return EscalationDecision(
            action="auto_handle",
            reason="self_service_troubleshooting",
            confidence=intent_pred.confidence,
            risk_level="low",
            signals_detected=signals,
            explanation="Autonomous self-service troubleshooting applicable."
        )
