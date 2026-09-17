"""
Unit tests for EscalationEngine.
Validates multi-signal risk rules, PII safety overrides, and explicit reasons.
"""

import pytest
from src.escalation.policy import EscalationEngine
from src.models import IntentPrediction

def test_escalation_pii_detection():
    engine = EscalationEngine()
    pred_intent = IntentPrediction(intent="software_os", confidence=0.90)
    decision = engine.evaluate("My email is customer@gmail.com and serial is F2LN6123ABCD", pred_intent)
    assert decision.action == "escalate"
    assert decision.reason == "requires_pii_or_dm"
    assert "pii_or_credentials_detected" in decision.signals_detected

def test_escalation_hardware_critical_hazard():
    engine = EscalationEngine()
    pred_intent = IntentPrediction(intent="hardware_defect", confidence=0.92)
    decision = engine.evaluate("My phone battery is swollen and smoking!", pred_intent)
    assert decision.action == "escalate"
    assert decision.reason == "hardware_safety_repair"
    assert decision.risk_level == "critical"

def test_escalation_financial_dispute():
    engine = EscalationEngine()
    pred_intent = IntentPrediction(intent="billing_subscription", confidence=0.88)
    decision = engine.evaluate("I was charged twice on my credit card and need a refund", pred_intent)
    assert decision.action == "escalate"
    assert decision.reason == "financial_dispute"

def test_escalation_exhausted_troubleshooting():
    engine = EscalationEngine()
    pred_intent = IntentPrediction(intent="software_os", confidence=0.85)
    decision = engine.evaluate("I already tried restarting and reset my settings, still not working", pred_intent)
    assert decision.action == "escalate"
    assert decision.reason == "repeated_failure_exhausted"

def test_escalation_confidence_fallback():
    engine = EscalationEngine(confidence_threshold=0.55)
    pred_intent = IntentPrediction(intent="software_os", confidence=0.42)
    decision = engine.evaluate("Something weird is happening", pred_intent)
    assert decision.action == "escalate"
    assert decision.reason == "low_confidence_fallback"

def test_escalation_autohandle_routine():
    engine = EscalationEngine()
    pred_intent = IntentPrediction(intent="software_os", confidence=0.92)
    decision = engine.evaluate("How do I update to the new OS version?", pred_intent)
    assert decision.action == "auto_handle"
    assert decision.reason == "self_service_troubleshooting"

def test_escalation_physical_damage_categories():
    """Verify physical damage/hardware defects escalate with evidence-based reason."""
    engine = EscalationEngine()
    test_cases = [
        "my screen is cracked and broken",
        "the charging port is physically damaged",
        "phone is overheating and burning hot",
        "the home button broke off"
    ]
    for text in test_cases:
        pred_intent = IntentPrediction(intent="software_os", confidence=0.80)
        dec = engine.evaluate(text, pred_intent)
        assert dec.action == "escalate"
        assert dec.reason == "hardware_safety_repair"
        assert "physical device damage" in dec.explanation.lower()

def test_escalation_account_security_categories():
    """Verify account compromise, hacking, credentials escalate with evidence-based reason."""
    engine = EscalationEngine()
    test_cases = [
        "someone hacked my Account ID account",
        "unauthorized login to my icloud credentials",
        "locked out of my account and lost 2fa verification code"
    ]
    for text in test_cases:
        pred_intent = IntentPrediction(intent="software_os", confidence=0.80)
        dec = engine.evaluate(text, pred_intent)
        assert dec.action == "escalate"
        assert dec.reason == "requires_pii_or_dm"
        assert "credentials" in dec.explanation.lower()

def test_escalation_repeated_troubleshooting_phrases():
    """Verify repeated troubleshooting failure signals escalate."""
    engine = EscalationEngine()
    test_cases = [
        "I already tried restarting and nothing worked",
        "Done all the steps, still not working after multiple attempts",
        "Same issue after restart, wasted hours on this"
    ]
    for text in test_cases:
        pred_intent = IntentPrediction(intent="software_os", confidence=0.85)
        dec = engine.evaluate(text, pred_intent)
        assert dec.action == "escalate"
        assert dec.reason == "repeated_failure_exhausted"

def test_escalation_safety_sensitive_confidence_gating():
    """Verify that lower confidence on safety-sensitive intents triggers fallback escalation."""
    engine = EscalationEngine()
    # At 0.65 confidence, safety sensitive hardware_defect must escalate
    pred_intent = IntentPrediction(intent="hardware_defect", confidence=0.65)
    dec = engine.evaluate("general device question", pred_intent)
    assert dec.action == "escalate"
    assert dec.reason == "hardware_safety_repair"  # hardware intent routes to repair or fallback

