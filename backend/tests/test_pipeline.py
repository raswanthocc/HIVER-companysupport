"""
Unit and Integration Tests for CompanySupport AI Agent Pipeline.
"""

import os
import sys
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models import CustomerQuery, IntentPrediction, EscalationDecision
from src.classifier import CalibratedIntentClassifier, SimpleClassifier, TrivialClassifier
from src.escalation import EscalationEngine
from src.retriever import ResolutionRetriever
from src.generator import GroundedResponseGenerator
from src.agent import CompanySupportAgent
from evaluation.metrics_utils import compute_rouge_n, compute_rouge_l, compute_bleu, compute_cohens_kappa
from evaluation.llm_judge import RealLLMJudge

@pytest.fixture
def sample_data():
    return [
        ("My device 11 battery drains in 2 hours since updating to the latest OS", "software_os"),
        ("Screen is completely cracked and battery seems swollen", "hardware_defect"),
        ("Locked out of my Account ID and cannot reset password", "account_security"),
        ("I was charged twice for iCloud subscription this month", "billing_subscription"),
        ("Wi-Fi keeps disconnecting every few minutes", "connectivity_features"),
        ("Company products are terrible and annoying", "general_feedback")
    ]

def test_classifier_training_and_prediction(sample_data):
    texts, labels = zip(*sample_data)
    clf = CalibratedIntentClassifier()
    clf.fit(list(texts), list(labels))
    
    pred = clf.predict("My device battery is dying fast after OS update")
    assert pred.intent in ["software_os", "hardware_defect", "account_security", "billing_subscription", "connectivity_features", "general_feedback"]
    assert 0.0 <= pred.confidence <= 1.0
    assert len(pred.probabilities) == 6

def test_escalation_pii_detection():
    engine = EscalationEngine()
    pred_intent = IntentPrediction(intent="account_security", confidence=0.9)
    decision = engine.evaluate("Here is my serial number F2LN6123ABCD and email user@example.com", pred_intent)
    assert decision.action == "escalate"
    assert decision.reason == "requires_pii_or_dm"
    assert "pii_or_credentials_detected" in decision.signals_detected

def test_escalation_hardware_hazard():
    engine = EscalationEngine()
    pred_intent = IntentPrediction(intent="hardware_defect", confidence=0.9)
    decision = engine.evaluate("My battery is swollen and screen is bulging", pred_intent)
    assert decision.action == "escalate"
    assert decision.reason == "hardware_safety_repair"
    assert decision.risk_level == "critical"

def test_escalation_exhausted_troubleshooting():
    engine = EscalationEngine()
    pred_intent = IntentPrediction(intent="software_os", confidence=0.85)
    decision = engine.evaluate("I already restarted my phone and reset my settings, still not working", pred_intent)
    assert decision.action == "escalate"
    assert decision.reason == "repeated_failure_exhausted"

def test_escalation_autohandle_routine_query():
    engine = EscalationEngine()
    pred_intent = IntentPrediction(intent="software_os", confidence=0.92)
    decision = engine.evaluate("How do I clear Safari history on my device?", pred_intent)
    assert decision.action == "auto_handle"
    assert decision.reason == "self_service_troubleshooting"

def test_generator_dm_link_on_escalate():
    gen = GroundedResponseGenerator()
    intent_pred = IntentPrediction(intent="billing_subscription", confidence=0.9)
    escalation = EscalationEngine().evaluate("I need a refund for an unauthorized charge", intent_pred)
    reply = gen._generate_escalation_reply("I need a refund", escalation, None)
    assert "[Human Support Handoff Triggered]" in reply

def test_full_agent_e2e():
    clf = CalibratedIntentClassifier()
    clf.fit(
        ["battery drain after update", "cracked screen swollen", "Account ID password locked", "unauthorized charge refund", "wifi dropping", "Company is bad"],
        ["software_os", "hardware_defect", "account_security", "billing_subscription", "connectivity_features", "general_feedback"]
    )
    agent = CompanySupportAgent(classifier=clf)
    result = agent.process_query("My Wi-Fi keeps disconnecting")
    assert result.customer_text == "My Wi-Fi keeps disconnecting"
    assert result.predicted_intent is not None
    assert result.escalation_action in ["auto_handle", "escalate"]
    assert len(result.generated_reply) > 10
    assert result.latency_ms >= 0.0

def test_metrics_utils():
    cand = ["hello", "world", "this", "is", "a", "test"]
    ref = ["hello", "world", "this", "is", "great"]
    r1 = compute_rouge_n(cand, ref, 1)
    rl = compute_rouge_l(cand, ref)
    b = compute_bleu(cand, ref, 2)
    assert 0.0 <= r1 <= 1.0
    assert 0.0 <= rl <= 1.0
    assert 0.0 <= b <= 1.0
    kappa = compute_cohens_kappa(["a", "b", "a"], ["a", "b", "b"])
    assert -1.0 <= kappa <= 1.0

def test_judge_rubric():
    judge = RealLLMJudge()
    pred = CompanySupportAgent().process_query("How do I update OS?")
    evaluation = judge.evaluate_reply(
        customer_text="How do I update OS?",
        generated_reply=pred.generated_reply,
        predicted_action="auto_handle",
        predicted_reason="self_service_troubleshooting"
    )
    assert evaluation["status"] in ["EXECUTED", "NOT_EXECUTED", "ERROR"]
    if evaluation["status"] == "EXECUTED":
        assert 1.0 <= evaluation["composite_score"] <= 5.0
        assert len(evaluation["dimensional_scores"]) == 5
    elif evaluation["status"] == "ERROR":
        assert evaluation["composite_score"] is None
        assert "error" in evaluation
    else:
        assert evaluation["composite_score"] is None
        assert "NOT_EXECUTED" in evaluation["status"]
