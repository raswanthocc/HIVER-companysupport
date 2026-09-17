"""
Comprehensive unit tests for retrieval grounding and evidence relevance gating.
Validates the 5 grounding guarantees required for reliable, leak-free customer support:
1. Relevant retrieval -> historical evidence is adapted.
2. Wrong-intent retrieval -> historical evidence is rejected.
3. Weak similarity -> safe KB fallback is used.
4. Golden examples are NEVER present in retrieval corpus.
5. Generated reply does not blindly return an unrelated historical reply.
"""

import json
import os
import pytest

from src.models import IntentPrediction, EscalationDecision, RetrievedContext
from src.generation.generator import GroundedResponseGenerator
from src.retrieval.retriever import ResolutionRetriever
from src.agent import CompanySupportAgent

@pytest.fixture
def generator():
    return GroundedResponseGenerator(min_exemplar_similarity=0.28)

@pytest.fixture
def auto_decision():
    return EscalationDecision(
        action="auto_handle",
        reason="self_service_troubleshooting",
        confidence=0.90,
        risk_level="low",
        signals_detected=["safe_to_autohandle"]
    )

def test_grounding_relevant_retrieval_used(generator, auto_decision):
    """1. Relevant retrieval -> historical evidence is accepted and adapted (not copied verbatim)."""
    intent_pred = IntentPrediction(intent="software_os", confidence=0.88)
    customer_text = "My device keeps freezing after updating to the latest OS."
    
    historical_exemplar = {
        "customer_text": "device keeps freezing randomly since new update.",
        "agent_text": "@customer Try force restarting your device and ensure you have the latest software update installed.",
        "intent": "software_os",
        "similarity_score": 0.45,
        "is_same_intent": True
    }
    retrieved = RetrievedContext(
        kb_guide={"default_routing": "auto_handle"},
        historical_exemplars=[historical_exemplar],
        similarity_scores=[0.45]
    )

    reply = generator.generate(customer_text, intent_pred, auto_decision, retrieved)

    # Must accept evidence
    assert retrieved.grounding_status == "grounded_historical_adaptation"
    # Must NOT copy verbatim with customer handle
    assert "@customer" not in reply
    # Must adapt the resolution to the customer's freezing issue
    assert "freezing" in reply.lower()
    assert "restart" in reply.lower()

def test_grounding_wrong_intent_rejected(generator, auto_decision):
    """2. Wrong-intent retrieval -> historical evidence is rejected and KB fallback used."""
    intent_pred = IntentPrediction(intent="software_os", confidence=0.88)
    customer_text = "My battery is draining rapidly on the latest OS."

    # Historical exemplar from wrong intent (e.g. hardware_defect)
    cross_intent_exemplar = {
        "customer_text": "My Touch ID sensor stopped reading my thumb completely.",
        "agent_text": "We recommend bringing your device into an Company Store Genius Bar for a sensor diagnostic.",
        "intent": "hardware_defect",
        "similarity_score": 0.35,
        "is_same_intent": False
    }
    retrieved = RetrievedContext(
        kb_guide={"default_routing": "auto_handle"},
        historical_exemplars=[cross_intent_exemplar],
        similarity_scores=[0.35]
    )

    reply = generator.generate(customer_text, intent_pred, auto_decision, retrieved)

    # Must reject cross-intent evidence
    assert retrieved.grounding_status == "kb_policy_fallback"
    assert "Genius Bar" not in reply
    assert "Touch ID" not in reply
    # Must use software/battery diagnostic guidance
    assert any(w in reply.lower() for w in ["battery", "restart", "settings"])

def test_grounding_weak_similarity_fallback(generator, auto_decision):
    """3. Weak similarity (< 0.28) -> evidence gate rejects match and safe KB fallback is used."""
    intent_pred = IntentPrediction(intent="software_os", confidence=0.85)
    customer_text = "Why is my device acting strange today?"

    weak_exemplar = {
        "customer_text": "My phone is acting odd after yesterday.",
        "agent_text": "Reset all network settings by going to Settings > General > Reset.",
        "intent": "software_os",
        "similarity_score": 0.15,  # Below 0.28 threshold
        "is_same_intent": True
    }
    retrieved = RetrievedContext(
        kb_guide={},
        historical_exemplars=[weak_exemplar],
        similarity_scores=[0.15]
    )

    reply = generator.generate(customer_text, intent_pred, auto_decision, retrieved)

    # Must reject weak match
    assert retrieved.grounding_status == "kb_policy_fallback"
    # Must ask safe diagnostic question
    assert "?" in reply

def test_golden_example_never_retrieved():
    """4. Golden evaluation set items must NEVER be present in the retrieval corpus."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    golden_path = os.path.join(base_dir, "data", "golden_eval_set.json")
    with open(golden_path, encoding="utf-8") as f:
        golden_items = json.load(f)

    retriever = ResolutionRetriever()
    retrieval_tids = {str(p["customer_tweet_id"]) for p in retriever.historical_pairs if p.get("customer_tweet_id")}
    retrieval_texts = {p["customer_text"].strip().lower() for p in retriever.historical_pairs if p.get("customer_text")}

    for g in golden_items:
        g_tid = str(g["customer_tweet_id"])
        g_text = g["customer_text"].strip().lower()

        assert g_tid not in retrieval_tids, f"Golden tweet ID {g_tid} leaked into retrieval corpus!"
        assert g_text not in retrieval_texts, f"Golden text '{g_text[:40]}' leaked into retrieval corpus!"

def test_grounding_does_not_blindly_return_unrelated_reply(generator, auto_decision):
    """5. Generated response must not blindly return an unrelated reply (e.g. Touch ID for battery drain)."""
    intent_pred = IntentPrediction(intent="software_os", confidence=0.90)
    customer_text = "My battery drains extremely quickly."

    unrelated_exemplar = {
        "customer_text": "Touch ID is not recognizing my fingerprint after update.",
        "agent_text": "Have you tried deleting and re-registering your fingerprint in Touch ID settings?",
        "intent": "software_os",  # Even if mislabeled as software_os, topic check catches it
        "similarity_score": 0.32,
        "is_same_intent": True
    }
    retrieved = RetrievedContext(
        kb_guide={},
        historical_exemplars=[unrelated_exemplar],
        similarity_scores=[0.32]
    )

    reply = generator.generate(customer_text, intent_pred, auto_decision, retrieved)

    # The topic check must detect incompatible topic and reject it
    assert retrieved.grounding_status == "kb_policy_fallback"
    assert "fingerprint" not in reply.lower()
    assert "touch id" not in reply.lower()
    assert "battery" in reply.lower()
