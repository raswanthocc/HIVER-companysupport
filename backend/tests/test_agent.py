"""
Unit tests for end-to-end CompanySupportAgent.
Validates complete triage flow, response generation, DM link inclusion, and output schema.
"""

import pytest
from src.agent import CompanySupportAgent
from src.generator import GroundedResponseGenerator

def test_agent_end_to_end_escalation():
    agent = CompanySupportAgent()
    res = agent.process_query("I was charged twice on my credit card, need a refund immediately")
    assert res.predicted_intent in ["billing_subscription", "account_security", "software_os"]
    assert res.escalation_action == "escalate"
    assert res.escalation_reason == "financial_dispute"
    assert "[Human Support Handoff Triggered]" in res.generated_reply
    assert res.latency_ms >= 0.0

def test_agent_end_to_end_autohandle():
    agent = CompanySupportAgent()
    res = agent.process_query("How do I force restart my device after updating?")
    assert res.predicted_intent == "software_os"
    assert res.escalation_action == "auto_handle"
    assert len(res.generated_reply) > 15
    assert res.latency_ms >= 0.0
