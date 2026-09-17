"""
Data schemas for the CompanySupport AI Agent.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class CustomerQuery:
    tweet_id: str
    text: str
    created_at: Optional[str] = None
    author_id: Optional[str] = None

@dataclass
class IntentPrediction:
    intent: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)

@dataclass
class EscalationDecision:
    action: str  # "auto_handle" or "escalate"
    reason: str  # explicit stated reason
    confidence: float
    risk_level: str  # "low", "medium", "high", "critical"
    signals_detected: List[str] = field(default_factory=list)
    explanation: str = ""

@dataclass
class RetrievedContext:
    kb_guide: Optional[Dict[str, Any]] = None
    historical_exemplars: List[Dict[str, Any]] = field(default_factory=list)
    similarity_scores: List[float] = field(default_factory=list)
    grounding_status: str = "kb_policy_fallback"  # "grounded_historical_adaptation" or "kb_policy_fallback"

@dataclass
class AgentPrediction:
    query_id: str
    customer_text: str
    predicted_intent: str
    intent_confidence: float
    escalation_action: str
    escalation_reason: str
    risk_level: str
    generated_reply: str
    retrieved_exemplars: List[str] = field(default_factory=list)
    retrieved_evidence: List[Dict[str, Any]] = field(default_factory=list)
    signals: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    escalation_explanation: str = ""
    grounding_status: str = "kb_policy_fallback"
    judge_evaluation: Optional[Dict[str, Any]] = None

@dataclass
class EvaluationRecord:
    id: int
    customer_text: str
    historical_agent_text: str
    ground_truth_intent: str
    predicted_intent: str
    intent_correct: bool
    ground_truth_action: str
    predicted_action: str
    action_correct: bool
    ground_truth_reason: str
    predicted_reason: str
    reason_correct: bool
    generated_reply: str
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0
    bleu: float = 0.0
    judge_score: float = 0.0
    judge_critique: str = ""
