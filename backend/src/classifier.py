"""
Intent classification module for CompanySupport.
Re-exports classifiers from src.intent.classifier for backwards compatibility.
"""

from src.intent.classifier import (
    clean_text,
    TrivialClassifier,
    SimpleClassifier,
    CalibratedIntentClassifier,
    INTENT_LABELS,
    IntentPrediction
)

__all__ = [
    "clean_text",
    "TrivialClassifier",
    "SimpleClassifier",
    "CalibratedIntentClassifier",
    "INTENT_LABELS",
    "IntentPrediction"
]
