"""
Unit tests for intent classifiers (Trivial, Simple, and Calibrated).
"""

import pytest
from src.intent.classifier import TrivialClassifier, SimpleClassifier, CalibratedIntentClassifier, clean_text
from src.intent.taxonomy import INTENT_LABELS

@pytest.fixture
def mini_train_data():
    return [
        ("My device battery drains in 2 hours since updating to the latest OS", "software_os"),
        ("Screen is completely cracked and battery seems swollen", "hardware_defect"),
        ("Locked out of my Account ID and cannot reset password", "account_security"),
        ("I was charged twice for iCloud subscription this month", "billing_subscription"),
        ("Wi-Fi keeps disconnecting every few minutes", "connectivity_features"),
        ("Company products are terrible and annoying", "general_feedback"),
        ("Battery dying after update", "software_os"),
        ("Cracked front glass", "hardware_defect"),
        ("Forgot my Account ID passcode", "account_security"),
        ("Unauthorized charge on receipt", "billing_subscription"),
        ("Bluetooth disconnects in car", "connectivity_features"),
        ("Worst company ever", "general_feedback"),
        ("App crashes on launch", "software_os"),
        ("Speaker buzzing sound", "hardware_defect"),
        ("2FA code not received", "account_security"),
        ("Refund for app store", "billing_subscription"),
        ("AirDrop not finding device", "connectivity_features"),
        ("I hate Company updates", "general_feedback")
    ]

def test_trivial_classifier(mini_train_data):
    texts, labels = zip(*mini_train_data)
    clf = TrivialClassifier()
    clf.fit(list(texts), list(labels))
    pred = clf.predict("Any arbitrary text")
    assert pred.intent == "software_os"
    assert pred.confidence == 1.0

def test_simple_classifier(mini_train_data):
    texts, labels = zip(*mini_train_data)
    clf = SimpleClassifier()
    clf.fit(list(texts), list(labels))
    pred = clf.predict("device battery dying fast")
    assert pred.intent in INTENT_LABELS
    assert 0.0 <= pred.confidence <= 1.0

def test_calibrated_classifier(mini_train_data):
    texts, labels = zip(*mini_train_data)
    clf = CalibratedIntentClassifier()
    clf.fit(list(texts), list(labels))
    pred = clf.predict("Battery is draining rapidly on the latest OS")
    assert pred.intent in INTENT_LABELS
    assert 0.0 <= pred.confidence <= 1.0
    assert len(pred.probabilities) == 6
    assert abs(sum(pred.probabilities.values()) - 1.0) < 0.05

def test_general_hardware_defect_queries():
    """Verify general unseen hardware defect queries classify as hardware_defect without hardcoding Golden items."""
    clf = CalibratedIntentClassifier()
    general_hw_queries = [
        "my screen is physically cracked",
        "the battery is swollen",
        "the charging port is broken",
        "my phone is overheating",
        "the button is physically damaged"
    ]
    for query in general_hw_queries:
        pred = clf.predict(query)
        assert pred.intent == "hardware_defect", f"Failed for query: {query}"
        assert pred.confidence >= 0.85

