"""
Unit tests for data leakage verification engine.
Validates that the pipeline detects contamination and fails loudly on any leak.
"""

import os
import pytest
from src.data.leakage_check import (
    verify_zero_leakage,
    DataLeakageError,
    compute_jaccard_similarity,
    compute_word_shingles
)

def test_genuine_dataset_zero_leakage():
    """Verify that current training, validation, and retrieval sets have zero leakage from Golden."""
    res = verify_zero_leakage()
    assert res["status"] == "PASSED_ZERO_LEAKAGE"
    assert res["tweet_id_overlap"] == 0
    assert res["exact_text_overlap"] == 0
    assert res["near_duplicates_exceeding_threshold"] == 0

def test_jaccard_similarity_calculation():
    s1 = compute_word_shingles("My device battery is dying fast")
    s2 = compute_word_shingles("My device battery is dying fast after update")
    sim = compute_jaccard_similarity(s1, s2)
    assert 0.60 <= sim <= 0.90

    # Completely disjoint sets
    s3 = compute_word_shingles("Completely unrelated words here")
    assert compute_jaccard_similarity(s1, s3) == 0.0

def test_fails_loudly_on_injected_leakage(tmp_path):
    """Simulate a contaminated dataset and assert DataLeakageError is raised."""
    import json
    import csv

    # Create mock golden file
    golden_file = tmp_path / "mock_golden.json"
    golden_data = [{
        "id": 1,
        "customer_tweet_id": "99999",
        "customer_text": "Secret golden text that must not leak",
        "intent": "software_os",
        "ground_truth_action": "auto_handle",
        "ground_truth_reason": "self_service_troubleshooting"
    }]
    golden_file.write_text(json.dumps(golden_data), encoding="utf-8")

    # Create contaminated train file
    train_file = tmp_path / "mock_train.json"
    train_data = [{
        "thread_id": "T1",
        "customer_tweet_id": "99999",  # Contaminated ID!
        "customer_text": "Normal query",
        "intent": "software_os"
    }]
    train_file.write_text(json.dumps(train_data), encoding="utf-8")

    # Create dummy retrieval file
    retrieval_file = tmp_path / "mock_retrieval.csv"
    with open(retrieval_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["conversation_id", "customer_tweet_id", "customer_text", "agent_text", "intent"])
        writer.writeheader()
        writer.writerow({"conversation_id": "T1", "customer_tweet_id": "88888", "customer_text": "Q", "agent_text": "A", "intent": "software_os"})

    # Must raise DataLeakageError
    with pytest.raises(DataLeakageError) as exc_info:
        verify_zero_leakage(
            golden_path=str(golden_file),
            train_path=str(train_file),
            retrieval_path=str(retrieval_file)
        )
    assert "CRITICAL LEAKAGE DETECTED" in str(exc_info.value)
