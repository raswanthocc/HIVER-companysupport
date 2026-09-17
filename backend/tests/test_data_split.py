"""
Unit tests for conversation thread reconstruction and thread-level data splitting.
"""

import os
import json
import pytest
from src.data.conversations import (
    extract_customer_author_id,
    group_into_threads,
    find_thread_ids_for_golden
)
from src.data.split import partition_data_at_thread_level

def test_extract_customer_author_id():
    raw_agent = "@115854 Lets take a closer look into this issue."
    cid = extract_customer_author_id(raw_agent, fallback_id="999")
    assert cid == "115854"

    # Fallback when no @<id> present
    no_id_agent = "We are happy to help you today."
    fallback = extract_customer_author_id(no_id_agent, fallback_id="999")
    assert fallback == "999"

def test_group_into_threads():
    interactions = [
        {"agent_text_raw": "@1001 Hello", "customer_tweet_id": "1", "customer_text": "Q1"},
        {"agent_text_raw": "@1001 Follow-up", "customer_tweet_id": "2", "customer_text": "Q2"},
        {"agent_text_raw": "@1002 Hi there", "customer_tweet_id": "3", "customer_text": "Q3"}
    ]
    threads = group_into_threads(interactions)
    assert len(threads) == 2
    assert len(threads["1001"]) == 2
    assert len(threads["1002"]) == 1

def test_thread_partition_disjointness():
    # Verify generated train and val partitions have zero thread overlap
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "train_data.json")
    val_path = os.path.join(base_dir, "data", "val_data.json")

    assert os.path.exists(train_path), "train_data.json must exist"
    assert os.path.exists(val_path), "val_data.json must exist"

    with open(train_path, encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, encoding="utf-8") as f:
        val_data = json.load(f)

    train_threads = {r["thread_id"] for r in train_data}
    val_threads = {r["thread_id"] for r in val_data}

    # Strict zero cross-thread overlap
    assert len(train_threads & val_threads) == 0
    assert len(train_data) > 0
    assert len(val_data) > 0
