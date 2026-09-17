"""
Deterministic Thread-Level Data Splitting Pipeline.
Strictly isolates the Golden Evaluation Set at the customer conversation thread level,
and partitions the remaining development pool into Train (80%) and Validation (20%) splits.
All dataset counts are dynamically derived from the underlying data without hardcoding.
"""

import csv
import json
import os
import random
import re
from typing import Dict, Any, List, Set, Tuple

from src.data.loader import load_raw_interactions, load_golden_eval_set
from src.data.conversations import (
    extract_customer_author_id,
    group_into_threads,
    find_thread_ids_for_golden
)
from src.intent.taxonomy import silver_label_interaction

def partition_data_at_thread_level(
    raw_sample_path: str = "data/raw_sample.csv",
    golden_path: str = "data/golden_eval_set.json",
    train_ratio: float = 0.80,
    random_seed: int = 42,
    output_dir: str = "data"
) -> Dict[str, Any]:
    """
    Perform leak-free thread-level dataset partitioning.
    1. Reconstruct conversation threads from raw interactions.
    2. Extract all thread IDs associated with Golden evaluation items.
    3. Segregate Golden threads entirely into the held-out test partition.
    4. Deterministically partition remaining development threads into Train and Validation.
    5. Save train_data.json, val_data.json, and train_retrieval_corpus.csv.
    """
    raw_interactions = load_raw_interactions(raw_sample_path)
    golden_data = load_golden_eval_set(golden_path)

    # 1. Group all raw interactions into conversation threads by customer ID
    all_threads = group_into_threads(raw_interactions)
    total_raw_turns = len(raw_interactions)
    total_thread_count = len(all_threads)

    # 2. Extract thread IDs corresponding to Golden evaluation set
    golden_thread_ids = find_thread_ids_for_golden(golden_data, raw_interactions)
    golden_thread_count = len(golden_thread_ids)

    # 3. Segregate into Golden partition vs Development pool
    golden_interactions = []
    dev_threads: Dict[str, List[Dict[str, Any]]] = {}

    for tid, rows in all_threads.items():
        if tid in golden_thread_ids:
            golden_interactions.extend(rows)
        else:
            dev_threads[tid] = rows

    dev_thread_ids = sorted(list(dev_threads.keys()))
    dev_thread_count = len(dev_thread_ids)
    dev_turns_count = sum(len(dev_threads[tid]) for tid in dev_thread_ids)

    # 4. Deterministic partition of development threads into Train & Validation
    rng = random.Random(random_seed)
    shuffled_dev_tids = list(dev_thread_ids)
    rng.shuffle(shuffled_dev_tids)

    num_train_threads = int(dev_thread_count * train_ratio)
    train_thread_ids = set(shuffled_dev_tids[:num_train_threads])
    val_thread_ids = set(shuffled_dev_tids[num_train_threads:])

    train_records = []
    retrieval_pairs = []
    for tid in train_thread_ids:
        for r in dev_threads[tid]:
            silver_intent = silver_label_interaction(r["customer_text"], r.get("agent_text", ""))
            record = {
                "thread_id": tid,
                "customer_tweet_id": r.get("customer_tweet_id", ""),
                "customer_text": r.get("customer_text", ""),
                "agent_tweet_id": r.get("agent_tweet_id", ""),
                "agent_text": r.get("agent_text", ""),
                "created_at": r.get("created_at", ""),
                "intent": silver_intent,
                "split": "train",
                "brand": r.get("brand", "Unknown")
            }
            train_records.append(record)
            retrieval_pairs.append({
                "conversation_id": tid,
                "customer_tweet_id": r.get("customer_tweet_id", ""),
                "customer_text": r.get("customer_text", ""),
                "agent_tweet_id": r.get("agent_tweet_id", ""),
                "agent_text": r.get("agent_text", ""),
                "intent": silver_intent,
                "brand": r.get("brand", "Unknown")
            })

    val_records = []
    for tid in val_thread_ids:
        for r in dev_threads[tid]:
            silver_intent = silver_label_interaction(r["customer_text"], r.get("agent_text", ""))
            record = {
                "thread_id": tid,
                "customer_tweet_id": r.get("customer_tweet_id", ""),
                "customer_text": r.get("customer_text", ""),
                "agent_tweet_id": r.get("agent_tweet_id", ""),
                "agent_text": r.get("agent_text", ""),
                "created_at": r.get("created_at", ""),
                "intent": silver_intent,
                "split": "val",
                "brand": r.get("brand", "Unknown")
            }
            val_records.append(record)

    # 5. Persist isolated files
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train_data.json")
    val_path = os.path.join(output_dir, "val_data.json")
    retrieval_path = os.path.join(output_dir, "train_retrieval_corpus.csv")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_records, f, indent=2)

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_records, f, indent=2)

    with open(retrieval_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "conversation_id", "customer_tweet_id", "customer_text",
            "agent_tweet_id", "agent_text", "intent", "brand"
        ])
        writer.writeheader()
        writer.writerows(retrieval_pairs)

    summary = {
        "total_raw_interactions": total_raw_turns,
        "total_conversation_threads": total_thread_count,
        "golden_test_items": len(golden_data),
        "golden_held_out_threads": golden_thread_count,
        "golden_thread_turns_quarantined": len(golden_interactions),
        "development_pool_threads": dev_thread_count,
        "development_pool_turns": dev_turns_count,
        "train_threads": len(train_thread_ids),
        "train_turns": len(train_records),
        "val_threads": len(val_thread_ids),
        "val_turns": len(val_records),
        "retrieval_corpus_size": len(retrieval_pairs),
        "train_path": train_path,
        "val_path": val_path,
        "retrieval_path": retrieval_path
    }

    return summary

if __name__ == "__main__":
    res = partition_data_at_thread_level()
    print("Data splitting complete. Summary:")
    for k, v in res.items():
        print(f"  {k}: {v}")
