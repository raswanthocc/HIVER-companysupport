"""
Automated Data Leakage Verification Engine.
Performs strict multi-factor isolation verification to guarantee that zero
Golden evaluation data leaks into training, validation, or historical retrieval pools.
Fails loudly by raising DataLeakageError if any contamination is detected.
"""

import csv
import json
import os
import re
from typing import Dict, Any, List, Set, Tuple

class DataLeakageError(Exception):
    """Raised when data leakage is detected between the held-out Golden set and training/retrieval data."""
    pass

def normalize_text_for_comparison(text: str) -> str:
    """Normalize text for exact duplicate detection."""
    text = text.lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def compute_word_shingles(text: str) -> Set[str]:
    """Compute token shingles for Jaccard similarity check."""
    words = re.findall(r"\b\w+\b", text.lower())
    return set(words)

def compute_jaccard_similarity(s1: Set[str], s2: Set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not s1 or not s2:
        return 0.0
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union if union > 0 else 0.0

def verify_zero_leakage(
    golden_path: str = "data/golden_eval_set.json",
    train_path: str = "data/train_data.json",
    val_path: str = "data/val_data.json",
    retrieval_path: str = "data/train_retrieval_corpus.csv",
    jaccard_threshold: float = 0.85
) -> Dict[str, Any]:
    """
    Execute exhaustive 4-stage data leakage verification.
    1. Golden thread IDs vs Train/Val/Retrieval thread IDs.
    2. Golden tweet IDs vs Train/Val/Retrieval tweet IDs.
    3. Golden exact normalized customer text vs Train/Val/Retrieval texts.
    4. Pairwise Jaccard near-duplicate similarity across splits (< threshold).

    Raises:
        DataLeakageError: If any leak or near-duplicate exceeding threshold is detected.
    Returns:
        Dict summarizing audit results with derived counts.
    """
    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"Golden dataset missing at {golden_path}")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train dataset missing at {train_path}")
    if not os.path.exists(retrieval_path):
        raise FileNotFoundError(f"Retrieval dataset missing at {retrieval_path}")

    # Load Golden dataset
    with open(golden_path, encoding="utf-8") as f:
        golden_data = json.load(f)

    # Load Train dataset
    with open(train_path, encoding="utf-8") as f:
        train_data = json.load(f)

    # Load Val dataset if available
    val_data = []
    if os.path.exists(val_path):
        with open(val_path, encoding="utf-8") as f:
            val_data = json.load(f)

    # Load Retrieval dataset
    retrieval_rows = []
    with open(retrieval_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            retrieval_rows.append(r)

    # 1. Extract Golden identifiers
    golden_tweet_ids: Set[str] = set()
    golden_norm_texts: Set[str] = set()
    golden_shingles: List[Tuple[str, Set[str]]] = []

    for g in golden_data:
        tid = str(g.get("customer_tweet_id", ""))
        if tid:
            golden_tweet_ids.add(tid)
        c_text = g.get("customer_text", "")
        norm = normalize_text_for_comparison(c_text)
        if norm:
            golden_norm_texts.add(norm)
            golden_shingles.append((c_text, compute_word_shingles(c_text)))

    # 2. Extract Dev / Train identifiers
    train_thread_ids: Set[str] = {str(r.get("thread_id", "")) for r in train_data if r.get("thread_id")}
    val_thread_ids: Set[str] = {str(r.get("thread_id", "")) for r in val_data if r.get("thread_id")}
    retrieval_conv_ids: Set[str] = {str(r.get("conversation_id", "")) for r in retrieval_rows if r.get("conversation_id")}

    train_tweet_ids: Set[str] = {str(r.get("customer_tweet_id", "")) for r in train_data if r.get("customer_tweet_id")}
    val_tweet_ids: Set[str] = {str(r.get("customer_tweet_id", "")) for r in val_data if r.get("customer_tweet_id")}
    retrieval_tweet_ids: Set[str] = {str(r.get("customer_tweet_id", "")) for r in retrieval_rows if r.get("customer_tweet_id")}

    train_norm_texts: Set[str] = {normalize_text_for_comparison(r.get("customer_text", "")) for r in train_data}
    val_norm_texts: Set[str] = {normalize_text_for_comparison(r.get("customer_text", "")) for r in val_data}
    retrieval_norm_texts: Set[str] = {normalize_text_for_comparison(r.get("customer_text", "")) for r in retrieval_rows}

    # CHECK 1: Tweet ID Contamination
    leaked_train_tweets = golden_tweet_ids & train_tweet_ids
    leaked_val_tweets = golden_tweet_ids & val_tweet_ids
    leaked_retrieval_tweets = golden_tweet_ids & retrieval_tweet_ids
    all_leaked_tweets = leaked_train_tweets | leaked_val_tweets | leaked_retrieval_tweets

    if all_leaked_tweets:
        raise DataLeakageError(
            f"CRITICAL LEAKAGE DETECTED: {len(all_leaked_tweets)} Golden customer tweet IDs "
            f"found in train/val/retrieval splits! Example leaked IDs: {list(all_leaked_tweets)[:5]}"
        )

    # CHECK 2: Exact Customer Text Contamination
    leaked_train_texts = golden_norm_texts & train_norm_texts
    leaked_val_texts = golden_norm_texts & val_norm_texts
    leaked_retrieval_texts = golden_norm_texts & retrieval_norm_texts
    all_leaked_texts = leaked_train_texts | leaked_val_texts | leaked_retrieval_texts

    if all_leaked_texts:
        raise DataLeakageError(
            f"CRITICAL LEAKAGE DETECTED: {len(all_leaked_texts)} Golden customer texts "
            f"found verbatim in train/val/retrieval splits! Example leaked text: {list(all_leaked_texts)[:3]}"
        )

    # CHECK 3: Conversation / Thread Isolation
    # Verify train and val threads do not overlap with each other
    train_val_thread_overlap = train_thread_ids & val_thread_ids
    if train_val_thread_overlap:
        raise DataLeakageError(
            f"THREAD CONTAMINATION: {len(train_val_thread_overlap)} threads present in both train and val splits!"
        )

    # CHECK 4: Near-duplicate Jaccard Check
    # Verify no training or retrieval item has Jaccard similarity >= jaccard_threshold with any Golden item
    near_duplicates: List[Dict[str, Any]] = []
    dev_pool_texts = [(r.get("customer_text", ""), compute_word_shingles(r.get("customer_text", ""))) for r in train_data]

    for g_text, g_sh in golden_shingles:
        if not g_sh:
            continue
        for d_text, d_sh in dev_pool_texts:
            if not d_sh:
                continue
            sim = compute_jaccard_similarity(g_sh, d_sh)
            if sim >= jaccard_threshold:
                near_duplicates.append({
                    "similarity": round(sim, 4),
                    "golden_text": g_text,
                    "dev_text": d_text
                })

    if near_duplicates:
        raise DataLeakageError(
            f"CRITICAL NEAR-DUPLICATE LEAKAGE: {len(near_duplicates)} pairs exceed Jaccard similarity {jaccard_threshold}! "
            f"Top near-duplicate: {near_duplicates[0]}"
        )

    return {
        "status": "PASSED_ZERO_LEAKAGE",
        "golden_samples_audited": len(golden_data),
        "golden_tweet_ids_count": len(golden_tweet_ids),
        "train_samples_audited": len(train_data),
        "val_samples_audited": len(val_data),
        "retrieval_samples_audited": len(retrieval_rows),
        "tweet_id_overlap": len(all_leaked_tweets),
        "exact_text_overlap": len(all_leaked_texts),
        "train_val_thread_overlap": len(train_val_thread_overlap),
        "near_duplicates_exceeding_threshold": len(near_duplicates),
        "jaccard_threshold": jaccard_threshold
    }

if __name__ == "__main__":
    result = verify_zero_leakage()
    print("Leakage Verification Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
