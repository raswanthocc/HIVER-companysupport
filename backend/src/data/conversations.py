"""
Conversation and Thread Reconstruction for Customer Support on Twitter.
Groups raw turn pairs into conversation threads based on customer author IDs
and tweet response relationships.
"""

import re
from typing import List, Dict, Any, Set, Tuple

def extract_customer_author_id(agent_text_raw: str, fallback_id: str = "") -> str:
    """
    Extract the customer author ID from the agent's raw response tweet.
    In the Twitter Customer Support dataset, CompanySupport replies to the customer
    using their anonymized user ID, e.g. '@115854 Lets take a closer look...'.
    """
    if agent_text_raw:
        match = re.search(r"@(\d+)", agent_text_raw)
        if match:
            return match.group(1)
    return str(fallback_id)

def group_into_threads(interactions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group customer-agent interaction pairs into conversation threads.
    Each unique customer author ID represents a continuous conversation session/thread.
    """
    threads: Dict[str, List[Dict[str, Any]]] = {}
    for row in interactions:
        cid = extract_customer_author_id(
            row.get("agent_text_raw", ""),
            fallback_id=row.get("customer_tweet_id", "")
        )
        threads.setdefault(cid, []).append(row)
    return threads

def find_thread_ids_for_golden(
    golden_data: List[Dict[str, Any]],
    raw_interactions: List[Dict[str, Any]]
) -> Set[str]:
    """
    Identify all conversation thread IDs associated with any golden evaluation item.
    Matches by customer_tweet_id first, then by normalized customer text.
    """
    raw_by_tweet_id = {str(r.get("customer_tweet_id", "")): r for r in raw_interactions}
    raw_by_text = {re.sub(r"\s+", " ", r.get("customer_text", "").strip().lower()): r for r in raw_interactions}

    golden_thread_ids = set()
    for g in golden_data:
        tid = str(g.get("customer_tweet_id", ""))
        matched_row = None
        if tid in raw_by_tweet_id:
            matched_row = raw_by_tweet_id[tid]
        else:
            norm_text = re.sub(r"\s+", " ", g.get("customer_text", "").strip().lower())
            if norm_text in raw_by_text:
                matched_row = raw_by_text[norm_text]

        if matched_row:
            cid = extract_customer_author_id(
                matched_row.get("agent_text_raw", ""),
                fallback_id=matched_row.get("customer_tweet_id", "")
            )
            golden_thread_ids.add(cid)

    return golden_thread_ids
