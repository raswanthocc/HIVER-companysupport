"""
Data loading and basic cleaning utilities for CompanySupport dataset.
"""

import csv
import html
import json
import os
import re
from typing import List, Dict, Any, Optional

def clean_tweet_text(text: str) -> str:
    """Normalize tweet text: remove HTML entities, zero-width chars, and superfluous whitespace."""
    if not text:
        return ""
    text = html.unescape(text)
    text = text.replace("I️", "I").replace("i️", "I").replace("\ufe0f", "")
    text = re.sub(r"[\u200b-\u200d\uFEFF]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_leading_mentions(text: str) -> str:
    """Remove leading @handles from tweet text while preserving content."""
    cleaned = re.sub(r"^(@\w+\s*)+", "", text).strip()
    return cleaned if cleaned else text

def load_raw_interactions(csv_path: str) -> List[Dict[str, str]]:
    """Load raw paired customer-agent interactions from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw interactions file not found: {csv_path}")
    interactions = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            interactions.append(dict(row))
    return interactions

def load_golden_eval_set(json_path: str) -> List[Dict[str, Any]]:
    """Load the hand-labelled golden evaluation dataset."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Golden evaluation file not found: {json_path}")
    with open(json_path, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_knowledge_base(json_path: str) -> Dict[str, Any]:
    """Load the CompanySupport knowledge base catalog and policy rules."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Knowledge base file not found: {json_path}")
    with open(json_path, mode="r", encoding="utf-8") as f:
        return json.load(f)
