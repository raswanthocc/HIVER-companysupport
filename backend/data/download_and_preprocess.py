"""
Data extraction and preprocessing script for CompanySupport conversations.
Fetches a sample slice from Hugging Face (SunidhiSriram/twcs), cleans text,
reconstructs customer -> agent pairs, and writes data/raw_sample.csv.
"""

import csv
import html
import io
import os
import re
import sys
import urllib.request
import http.client
import json

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_SAMPLE_PATH = os.path.join(DATA_DIR, "raw_sample.csv")
HF_URL = "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv"

def clean_tweet_text(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = text.replace("I️", "I").replace("i️", "I").replace("\ufe0f", "")
    text = re.sub(r"[\u200b-\u200d\uFEFF]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_leading_mentions(text: str) -> str:
    cleaned = re.sub(r"^(@\w+\s*)+", "", text).strip()
    return cleaned if cleaned else text

def download_and_extract_pairs(max_bytes: int = 15_000_000):
    print(f"Streaming up to {max_bytes / (1024*1024):.1f} MB slice from {HF_URL}...")
    req = urllib.request.Request(
        HF_URL, 
        headers={
            "Range": f"bytes=0-{max_bytes}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
    )
    
    raw_chunks = []
    total_read = 0
    chunk_size = 1024 * 1024  # 1MB chunks

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            while total_read < max_bytes:
                chunk = response.read(min(chunk_size, max_bytes - total_read))
                if not chunk:
                    break
                raw_chunks.append(chunk)
                total_read += len(chunk)
                print(f"Read {total_read / (1024*1024):.2f} MB...", flush=True)
    except http.client.IncompleteRead as e:
        print(f"IncompleteRead encountered after {len(e.partial):,} bytes. Using partial stream.")
        raw_chunks.append(e.partial)
    except Exception as e:
        print(f"Warning during download: {e}. Proceeding with collected data.")

    raw_bytes = b"".join(raw_chunks)
    print(f"Total downloaded: {len(raw_bytes):,} bytes. Parsing CSV...")
    
    text_data = raw_bytes.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(text_data))
    
    tweets = []
    for row in reader:
        if row.get("tweet_id") and row.get("text"):
            tweets.append(row)
    
    print(f"Parsed {len(tweets):,} valid tweet rows.")
    tweet_dict = {t["tweet_id"]: t for t in tweets}

    try:
        with open(os.path.join(DATA_DIR, "top_brands.json"), "r") as f:
            top_brands = set(json.load(f))
    except Exception as e:
        print(f"Could not load top_brands.json: {e}")
        top_brands = set()

    pairs = []
    seen_ids = set()

    for t in tweets:
        author = t.get("author_id", "")
        if author in top_brands and t.get("in_response_to_tweet_id"):
            in_resp_id = t.get("in_response_to_tweet_id")
            if in_resp_id in tweet_dict and in_resp_id not in seen_ids:
                cust_tweet = tweet_dict[in_resp_id]
                if cust_tweet.get("inbound") == "True":
                    cust_raw = clean_tweet_text(cust_tweet.get("text", ""))
                    agent_raw = clean_tweet_text(t.get("text", ""))
                    cust_clean = remove_leading_mentions(cust_raw)
                    agent_clean = remove_leading_mentions(agent_raw)

                    # Filter out empty queries or pure photo uploads without text
                    if len(cust_clean) > 8 and not cust_clean.startswith("https://t.co"):
                        seen_ids.add(in_resp_id)
                        pairs.append({
                            "brand": author,
                            "customer_tweet_id": cust_tweet.get("tweet_id"),
                            "customer_text_raw": cust_raw,
                            "customer_text": cust_clean,
                            "agent_tweet_id": t.get("tweet_id"),
                            "agent_text_raw": agent_raw,
                            "agent_text": agent_clean,
                            "created_at": cust_tweet.get("created_at", "")
                        })

    print(f"Successfully reconstructed {len(pairs):,} unique customer -> agent pairs across multiple brands.")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(RAW_SAMPLE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "brand", "customer_tweet_id", "customer_text_raw", "customer_text",
            "agent_tweet_id", "agent_text_raw", "agent_text", "created_at"
        ])
        writer.writeheader()
        writer.writerows(pairs)

    print(f"Saved {len(pairs)} pairs to {RAW_SAMPLE_PATH}")
    return pairs

if __name__ == "__main__":
    download_and_extract_pairs()
