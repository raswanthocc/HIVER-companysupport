"""
Pure Python NLP and Inter-Rater Reliability Metrics.
Computes ROUGE-1, ROUGE-2, ROUGE-L, BLEU, Cohen's Kappa, Pearson r, Spearman rho, and MAD.
Self-contained, fast, and dependency-free.
"""

import math
import re
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional

def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    text = text.lower()
    return re.findall(r"\b\w+\b", text)

def get_ngrams(tokens: List[str], n: int) -> Counter:
    """Extract n-gram frequency counter."""
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def compute_rouge_n(cand_tokens: List[str], ref_tokens: List[str], n: int) -> float:
    """Compute ROUGE-N F1 score."""
    if not cand_tokens or not ref_tokens:
        return 0.0
    cand_ngrams = get_ngrams(cand_tokens, n)
    ref_ngrams = get_ngrams(ref_tokens, n)
    if not ref_ngrams:
        return 0.0
    overlap = sum((cand_ngrams & ref_ngrams).values())
    recall = overlap / sum(ref_ngrams.values())
    precision = overlap / sum(cand_ngrams.values()) if cand_ngrams else 0.0
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)

def lcs_length(x: List[str], y: List[str]) -> int:
    """Compute Longest Common Subsequence length."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def compute_rouge_l(cand_tokens: List[str], ref_tokens: List[str]) -> float:
    """Compute ROUGE-L F1 score based on Longest Common Subsequence."""
    if not cand_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(cand_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    rec = lcs / len(ref_tokens)
    prec = lcs / len(cand_tokens)
    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)

def compute_bleu(cand_tokens: List[str], ref_tokens: List[str], max_n: int = 2) -> float:
    """Compute BLEU score with brevity penalty."""
    if not cand_tokens or not ref_tokens:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        cand_ng = get_ngrams(cand_tokens, n)
        ref_ng = get_ngrams(ref_tokens, n)
        if not cand_ng:
            precisions.append(0.0)
            continue
        overlap = sum((cand_ng & ref_ng).values())
        precisions.append(overlap / sum(cand_ng.values()))

    if any(p == 0.0 for p in precisions):
        return 0.0

    c = len(cand_tokens)
    r = len(ref_tokens)
    bp = math.exp(min(0, 1 - r / c)) if c > 0 else 0.0
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    return bp * geo_mean

def compute_cohens_kappa(rater1: List[Any], rater2: List[Any]) -> float:
    """Compute Cohen's Kappa between two categorical raters."""
    if len(rater1) != len(rater2) or len(rater1) == 0:
        return 0.0
    n = len(rater1)
    categories = list(set(rater1).union(set(rater2)))
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)

    cm = [[0] * k for _ in range(k)]
    for r1, r2 in zip(rater1, rater2):
        cm[cat_to_idx[r1]][cat_to_idx[r2]] += 1

    po = sum(cm[i][i] for i in range(k)) / n
    pe = sum(sum(cm[i][j] for j in range(k)) * sum(cm[j][i] for j in range(k)) for i in range(k)) / (n * n)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)

def compute_pearson_r(x: List[float], y: List[float]) -> float:
    """Compute Pearson linear correlation coefficient."""
    n = len(x)
    if n < 2 or len(y) != n:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)
    if var_x <= 1e-9 or var_y <= 1e-9:
        return 0.0
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    return float(cov / math.sqrt(var_x * var_y))

def compute_spearman_rho(x: List[float], y: List[float]) -> float:
    """Compute Spearman rank-order correlation coefficient."""
    n = len(x)
    if n < 2 or len(y) != n:
        return 0.0

    def rank_values(vals: List[float]) -> List[float]:
        sorted_indices = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        for rank_idx, val_idx in enumerate(sorted_indices):
            ranks[val_idx] = float(rank_idx + 1)
        return ranks

    rank_x = rank_values(x)
    rank_y = rank_values(y)
    return compute_pearson_r(rank_x, rank_y)

def compute_mad(x: List[float], y: List[float]) -> float:
    """Compute Mean Absolute Difference (MAD) between two continuous rating lists."""
    if not x or len(x) != len(y):
        return 0.0
    return float(sum(abs(xi - yi) for xi, yi in zip(x, y)) / len(x))
