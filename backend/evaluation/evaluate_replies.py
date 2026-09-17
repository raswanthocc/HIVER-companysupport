"""
Lexical Response Quality Evaluation & Documentation of Metric Limitations.
Computes ROUGE-1, ROUGE-2, ROUGE-L, and BLEU against historical reference replies.

IMPORTANT METHODOLOGICAL NOTE ON ROUGE/BLEU IN CUSTOMER SUPPORT:
Lexical overlap metrics (ROUGE, BLEU) have severe limitations when evaluating customer service agents:
1. Copy-Paste Artifact: An agent that blindly memorizes and copies historical tweets can score
   artificially near 1.0 on ROUGE while completely ignoring user safety or intent correctness.
2. Valid Variation: A polite, safe, empathetic response that asks a diagnostic question may have
   low lexical n-gram overlap with a historical tweet that simply said 'DM us', yet is substantively superior.
3. Therefore, ROUGE/BLEU are supplementary lexical indicators only, NOT primary evidence of support quality.
"""

from typing import List, Dict, Any
import numpy as np
from evaluation.metrics_utils import tokenize, compute_rouge_n, compute_rouge_l, compute_bleu

def evaluate_generated_replies(
    generated_replies: List[str],
    reference_replies: List[str]
) -> Dict[str, Any]:
    """Compute average lexical similarity metrics across generated and reference replies."""
    rouge1_list = []
    rouge2_list = []
    rougel_list = []
    bleu_list = []

    for gen, ref in zip(generated_replies, reference_replies):
        c_toks = tokenize(gen)
        r_toks = tokenize(ref)

        rouge1_list.append(compute_rouge_n(c_toks, r_toks, 1))
        rouge2_list.append(compute_rouge_n(c_toks, r_toks, 2))
        rougel_list.append(compute_rouge_l(c_toks, r_toks))
        bleu_list.append(compute_bleu(c_toks, r_toks, max_n=2))

    return {
        "rouge_1": round(float(np.mean(rouge1_list)), 4),
        "rouge_2": round(float(np.mean(rouge2_list)), 4),
        "rouge_l": round(float(np.mean(rougel_list)), 4),
        "bleu": round(float(np.mean(bleu_list)), 4),
        "metric_limitation_notice": (
            "ROUGE and BLEU are supplementary surface lexical metrics only. "
            "High lexical similarity can be artificially produced by verbatim copying of training text."
        )
    }
