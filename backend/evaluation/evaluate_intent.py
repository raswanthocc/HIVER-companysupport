"""
Intent Classification Evaluation Harness on Held-Out Golden Test Set.
Computes Accuracy, Macro/Weighted F1, Macro Precision/Recall, Per-Class Metrics, and Confusion Matrix.
"""

from typing import List, Dict, Any
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
from src.intent.taxonomy import INTENT_LABELS

def evaluate_intent_predictions(
    ground_truth: List[str],
    predictions: List[str]
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics for intent predictions."""
    acc = float(accuracy_score(ground_truth, predictions))
    macro_f1 = float(f1_score(ground_truth, predictions, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(ground_truth, predictions, average="weighted", zero_division=0))
    macro_p = float(precision_score(ground_truth, predictions, average="macro", zero_division=0))
    macro_r = float(recall_score(ground_truth, predictions, average="macro", zero_division=0))

    cm = confusion_matrix(ground_truth, predictions, labels=INTENT_LABELS).tolist()
    report = classification_report(
        ground_truth,
        predictions,
        labels=INTENT_LABELS,
        output_dict=True,
        zero_division=0
    )

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "confusion_matrix": cm,
        "per_class_metrics": {
            lbl: {
                "precision": round(report[lbl]["precision"], 4),
                "recall": round(report[lbl]["recall"], 4),
                "f1_score": round(report[lbl]["f1-score"], 4),
                "support": report[lbl]["support"]
            }
            for lbl in INTENT_LABELS if lbl in report
        }
    }
