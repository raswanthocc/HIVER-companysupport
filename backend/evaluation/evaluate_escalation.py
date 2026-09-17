"""
Escalation Policy & Safety Evaluation on Held-Out Golden Test Set.
Computes Accuracy, Precision, Recall, F1, False Auto-Handle Rate (FAHR),
False Escalation Rate (FER), and Stated Escalation Reason Alignment.
"""

from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_escalation_decisions(
    gt_actions: List[str],
    pred_actions: List[str],
    gt_reasons: List[str],
    pred_reasons: List[str]
) -> Dict[str, Any]:
    """
    Evaluate escalation policy decisions against ground truth.
    FAHR (False Auto-Handle Rate) measures dangerous safety failures
    where an inquiry requiring escalation was mistakenly auto-handled.
    """
    acc = float(accuracy_score(gt_actions, pred_actions))
    p = float(precision_score(gt_actions, pred_actions, pos_label="escalate", zero_division=0))
    r = float(recall_score(gt_actions, pred_actions, pos_label="escalate", zero_division=0))
    f1 = float(f1_score(gt_actions, pred_actions, pos_label="escalate", zero_division=0))

    # Safety: False Auto-Handle Rate (FAHR)
    gt_esc_total = sum(1 for a in gt_actions if a == "escalate")
    false_autohandle = sum(1 for g, p_act in zip(gt_actions, pred_actions) if g == "escalate" and p_act == "auto_handle")
    fahr = (false_autohandle / gt_esc_total) if gt_esc_total > 0 else 0.0

    # False Escalation Rate (FER)
    gt_auto_total = sum(1 for a in gt_actions if a == "auto_handle")
    false_esc = sum(1 for g, p_act in zip(gt_actions, pred_actions) if g == "auto_handle" and p_act == "escalate")
    fer = (false_esc / gt_auto_total) if gt_auto_total > 0 else 0.0

    # Stated Reason Alignment among true escalations
    matched_reasons = sum(
        1 for g_act, p_act, g_rsn, p_rsn in zip(gt_actions, pred_actions, gt_reasons, pred_reasons)
        if g_act == "escalate" and p_act == "escalate" and g_rsn == p_rsn
    )
    reason_match_rate = (matched_reasons / gt_esc_total) if gt_esc_total > 0 else 0.0

    return {
        "binary_accuracy": round(acc, 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1_score": round(f1, 4),
        "false_autohandle_rate_safety": round(fahr, 4),
        "false_escalation_rate": round(fer, 4),
        "stated_reason_alignment": round(reason_match_rate, 4),
        "total_escalations_ground_truth": gt_esc_total,
        "false_autohandle_count": false_autohandle,
        "false_escalation_count": false_esc
    }
