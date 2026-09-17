"""
Human Evaluation Protocol & Agreement Analysis Module for CompanySupport.
Generates and maintains a 50-example human evaluation CSV from held-out Golden set predictions.
Strictly distinguishes NOT_YET_ANNOTATED from HUMAN_ANNOTATED.
Agreement metrics (Cohen's Kappa, Pearson r, Spearman rho, MAD) are calculated
against HUMAN scores matched by stable example IDs across composite and groundedness dimensions.
Zero scores are fabricated when un-annotated.
"""

import csv
import json
import os
from typing import Dict, Any, List, Optional, Tuple, Union

from evaluation.metrics_utils import (
    compute_cohens_kappa,
    compute_pearson_r,
    compute_spearman_rho,
    compute_mad
)

HUMAN_SAMPLE_COLUMNS = [
    "id",
    "customer_text",
    "predicted_intent",
    "predicted_action",
    "predicted_reason",
    "generated_reply",
    "human_relevance_1to5",
    "human_groundedness_1to5",
    "human_correctness_1to5",
    "human_helpfulness_1to5",
    "human_tone_1to5",
    "human_composite_score",
    "human_escalation_action",
    "human_notes",
    "status"
]

def generate_human_eval_sample(
    evaluation_records: List[Dict[str, Any]],
    output_path: str = "data/human_eval_sample.csv",
    sample_size: int = 50,
    seed: int = 42
) -> str:
    """
    Generate or synchronize the 50-example CSV template from held-out Golden set predictions.
    
    Guarantees:
    1. If output_path already exists with sample_size rows:
       - Preserves the EXACT set and order of stable example IDs.
       - Preserves any existing human annotations (ratings, notes, status).
       - Updates agent-generated replies/predictions for those exact IDs.
    2. If output_path does NOT exist:
       - Deterministically samples sample_size examples across intents.
       - All human rating columns are strictly empty with status='NOT_YET_ANNOTATED'.
    """
    import random
    rng = random.Random(seed)

    records_by_id = {str(r.get("id", "")).strip(): r for r in evaluation_records}

    # If the sample file already exists, maintain stable IDs and preserve any annotations
    if os.path.exists(output_path):
        existing_rows: List[Dict[str, Any]] = []
        try:
            with open(output_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_rows.append(row)
        except Exception:
            existing_rows = []

        if existing_rows:
            updated_rows = []
            for ex_row in existing_rows:
                r_id = str(ex_row.get("id", "")).strip()
                agent_rec = records_by_id.get(r_id)
                row_dict = {
                    "id": r_id,
                    "customer_text": agent_rec.get("customer_text", ex_row.get("customer_text", "")) if agent_rec else ex_row.get("customer_text", ""),
                    "predicted_intent": agent_rec.get("pred_intent", ex_row.get("predicted_intent", "")) if agent_rec else ex_row.get("predicted_intent", ""),
                    "predicted_action": agent_rec.get("pred_action", ex_row.get("predicted_action", "")) if agent_rec else ex_row.get("predicted_action", ""),
                    "predicted_reason": agent_rec.get("pred_reason", ex_row.get("predicted_reason", "")) if agent_rec else ex_row.get("predicted_reason", ""),
                    "generated_reply": agent_rec.get("generated_reply", ex_row.get("generated_reply", "")) if agent_rec else ex_row.get("generated_reply", ""),
                    "human_relevance_1to5": ex_row.get("human_relevance_1to5") or "",
                    "human_groundedness_1to5": ex_row.get("human_groundedness_1to5") or "",
                    "human_correctness_1to5": ex_row.get("human_correctness_1to5") or "",
                    "human_helpfulness_1to5": ex_row.get("human_helpfulness_1to5") or "",
                    "human_tone_1to5": ex_row.get("human_tone_1to5") or "",
                    "human_composite_score": ex_row.get("human_composite_score") or "",
                    "human_escalation_action": ex_row.get("human_escalation_action") or "",
                    "human_notes": ex_row.get("human_notes") or "",
                    "status": ex_row.get("status") or "NOT_YET_ANNOTATED"
                }
                updated_rows.append(row_dict)

            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=HUMAN_SAMPLE_COLUMNS)
                writer.writeheader()
                writer.writerows(updated_rows)

            return output_path

    # Fresh deterministic generation across intents
    if len(evaluation_records) <= sample_size:
        sampled_records = list(evaluation_records)
    else:
        by_intent: Dict[str, List[Dict[str, Any]]] = {}
        for r in evaluation_records:
            by_intent.setdefault(r.get("pred_intent", "software_os"), []).append(r)
        
        sampled_records = []
        per_class = max(1, sample_size // len(by_intent))
        for intent, items in by_intent.items():
            shuffled = list(items)
            rng.shuffle(shuffled)
            sampled_records.extend(shuffled[:per_class])

        remaining = [r for r in evaluation_records if r not in sampled_records]
        rng.shuffle(remaining)
        needed = sample_size - len(sampled_records)
        if needed > 0:
            sampled_records.extend(remaining[:needed])

    rows = []
    for r in sampled_records[:sample_size]:
        rows.append({
            "id": r.get("id", ""),
            "customer_text": r.get("customer_text", ""),
            "predicted_intent": r.get("pred_intent", ""),
            "predicted_action": r.get("pred_action", ""),
            "predicted_reason": r.get("pred_reason", ""),
            "generated_reply": r.get("generated_reply", ""),
            "human_relevance_1to5": "",
            "human_groundedness_1to5": "",
            "human_correctness_1to5": "",
            "human_helpfulness_1to5": "",
            "human_tone_1to5": "",
            "human_composite_score": "",
            "human_escalation_action": "",
            "human_notes": "",
            "status": "NOT_YET_ANNOTATED"
        })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HUMAN_SAMPLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return output_path

def compute_human_agreement_statistics(
    csv_path: str = "data/human_eval_sample.csv",
    llm_evaluations: Optional[Union[Dict[str, Any], str, List[Dict[str, Any]]]] = None,
    llm_scores: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Calculate inter-rater agreement statistics comparing LLM Judge scores against HUMAN scores.
    Matched strictly by stable example IDs.
    
    Includes:
    - Composite Score: Pearson r, Spearman rho, Mean Absolute Difference (MAD)
    - Groundedness / Evidence: Pearson r, Spearman rho, MAD
    - Rubric Dimensional Breakdown: Relevance, Correctness, Helpfulness, Tone
    - Escalation Action Agreement: Cohen's Kappa
    
    STRICT RULE: If no genuine human ratings exist, returns status='NOT_YET_ANNOTATED'
    and None for all agreement metrics. Zero scores are fabricated.
    """
    if not os.path.exists(csv_path):
        return {
            "status": "NOT_AVAILABLE",
            "message": f"Human evaluation sample not found at {csv_path}. Run generate_human_eval_sample first.",
            "annotated_count": 0,
            "total_samples": 0,
            "composite_pearson_correlation": None,
            "composite_spearman_rank_correlation": None,
            "composite_mean_absolute_difference": None,
            "groundedness_pearson_correlation": None,
            "groundedness_spearman_rank_correlation": None,
            "groundedness_mean_absolute_difference": None,
            "dimensional_agreement": None,
            "cohens_kappa_escalation": None,
            "pearson_correlation": None,
            "spearman_rank_correlation": None,
            "mean_absolute_difference": None
        }

    annotated_rows: List[Dict[str, Any]] = []
    total_samples = 0

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_samples += 1
            status = row.get("status", "").strip().upper()
            score_str = row.get("human_composite_score", "").strip()
            if status == "HUMAN_ANNOTATED" or (score_str and score_str != ""):
                try:
                    score = float(score_str)
                    annotated_rows.append(row)
                except ValueError:
                    pass

    annotated_count = len(annotated_rows)

    if annotated_count == 0:
        return {
            "status": "NOT_YET_ANNOTATED",
            "message": (
                f"Human evaluation sample of {total_samples} examples located at {csv_path}. "
                "Human annotations are currently pending. No agreement scores were fabricated."
            ),
            "annotated_count": 0,
            "total_samples": total_samples,
            "composite_pearson_correlation": None,
            "composite_spearman_rank_correlation": None,
            "composite_mean_absolute_difference": None,
            "groundedness_pearson_correlation": None,
            "groundedness_spearman_rank_correlation": None,
            "groundedness_mean_absolute_difference": None,
            "dimensional_agreement": None,
            "cohens_kappa_escalation": None,
            "pearson_correlation": None,
            "spearman_rank_correlation": None,
            "mean_absolute_difference": None
        }

    # Extract human composite scores and escalation actions
    human_composite_scores = [float(r["human_composite_score"]) for r in annotated_rows]
    human_actions = [r.get("human_escalation_action", "").strip() for r in annotated_rows]
    agent_actions = [r.get("predicted_action", "").strip() for r in annotated_rows]

    # Resolve LLM Judge evaluations dictionary keyed by stable example ID
    judge_map: Dict[str, Dict[str, Any]] = {}
    if isinstance(llm_evaluations, str) and os.path.exists(llm_evaluations):
        try:
            with open(llm_evaluations, encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict) and "evaluations" in loaded and isinstance(loaded["evaluations"], dict):
                    judge_map = {str(k).strip(): v for k, v in loaded["evaluations"].items()}
                elif isinstance(loaded, dict):
                    judge_map = {str(k).strip(): v for k, v in loaded.items() if isinstance(v, dict)}
                elif isinstance(loaded, list):
                    judge_map = {str(item.get("id")).strip(): item for item in loaded if isinstance(item, dict) and "id" in item}
        except Exception:
            judge_map = {}
    elif isinstance(llm_evaluations, dict):
        if "evaluations" in llm_evaluations and isinstance(llm_evaluations["evaluations"], dict):
            judge_map = {str(k).strip(): v for k, v in llm_evaluations["evaluations"].items()}
        else:
            judge_map = {str(k).strip(): v for k, v in llm_evaluations.items() if isinstance(v, dict)}
    elif isinstance(llm_evaluations, list):
        judge_map = {str(item.get("id")).strip(): item for item in llm_evaluations if isinstance(item, dict) and "id" in item}
    elif llm_evaluations is None and os.path.exists("data/llm_judge_evaluations.json"):
        try:
            with open("data/llm_judge_evaluations.json", encoding="utf-8") as f:
                loaded = json.load(f)
                if loaded.get("status") == "EXECUTED" and "evaluations" in loaded:
                    judge_map = {str(k).strip(): v for k, v in loaded["evaluations"].items()}
        except Exception:
            judge_map = {}

    # Match human annotations against LLM judge evaluations by stable ID
    paired_composite: List[Tuple[float, float]] = []
    paired_groundedness: List[Tuple[float, float]] = []
    paired_relevance: List[Tuple[float, float]] = []
    paired_correctness: List[Tuple[float, float]] = []
    paired_helpfulness: List[Tuple[float, float]] = []
    paired_tone: List[Tuple[float, float]] = []

    for r in annotated_rows:
        row_id = str(r.get("id", "")).strip()
        if row_id in judge_map:
            j_eval = judge_map[row_id]
            dim_scores = j_eval.get("dimensional_scores", {}) if isinstance(j_eval.get("dimensional_scores"), dict) else {}

            # Composite pairing
            h_comp = r.get("human_composite_score", "").strip()
            j_comp = j_eval.get("composite_score")
            if h_comp and j_comp is not None:
                try:
                    paired_composite.append((float(h_comp), float(j_comp)))
                except ValueError:
                    pass

            # Groundedness / evidence pairing
            h_grd = r.get("human_groundedness_1to5", "").strip()
            j_grd = j_eval.get("groundedness", dim_scores.get("groundedness"))
            if h_grd and j_grd is not None:
                try:
                    paired_groundedness.append((float(h_grd), float(j_grd)))
                except ValueError:
                    pass

            # Relevance pairing
            h_rel = r.get("human_relevance_1to5", "").strip()
            j_rel = j_eval.get("relevance", dim_scores.get("relevance"))
            if h_rel and j_rel is not None:
                try:
                    paired_relevance.append((float(h_rel), float(j_rel)))
                except ValueError:
                    pass

            # Correctness pairing
            h_cor = r.get("human_correctness_1to5", "").strip()
            j_cor = j_eval.get("correctness", dim_scores.get("correctness"))
            if h_cor and j_cor is not None:
                try:
                    paired_correctness.append((float(h_cor), float(j_cor)))
                except ValueError:
                    pass

            # Helpfulness pairing
            h_hlp = r.get("human_helpfulness_1to5", "").strip()
            j_hlp = j_eval.get("helpfulness", dim_scores.get("helpfulness"))
            if h_hlp and j_hlp is not None:
                try:
                    paired_helpfulness.append((float(h_hlp), float(j_hlp)))
                except ValueError:
                    pass

            # Tone pairing
            h_ton = r.get("human_tone_1to5", "").strip()
            j_ton = j_eval.get("tone", dim_scores.get("tone"))
            if h_ton and j_ton is not None:
                try:
                    paired_tone.append((float(h_ton), float(j_ton)))
                except ValueError:
                    pass

    # Legacy flat-list support if no dictionary provided
    if not paired_composite and llm_scores and len(llm_scores) == annotated_count:
        paired_composite = list(zip(human_composite_scores, [float(s) for s in llm_scores]))

    def _calc_pair_stats(pairs: List[Tuple[float, float]]) -> Dict[str, Optional[float]]:
        if not pairs or len(pairs) < 2:
            return {
                "pearson_correlation": None,
                "spearman_rank_correlation": None,
                "mean_absolute_difference": None,
                "paired_count": len(pairs),
                "sample_count": len(pairs)
            }
        x, y = zip(*pairs)
        return {
            "pearson_correlation": round(compute_pearson_r(list(x), list(y)), 4),
            "spearman_rank_correlation": round(compute_spearman_rho(list(x), list(y)), 4),
            "mean_absolute_difference": round(compute_mad(list(x), list(y)), 4),
            "paired_count": len(pairs),
            "sample_count": len(pairs)
        }

    comp_stats = _calc_pair_stats(paired_composite)
    grd_stats = _calc_pair_stats(paired_groundedness)
    rel_stats = _calc_pair_stats(paired_relevance)
    cor_stats = _calc_pair_stats(paired_correctness)
    hlp_stats = _calc_pair_stats(paired_helpfulness)
    ton_stats = _calc_pair_stats(paired_tone)

    # Cohen's Kappa for escalation action (human vs agent action)
    valid_action_pairs = [(h, a) for h, a in zip(human_actions, agent_actions) if h and a]
    if valid_action_pairs and len(valid_action_pairs) >= 2:
        h_acts, a_acts = zip(*valid_action_pairs)
        cohens_kappa = round(compute_cohens_kappa(list(h_acts), list(a_acts)), 4)
    else:
        cohens_kappa = None

    return {
        "status": "HUMAN_ANNOTATED",
        "annotated_count": annotated_count,
        "total_samples": total_samples,
        "mean_human_composite_score": round(sum(human_composite_scores) / annotated_count, 2),
        "paired_judge_evaluations_count": len(paired_composite),
        
        # LLM Judge vs Human Agreement: Composite
        "composite_pearson_correlation": comp_stats["pearson_correlation"],
        "composite_spearman_rank_correlation": comp_stats["spearman_rank_correlation"],
        "composite_mean_absolute_difference": comp_stats["mean_absolute_difference"],
        
        # LLM Judge vs Human Agreement: Groundedness / Evidence
        "groundedness_pearson_correlation": grd_stats["pearson_correlation"],
        "groundedness_spearman_rank_correlation": grd_stats["spearman_rank_correlation"],
        "groundedness_mean_absolute_difference": grd_stats["mean_absolute_difference"],
        
        # Dimensional Agreement Breakdown
        "dimensional_agreement": {
            "groundedness": grd_stats,
            "relevance": rel_stats,
            "correctness": cor_stats,
            "helpfulness": hlp_stats,
            "tone": ton_stats
        },
        
        # Escalation policy agreement
        "cohens_kappa_escalation": cohens_kappa,
        
        # Backward-compatibility aliases
        "pearson_correlation": comp_stats["pearson_correlation"],
        "spearman_rank_correlation": comp_stats["spearman_rank_correlation"],
        "mean_absolute_difference": comp_stats["mean_absolute_difference"]
    }
