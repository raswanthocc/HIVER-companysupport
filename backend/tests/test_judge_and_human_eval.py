"""
Unit Tests for LLM Judge, Human Evaluation Pipeline, and Inter-Annotator Agreement.

Verifies:
1. LLM Judge evaluates the exact same 50 examples in data/human_eval_sample.csv.
2. Per-example LLM judge scores are structured with stable example IDs.
3. Human scores can be matched to those exact IDs.
4. judge_agreement.py compares LLM judge scores against HUMAN scores.
5. Groundedness/evidence scores are explicitly included in the comparison.
6. When unconfigured / unannotated, NO API calls are made and NO scores are fabricated.
"""

import os
import sys
import csv
import json
import tempfile
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from evaluation.llm_judge import RealLLMJudge
from evaluation.judge_agreement import (
    generate_human_eval_sample,
    compute_human_agreement_statistics,
    HUMAN_SAMPLE_COLUMNS
)

def test_human_eval_sample_format_and_count():
    """Verify that data/human_eval_sample.csv contains exactly 50 un-annotated examples with valid IDs."""
    sample_path = os.path.join(BASE_DIR, "data", "human_eval_sample.csv")
    assert os.path.exists(sample_path), "data/human_eval_sample.csv must exist"

    with open(sample_path, encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 50, f"Expected exactly 50 human evaluation examples, got {len(reader)}"
    
    ids = [r["id"] for r in reader]
    assert len(set(ids)) == 50, "All 50 sample IDs must be unique"
    for r_id in ids:
        assert r_id.isdigit(), f"ID {r_id} must be an integer string matching Golden set IDs"

    for r in reader:
        assert r["status"] in ["NOT_YET_ANNOTATED", "HUMAN_ANNOTATED"], "Status must be valid"

def test_unconfigured_judge_zero_calls_and_zero_fabrication():
    """Verify RealLLMJudge returns NOT_EXECUTED without making API calls when unconfigured."""
    judge = RealLLMJudge(api_key="", base_url="", model="")
    judge.provider = "none"
    judge.api_key = None
    judge.base_url = None

    assert not judge.is_configured()

    # Single reply test
    single_res = judge.evaluate_reply("Test query", "Test reply", "software_os", "auto_handle", "self_service")
    assert single_res["status"] == "NOT_EXECUTED"
    assert single_res["composite_score"] is None
    assert single_res["dimensional_scores"] is None

    # Batch 50-example test
    sample_path = os.path.join(BASE_DIR, "data", "human_eval_sample.csv")
    batch_res = judge.evaluate_human_eval_sample(sample_csv_path=sample_path, output_json_path=None)
    assert batch_res["status"] == "NOT_EXECUTED"
    assert batch_res["evaluated_samples"] == 0
    assert batch_res["composite_score"] is None
    assert batch_res["dimensional_scores"] is None
    assert batch_res["evaluations"] == {}

def test_missing_gemini_key_behavior_and_secure_dotenv_loading(monkeypatch):
    """
    Verify:
    1. Missing/empty GEMINI_API_KEY causes RealLLMJudge to report NOT_EXECUTED without making network calls.
    2. python-dotenv successfully loads GEMINI_API_KEY from .env when configured.
    3. The actual API key is never exposed in judge.__repr__() or results.
    4. Zero network calls occur during missing-key checks.
    """
    # Fail test loudly if any network call is attempted
    def mock_urlopen(*args, **kwargs):
        raise AssertionError("Network call attempted during missing-key test!")
    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    # Ensure environment variables are clear
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    # Test 1: Empty .env / Missing key behavior
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_env = os.path.join(tmpdir, ".env")
        with open(empty_env, "w", encoding="utf-8") as f:
            f.write("# Empty env\nGEMINI_API_KEY=\n")

        judge = RealLLMJudge(env_path=empty_env)
        assert not judge.is_configured()
        assert judge.provider == "none"

        single_res = judge.evaluate_reply("Query", "Reply", "software_os", "auto_handle", "reason")
        assert single_res["status"] == "NOT_EXECUTED"
        assert single_res["composite_score"] is None

        batch_res = judge.evaluate_human_eval_sample(
            sample_csv_path=os.path.join(BASE_DIR, "data", "human_eval_sample.csv"),
            output_json_path=None
        )
        assert batch_res["status"] == "NOT_EXECUTED"
        assert batch_res["evaluated_samples"] == 0

    # Test 2: Secure loading via python-dotenv
    with tempfile.TemporaryDirectory() as tmpdir:
        test_key = "AIzaSyTestMockKeyForDotenvLoadingOnly12345"
        mock_env = os.path.join(tmpdir, ".env")
        with open(mock_env, "w", encoding="utf-8") as f:
            f.write(f"GEMINI_API_KEY={test_key}\n")

        judge_with_key = RealLLMJudge(env_path=mock_env)
        assert judge_with_key.is_configured()
        assert judge_with_key.provider == "gemini"
        assert "gemini" in judge_with_key.model
        
        # Verify that repr never exposes the secret key
        rep = repr(judge_with_key)
        assert test_key not in rep

def test_stable_id_preservation_and_annotation_protection():
    """Verify generate_human_eval_sample preserves existing IDs and annotations on update."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_csv = os.path.join(tmpdir, "test_sample.csv")
        
        # Initial dummy records
        records = [
            {"id": i, "customer_text": f"text {i}", "pred_intent": "software_os", "pred_action": "auto_handle", "pred_reason": "r", "generated_reply": f"reply {i}"}
            for i in range(50)
        ]
        generate_human_eval_sample(records, output_path=tmp_csv, sample_size=50)

        # Simulate human annotation on row 0 (id '0')
        rows = []
        with open(tmp_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        
        rows[0]["human_composite_score"] = "4.5"
        rows[0]["human_groundedness_1to5"] = "5"
        rows[0]["status"] = "HUMAN_ANNOTATED"
        
        with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HUMAN_SAMPLE_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        # Re-run generate_human_eval_sample with updated agent replies
        updated_records = [
            {"id": i, "customer_text": f"text {i}", "pred_intent": "software_os", "pred_action": "auto_handle", "pred_reason": "r", "generated_reply": f"NEW reply {i}"}
            for i in range(50)
        ]
        generate_human_eval_sample(updated_records, output_path=tmp_csv, sample_size=50)

        # Verify preservation
        with open(tmp_csv, encoding="utf-8") as f:
            reloaded = list(csv.DictReader(f))

        assert len(reloaded) == 50
        assert reloaded[0]["id"] == "0"
        assert reloaded[0]["human_composite_score"] == "4.5", "Human annotation must be preserved!"
        assert reloaded[0]["human_groundedness_1to5"] == "5", "Human groundedness must be preserved!"
        assert reloaded[0]["status"] == "HUMAN_ANNOTATED", "Status must be preserved!"
        assert reloaded[0]["generated_reply"] == "NEW reply 0", "Agent reply should be synced!"

def test_repeated_eval_runs_preserve_exact_ids_and_annotations():
    """
    Regression Test: Verify that across multiple repeated evaluation cycles:
    1. Exact same 50 sample IDs are preserved in identical order.
    2. Any existing human annotations are never overwritten or cleared.
    3. Unannotated rows remain NOT_YET_ANNOTATED.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_csv = os.path.join(tmpdir, "human_eval_sample.csv")
        
        # Initial 50 records
        base_records = [
            {
                "id": 100 + i,
                "customer_text": f"Inquiry {100 + i}",
                "pred_intent": "software_os",
                "pred_action": "auto_handle",
                "pred_reason": "self_service_troubleshooting",
                "generated_reply": f"Initial Reply {100 + i}"
            }
            for i in range(50)
        ]
        generate_human_eval_sample(base_records, output_path=tmp_csv, sample_size=50)

        # Record initial IDs
        with open(tmp_csv, encoding="utf-8") as f:
            initial_rows = list(csv.DictReader(f))
        initial_ids = [r["id"] for r in initial_rows]
        assert len(initial_ids) == 50

        # Annotate multiple specific rows: first, middle, last
        annotated_indices = [0, 24, 49]
        for idx in annotated_indices:
            initial_rows[idx]["human_relevance_1to5"] = "5"
            initial_rows[idx]["human_groundedness_1to5"] = "4"
            initial_rows[idx]["human_correctness_1to5"] = "5"
            initial_rows[idx]["human_helpfulness_1to5"] = "4"
            initial_rows[idx]["human_tone_1to5"] = "5"
            initial_rows[idx]["human_composite_score"] = "4.6"
            initial_rows[idx]["human_escalation_action"] = "auto_handle"
            initial_rows[idx]["human_notes"] = f"Human verified item {initial_ids[idx]}"
            initial_rows[idx]["status"] = "HUMAN_ANNOTATED"

        with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HUMAN_SAMPLE_COLUMNS)
            writer.writeheader()
            writer.writerows(initial_rows)

        # Simulate 3 repeated evaluation runs with updated agent responses
        for run_idx in range(1, 4):
            updated_records = [
                {
                    "id": 100 + i,
                    "customer_text": f"Inquiry {100 + i}",
                    "pred_intent": "software_os",
                    "pred_action": "auto_handle",
                    "pred_reason": "self_service_troubleshooting",
                    "generated_reply": f"Run {run_idx} Updated Reply {100 + i}"
                }
                for i in range(50)
            ]
            generate_human_eval_sample(updated_records, output_path=tmp_csv, sample_size=50)

            # Audit file after each run
            with open(tmp_csv, encoding="utf-8") as f:
                current_rows = list(csv.DictReader(f))

            current_ids = [r["id"] for r in current_rows]
            assert current_ids == initial_ids, f"Run {run_idx}: IDs changed or reordered!"

            # Verify annotated rows are 100% preserved
            for idx in annotated_indices:
                row = current_rows[idx]
                assert row["human_composite_score"] == "4.6", f"Run {run_idx}: Human score lost on {row['id']}"
                assert row["human_groundedness_1to5"] == "4", f"Run {run_idx}: Groundedness lost on {row['id']}"
                assert row["human_relevance_1to5"] == "5"
                assert row["human_notes"] == f"Human verified item {initial_ids[idx]}"
                assert row["status"] == "HUMAN_ANNOTATED"
                assert row["generated_reply"] == f"Run {run_idx} Updated Reply {initial_ids[idx]}"

            # Verify unannotated rows remain NOT_YET_ANNOTATED
            for idx in range(50):
                if idx not in annotated_indices:
                    row = current_rows[idx]
                    assert row["status"] == "NOT_YET_ANNOTATED"
                    assert row["human_composite_score"] == ""
                    assert row["human_groundedness_1to5"] == ""


def test_agreement_statistics_on_unannotated_sample():
    """Verify compute_human_agreement_statistics returns expected status."""
    sample_path = os.path.join(BASE_DIR, "data", "human_eval_sample.csv")
    stats = compute_human_agreement_statistics(sample_path)
    
    assert stats["status"] in ["NOT_YET_ANNOTATED", "HUMAN_ANNOTATED"]

def test_agreement_statistics_with_mock_annotations_and_judge_evaluations():
    """
    Verify that compute_human_agreement_statistics accurately compares LLM Judge vs Human scores,
    including groundedness/evidence scores matched strictly by stable example IDs.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_csv = os.path.join(tmpdir, "annotated_sample.csv")
        
        # 5 sample examples with distinct IDs
        sample_ids = ["26", "76", "199", "11", "194"]
        
        human_rows = [
            {
                "id": "26",
                "customer_text": "Battery issue",
                "predicted_intent": "software_os",
                "predicted_action": "escalate",
                "predicted_reason": "high_churn_frustration",
                "generated_reply": "Reply 26",
                "human_relevance_1to5": "5",
                "human_groundedness_1to5": "5",
                "human_correctness_1to5": "5",
                "human_helpfulness_1to5": "4",
                "human_tone_1to5": "5",
                "human_composite_score": "4.8",
                "human_escalation_action": "escalate",
                "human_notes": "Great response",
                "status": "HUMAN_ANNOTATED"
            },
            {
                "id": "76",
                "customer_text": "Frozen phone",
                "predicted_intent": "software_os",
                "predicted_action": "auto_handle",
                "predicted_reason": "self_service_troubleshooting",
                "generated_reply": "Reply 76",
                "human_relevance_1to5": "4",
                "human_groundedness_1to5": "4",
                "human_correctness_1to5": "4",
                "human_helpfulness_1to5": "4",
                "human_tone_1to5": "4",
                "human_composite_score": "4.0",
                "human_escalation_action": "auto_handle",
                "human_notes": "Solid advice",
                "status": "HUMAN_ANNOTATED"
            },
            {
                "id": "199",
                "customer_text": "Emoji bug",
                "predicted_intent": "software_os",
                "predicted_action": "auto_handle",
                "predicted_reason": "self_service_troubleshooting",
                "generated_reply": "Reply 199",
                "human_relevance_1to5": "3",
                "human_groundedness_1to5": "3",
                "human_correctness_1to5": "3",
                "human_helpfulness_1to5": "3",
                "human_tone_1to5": "3",
                "human_composite_score": "3.0",
                "human_escalation_action": "auto_handle",
                "human_notes": "Generic",
                "status": "HUMAN_ANNOTATED"
            },
            {
                "id": "11",
                "customer_text": "Battery draining 6s",
                "predicted_intent": "software_os",
                "predicted_action": "auto_handle",
                "predicted_reason": "self_service_troubleshooting",
                "generated_reply": "Reply 11",
                "human_relevance_1to5": "2",
                "human_groundedness_1to5": "2",
                "human_correctness_1to5": "2",
                "human_helpfulness_1to5": "2",
                "human_tone_1to5": "3",
                "human_composite_score": "2.2",
                "human_escalation_action": "auto_handle",
                "human_notes": "Suboptimal",
                "status": "HUMAN_ANNOTATED"
            },
            {
                "id": "194",
                "customer_text": "Error 21 update fail",
                "predicted_intent": "software_os",
                "predicted_action": "auto_handle",
                "predicted_reason": "self_service_troubleshooting",
                "generated_reply": "Reply 194",
                "human_relevance_1to5": "1",
                "human_groundedness_1to5": "1",
                "human_correctness_1to5": "1",
                "human_helpfulness_1to5": "1",
                "human_tone_1to5": "2",
                "human_composite_score": "1.2",
                "human_escalation_action": "escalate",
                "human_notes": "Wrong guidance",
                "status": "HUMAN_ANNOTATED"
            }
        ]

        with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HUMAN_SAMPLE_COLUMNS)
            writer.writeheader()
            writer.writerows(human_rows)

        # Paired LLM judge evaluations keyed by the exact same stable IDs
        mock_judge_evaluations = {
            "status": "EXECUTED",
            "evaluations": {
                "26": {
                    "id": 26,
                    "relevance": 5,
                    "groundedness": 5,
                    "correctness": 5,
                    "helpfulness": 4,
                    "tone": 5,
                    "composite_score": 4.8
                },
                "76": {
                    "id": 76,
                    "relevance": 4,
                    "groundedness": 4,
                    "correctness": 4,
                    "helpfulness": 4,
                    "tone": 4,
                    "composite_score": 4.0
                },
                "199": {
                    "id": 199,
                    "relevance": 3,
                    "groundedness": 3,
                    "correctness": 3,
                    "helpfulness": 3,
                    "tone": 3,
                    "composite_score": 3.0
                },
                "11": {
                    "id": 11,
                    "relevance": 2,
                    "groundedness": 2,
                    "correctness": 2,
                    "helpfulness": 2,
                    "tone": 3,
                    "composite_score": 2.2
                },
                "194": {
                    "id": 194,
                    "relevance": 1,
                    "groundedness": 1,
                    "correctness": 1,
                    "helpfulness": 1,
                    "tone": 2,
                    "composite_score": 1.2
                }
            }
        }

        stats = compute_human_agreement_statistics(tmp_csv, llm_evaluations=mock_judge_evaluations)

        assert stats["status"] == "HUMAN_ANNOTATED"
        assert stats["annotated_count"] == 5
        assert stats["paired_judge_evaluations_count"] == 5

        # Verify Composite Score comparison (LLM Judge vs Human)
        assert stats["composite_pearson_correlation"] is not None
        assert stats["composite_pearson_correlation"] > 0.95, "Pearson r should reflect strong correlation"
        assert stats["composite_mean_absolute_difference"] == 0.0, "MAD should be 0.0 for identical scores"

        # Verify Groundedness / Evidence score comparison (Requirement 5)
        assert stats["groundedness_pearson_correlation"] is not None
        assert stats["groundedness_pearson_correlation"] > 0.95
        assert stats["groundedness_mean_absolute_difference"] == 0.0
        assert "groundedness" in stats["dimensional_agreement"]
        assert stats["dimensional_agreement"]["groundedness"]["sample_count"] == 5

        # Verify Escalation action agreement (Cohen's Kappa)
        assert stats["cohens_kappa_escalation"] is not None
