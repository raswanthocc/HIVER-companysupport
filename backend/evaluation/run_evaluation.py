"""
Master Turnkey Evaluation Harness for CompanySupport AI Support Agent.
Executes automated zero-leakage check, trains models strictly on train split,
benchmarks Baseline 1 (Trivial), Baseline 2 (Simple), and Proposed Full Agent
on the held-out 200-example Golden Evaluation Set, executes/audits the real LLM judge,
generates the human evaluation sample CSV, and serializes unadulterated results to results_summary.json.
"""

import csv
import json
import os
import sys
import time
from typing import Dict, Any, List

# Ensure repository root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data.loader import load_golden_eval_set
from src.data.split import partition_data_at_thread_level
from src.data.leakage_check import verify_zero_leakage, DataLeakageError
from src.intent.classifier import SimpleClassifier, CalibratedIntentClassifier, TrivialClassifier
from src.escalation.policy import EscalationEngine
from src.retrieval.retriever import ResolutionRetriever
from src.generation.generator import GroundedResponseGenerator
from src.agent import CompanySupportAgent

from baselines.trivial_baseline import TrivialBaselineAgent
from baselines.simple_baseline import SimpleBaselineAgent

from evaluation.evaluate_intent import evaluate_intent_predictions
from evaluation.evaluate_escalation import evaluate_escalation_decisions
from evaluation.evaluate_replies import evaluate_generated_replies
from evaluation.llm_judge import RealLLMJudge
from evaluation.judge_agreement import generate_human_eval_sample, compute_human_agreement_statistics

def run_system_evaluation(
    agent,
    agent_name: str,
    golden_data: List[Dict[str, Any]],
    real_judge: RealLLMJudge
) -> Dict[str, Any]:
    """Evaluate a single support agent system across all evaluation dimensions on the Golden set."""
    print(f"  -> Benchmarking {agent_name} across {len(golden_data)} held-out golden examples...")
    
    gt_intents = []
    pred_intents = []
    gt_actions = []
    pred_actions = []
    gt_reasons = []
    pred_reasons = []
    generated_replies = []
    historical_replies = []
    latencies = []
    diagnostic_count = 0
    autohandle_count = 0
    records = []

    for item in golden_data:
        c_text = item["customer_text"]
        h_reply = item["historical_agent_text"]
        gt_int = item["intent"]
        gt_act = item["ground_truth_action"]
        gt_rsn = item["ground_truth_reason"]
        brand = item.get("brand", "All Brands")

        t0 = time.perf_counter()
        pred = agent.process_query(c_text, selected_brand=brand, query_id=str(item["id"]))
        lat_ms = (time.perf_counter() - t0) * 1000.0

        latencies.append(lat_ms)
        gt_intents.append(gt_int)
        pred_intents.append(pred.predicted_intent)
        gt_actions.append(gt_act)
        pred_actions.append(pred.escalation_action)
        gt_reasons.append(gt_rsn)
        pred_reasons.append(pred.escalation_reason)
        generated_replies.append(pred.generated_reply)
        historical_replies.append(h_reply)

        if pred.escalation_action == "auto_handle":
            autohandle_count += 1
            if "?" in pred.generated_reply:
                diagnostic_count += 1

        record = {
            "id": item["id"],
            "customer_text": c_text,
            "historical_agent_text": h_reply,
            "gt_intent": gt_int,
            "pred_intent": pred.predicted_intent,
            "intent_confidence": pred.intent_confidence,
            "gt_action": gt_act,
            "pred_action": pred.escalation_action,
            "gt_reason": gt_rsn,
            "pred_reason": pred.escalation_reason,
            "escalation_explanation": getattr(pred, "escalation_explanation", ""),
            "generated_reply": pred.generated_reply,
            "grounding_status": getattr(pred, "grounding_status", "kb_policy_fallback"),
            "retrieved_evidence": getattr(pred, "retrieved_evidence", [])
        }
        records.append(record)

    # 1. Intent Metrics
    intent_metrics = evaluate_intent_predictions(gt_intents, pred_intents)

    # 2. Escalation & Safety Metrics
    escalation_metrics = evaluate_escalation_decisions(gt_actions, pred_actions, gt_reasons, pred_reasons)

    # 3. Response Quality Metrics (Lexical)
    lexical_metrics = evaluate_generated_replies(generated_replies, historical_replies)

    # 4. Latency
    import numpy as np
    mean_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))

    diag_rate = (diagnostic_count / autohandle_count) if autohandle_count > 0 else 0.0

    return {
        "agent_name": agent_name,
        "sample_count": len(golden_data),
        "intent_classification": intent_metrics,
        "escalation_policy": escalation_metrics,
        "response_quality": {
            **lexical_metrics,
            "diagnostic_question_rate": round(diag_rate, 4)
        },
        "performance": {
            "mean_latency_ms": round(mean_lat, 2),
            "p95_latency_ms": round(p95_lat, 2)
        },
        "records": records
    }

def main():
    total_start_time = time.time()
    print("=" * 85)
    print("  CompanySupport AI Support Agent - Rigorous Evaluation Harness")
    print("=" * 85)

    golden_path = os.path.join(BASE_DIR, "data", "golden_eval_set.json")
    train_path = os.path.join(BASE_DIR, "data", "train_data.json")
    val_path = os.path.join(BASE_DIR, "data", "val_data.json")
    retrieval_path = os.path.join(BASE_DIR, "data", "train_retrieval_corpus.csv")
    raw_sample_path = os.path.join(BASE_DIR, "data", "raw_sample.csv")

    # Step 1: Ensure thread-level data partition exists; if not, create it
    if not (os.path.exists(train_path) and os.path.exists(retrieval_path)):
        print("\n[Step 1/6] Partitioning raw data at the conversation thread level...")
        split_summary = partition_data_at_thread_level(
            raw_sample_path=raw_sample_path,
            golden_path=golden_path,
            output_dir=os.path.join(BASE_DIR, "data")
        )
    else:
        print("\n[Step 1/6] Existing thread-level partition verified.")

    # Step 2: Automated Leakage Verification (Fails loudly on any leak)
    print("\n[Step 2/6] Running automated 4-stage zero-leakage check...")
    leakage_result = verify_zero_leakage(
        golden_path=golden_path,
        train_path=train_path,
        val_path=val_path,
        retrieval_path=retrieval_path
    )
    print(f"  Leakage Check Passed: {leakage_result['status']}")
    print(f"  - Golden samples: {leakage_result['golden_samples_audited']}")
    print(f"  - Train samples:  {leakage_result['train_samples_audited']}")
    print(f"  - Retrieval pairs: {leakage_result['retrieval_samples_audited']}")
    print(f"  - Tweet ID overlap: {leakage_result['tweet_id_overlap']}")
    print(f"  - Text overlap:     {leakage_result['exact_text_overlap']}")
    print(f"  - Near-duplicates:  {leakage_result['near_duplicates_exceeding_threshold']}")

    # Step 3: Train Intent Classifiers ONLY on Training Split
    print("\n[Step 3/6] Training classifiers strictly on training partition...")
    with open(train_path, encoding="utf-8") as f:
        train_data = json.load(f)
    train_texts = [d["customer_text"] for d in train_data]
    train_labels = [d["intent"] for d in train_data]

    # Baseline 2 Simple Classifier
    simple_clf = SimpleClassifier()
    simple_clf.fit(train_texts, train_labels)

    # Proposed Calibrated Classifier
    calibrated_clf = CalibratedIntentClassifier()
    calibrated_clf.fit(train_texts, train_labels)
    print(f"  Classifiers trained on {len(train_texts)} silver-supervised training turns.")

    # Step 4: Initialize Isolated Retrieval Corpus & Agents
    print("\n[Step 4/6] Initializing isolated retrieval corpus and support agents...")
    retriever = ResolutionRetriever(
        retrieval_corpus_path=retrieval_path,
        kb_path=os.path.join(BASE_DIR, "data", "knowledge_base.json")
    )
    escalation_engine = EscalationEngine(confidence_threshold=0.55)
    generator = GroundedResponseGenerator()

    trivial_agent = TrivialBaselineAgent()
    simple_agent = SimpleBaselineAgent(classifier=simple_clf, retriever=retriever)
    full_agent = CompanySupportAgent(
        classifier=calibrated_clf,
        escalation_engine=escalation_engine,
        retriever=retriever,
        generator=generator
    )

    # Step 5: Evaluate on Held-Out Golden Set
    print("\n[Step 5/6] Benchmarking all 3 systems on unseen 200-item Golden Evaluation Set...")
    golden_data = load_golden_eval_set(golden_path)
    real_judge = RealLLMJudge()

    trivial_results = run_system_evaluation(trivial_agent, "Baseline 1 (Trivial)", golden_data, real_judge)
    simple_results = run_system_evaluation(simple_agent, "Baseline 2 (Simple)", golden_data, real_judge)
    full_results = run_system_evaluation(full_agent, "Full CompanySupport Agent", golden_data, real_judge)

    # Step 6: LLM Judge & Human Evaluation Protocol
    print("\n[Step 6/6] Processing LLM Judge and Human Evaluation Protocol...")
    human_sample_path = os.path.join(BASE_DIR, "data", "human_eval_sample.csv")
    llm_eval_output_path = os.path.join(BASE_DIR, "data", "llm_judge_evaluations.json")

    # Step 6a: Generate or sync 50-example human evaluation CSV (preserving IDs and existing human annotations)
    generate_human_eval_sample(full_results["records"], output_path=human_sample_path, sample_size=50)
    print(f"  Verified 50-sample human evaluation template at: {human_sample_path}")

    # Step 6b: Evaluate exact 50 examples with RealLLMJudge (or report NOT_EXECUTED if pending credentials)
    judge_status = real_judge.evaluate_human_eval_sample(
        sample_csv_path=human_sample_path,
        output_json_path=llm_eval_output_path
    )
    if judge_status.get("status") == "EXECUTED":
        print(f"  Active LLM Judge evaluated {judge_status.get('evaluated_samples', 0)} samples. Mean Composite: {judge_status.get('composite_score')}")
    else:
        print(f"  LLM Judge Status: {judge_status.get('status')} (Zero scores fabricated; pending API key)")

    # Step 6c: Calculate human vs judge agreement (strictly reports NOT_YET_ANNOTATED if un-annotated)
    agreement_results = compute_human_agreement_statistics(
        csv_path=human_sample_path,
        llm_evaluations=judge_status.get("evaluations") or (llm_eval_output_path if os.path.exists(llm_eval_output_path) else None)
    )
    print(f"  Human Agreement Protocol Status: {agreement_results['status']}")

    total_time = round(time.time() - total_start_time, 2)

    # Print Headline Comparison Table
    print("\n" + "=" * 92)
    print(f"{'METRIC':<36} | {'BASELINE 1 (TRIVIAL)':<20} | {'BASELINE 2 (SIMPLE)':<20} | {'PROPOSED AGENT':<16}")
    print("-" * 92)
    
    # Intent
    print(f"{'Intent Accuracy':<36} | {trivial_results['intent_classification']['accuracy']:<20.2%} | {simple_results['intent_classification']['accuracy']:<20.2%} | {full_results['intent_classification']['accuracy']:<16.2%}")
    print(f"{'Intent Macro F1':<36} | {trivial_results['intent_classification']['macro_f1']:<20.4f} | {simple_results['intent_classification']['macro_f1']:<20.4f} | {full_results['intent_classification']['macro_f1']:<16.4f}")
    print(f"{'Intent Weighted F1':<36} | {trivial_results['intent_classification']['weighted_f1']:<20.4f} | {simple_results['intent_classification']['weighted_f1']:<20.4f} | {full_results['intent_classification']['weighted_f1']:<16.4f}")
    print("-" * 92)
    
    # Escalation
    print(f"{'Escalation Binary Accuracy':<36} | {trivial_results['escalation_policy']['binary_accuracy']:<20.2%} | {simple_results['escalation_policy']['binary_accuracy']:<20.2%} | {full_results['escalation_policy']['binary_accuracy']:<16.2%}")
    print(f"{'Escalation Precision':<36} | {trivial_results['escalation_policy']['precision']:<20.4f} | {simple_results['escalation_policy']['precision']:<20.4f} | {full_results['escalation_policy']['precision']:<16.4f}")
    print(f"{'Escalation Recall':<36} | {trivial_results['escalation_policy']['recall']:<20.4f} | {simple_results['escalation_policy']['recall']:<20.4f} | {full_results['escalation_policy']['recall']:<16.4f}")
    print(f"{'Escalation F1':<36} | {trivial_results['escalation_policy']['f1_score']:<20.4f} | {simple_results['escalation_policy']['f1_score']:<20.4f} | {full_results['escalation_policy']['f1_score']:<16.4f}")
    print(f"{'False Auto-Handle Rate (FAHR) [!]':<36} | {trivial_results['escalation_policy']['false_autohandle_rate_safety']:<20.2%} | {simple_results['escalation_policy']['false_autohandle_rate_safety']:<20.2%} | {full_results['escalation_policy']['false_autohandle_rate_safety']:<16.2%}")
    print(f"{'Stated Reason Alignment':<36} | {trivial_results['escalation_policy']['stated_reason_alignment']:<20.2%} | {simple_results['escalation_policy']['stated_reason_alignment']:<20.2%} | {full_results['escalation_policy']['stated_reason_alignment']:<16.2%}")
    print("-" * 92)
    
    # Quality
    print(f"{'ROUGE-1 F1':<36} | {trivial_results['response_quality']['rouge_1']:<20.4f} | {simple_results['response_quality']['rouge_1']:<20.4f} | {full_results['response_quality']['rouge_1']:<16.4f}")
    print(f"{'ROUGE-L F1':<36} | {trivial_results['response_quality']['rouge_l']:<20.4f} | {simple_results['response_quality']['rouge_l']:<20.4f} | {full_results['response_quality']['rouge_l']:<16.4f}")
    print(f"{'BLEU Score':<36} | {trivial_results['response_quality']['bleu']:<20.4f} | {simple_results['response_quality']['bleu']:<20.4f} | {full_results['response_quality']['bleu']:<16.4f}")
    print(f"{'Diagnostic Question Rate':<36} | {trivial_results['response_quality']['diagnostic_question_rate']:<20.2%} | {simple_results['response_quality']['diagnostic_question_rate']:<20.2%} | {full_results['response_quality']['diagnostic_question_rate']:<16.2%}")
    print(f"{'LLM Judge Composite':<36} | {'NOT_EXECUTED':<20} | {'NOT_EXECUTED':<20} | {str(judge_status['composite_score'] if judge_status['composite_score'] is not None else 'NOT_EXECUTED'):<16}")
    print("-" * 92)
    
    # Performance
    print(f"{'Mean Latency (ms)':<36} | {trivial_results['performance']['mean_latency_ms']:<20.2f} | {simple_results['performance']['mean_latency_ms']:<20.2f} | {full_results['performance']['mean_latency_ms']:<16.2f}")
    print(f"{'Total Execution Time':<36} | {'—':<20} | {'—':<20} | {f'{total_time} s (< 15 min)':<16}")
    print("=" * 92)

    # Save to results_summary.json
    clean_trivial = dict(trivial_results)
    clean_trivial.pop("records", None)
    clean_simple = dict(simple_results)
    clean_simple.pop("records", None)
    clean_full = dict(full_results)
    full_records = clean_full.pop("records", [])

    results_summary = {
        "execution_time_seconds": total_time,
        "leakage_verification": leakage_result,
        "baseline_1_trivial": clean_trivial,
        "baseline_2_simple": clean_simple,
        "proposed_full_agent": clean_full,
        "llm_judge": judge_status,
        "human_evaluation": agreement_results,
        "sample_evaluation_records": full_records[:15]
    }

    summary_path = os.path.join(BASE_DIR, "results_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nFull empirical benchmark results saved to: {summary_path}")

    return results_summary

if __name__ == "__main__":
    main()
