import os
import json
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.intent.classifier import CalibratedIntentClassifier

def main():
    print("Loading validation data...")
    val_path = os.path.join(base_dir, "data", "val_data.json")
    if not os.path.exists(val_path):
        print(f"File not found: {val_path}")
        return
        
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
        
    texts = [d["customer_text"] for d in val_data]
    labels = [d["intent"] for d in val_data]
    brands = [d.get("brand", "Unknown") for d in val_data]
    
    print("Loading and fitting classifier (auto-fits on train_data.json)...")
    clf = CalibratedIntentClassifier()
    clf.auto_fit_if_needed()
    clf.llm = None  # Disable API fallback to prevent hours-long sequential evaluations
    
    print("Evaluating classifier on validation set...")
    metrics = clf.evaluate(texts, labels, brands)
    
    print("\n--- Intent Classification Validation Metrics ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['macro_precision']:.4f}")
    print(f"Recall:    {metrics['macro_recall']:.4f}")
    print(f"F1 Score:  {metrics['macro_f1']:.4f}")
    
    # We will append this to the README.md
    readme_path = os.path.join(os.path.dirname(base_dir), "README.md")
    
    if os.path.exists(readme_path):
        with open(readme_path, "a", encoding="utf-8") as f:
            f.write("\n\n## Automated Metrics: Intent Classification\n")
            f.write(f"The multi-brand Logistic Regression intent classification model achieved the following performance on the hold-out validation set ({len(val_data)} queries):\n\n")
            f.write(f"- **Accuracy:** {metrics['accuracy']:.4f}\n")
            f.write(f"- **Precision:** {metrics['macro_precision']:.4f}\n")
            f.write(f"- **Recall:** {metrics['macro_recall']:.4f}\n")
            f.write(f"- **F1 Score:** {metrics['macro_f1']:.4f}\n\n")
            f.write("## LLM-as-a-Judge Evaluation\n")
            f.write("For end-to-end response generation, we utilize an **LLM-as-a-Judge** framework (implemented via `evaluation.llm_judge.RealLLMJudge`). ")
            f.write("The judge uses a strict 5-point rubric to evaluate the final synthesized reply based on Groundedness, Policy Adherence, Tone, and Completeness.\n")
        print(f"\nAppended metrics to {readme_path}")
    else:
        print(f"\nREADME not found at {readme_path}")

if __name__ == "__main__":
    main()
