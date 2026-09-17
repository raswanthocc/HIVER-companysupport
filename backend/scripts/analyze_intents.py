import os
import json
import csv
from collections import defaultdict, Counter

def analyze_intents():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "train_data.json")
    
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        return

    with open(train_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    brand_intents = defaultdict(list)
    for record in data:
        brand = record.get("brand", "Unknown")
        intent = record.get("intent", "other_unclear")
        brand_intents[brand].append(intent)

    analysis = {}
    for brand, intents in brand_intents.items():
        counts = Counter(intents)
        total = sum(counts.values())
        analysis[brand] = {
            "total_queries": total,
            "intents": dict(counts.most_common())
        }

    output_path = os.path.join(base_dir, "data", "brand_intents_analysis.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    print(f"Analysis saved to {output_path}")
    print(f"Found data for {len(analysis)} brands.")

if __name__ == "__main__":
    analyze_intents()
