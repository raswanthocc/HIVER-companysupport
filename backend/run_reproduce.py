"""
Turnkey Reproduction Script for CompanySupport AI Support Agent.
Guaranteed execution time < 15 minutes.
Executes automated leakage check, trains models strictly on isolated training data,
benchmarks Baseline 1, Baseline 2, and Proposed Full Agent on held-out 200-item Golden set,
and writes unadulterated empirical results to results_summary.json.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from evaluation.run_evaluation import main as run_eval_main

if __name__ == "__main__":
    run_eval_main()
