from __future__ import annotations

import json

from .config import GEMINI_BASELINE_RESULTS_FILE, GEMINI_RAG_RESULTS_FILE


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def run_analyze() -> None:
    baseline_data = json.loads(GEMINI_BASELINE_RESULTS_FILE.read_text(encoding="utf-8"))
    rag_data = json.loads(GEMINI_RAG_RESULTS_FILE.read_text(encoding="utf-8"))

    baseline_results = {r["id"]: r for r in baseline_data["results"]}
    rag_results = {r["id"]: r for r in rag_data["results"]}
    common_ids = set(baseline_results) & set(rag_results)

    both_correct = rag_only = baseline_only = both_wrong = 0
    for qid in common_ids:
        r, b = rag_results[qid]["correct"], baseline_results[qid]["correct"]
        if r and b:
            both_correct += 1
        elif r and not b:
            rag_only += 1
        elif b and not r:
            baseline_only += 1
        else:
            both_wrong += 1

    n = len(common_ids)
    rag_acc = (both_correct + rag_only) / n
    baseline_acc = (both_correct + baseline_only) / n

    print("===== Accuracy Comparison =====")
    print(f"Questions compared : {n}")
    print(f"Baseline accuracy  : {_pct(baseline_acc)}")
    print(f"RAG accuracy       : {_pct(rag_acc)}")
    print(f"Delta              : {_pct(rag_acc - baseline_acc)}")
    print()
    print("Breakdown:")
    print(f"  Both correct                : {both_correct:>4}  ({both_correct/n*100:.1f}%)")
    print(f"  RAG correct, baseline wrong : {rag_only:>4}  ({rag_only/n*100:.1f}%)  <- RAG adds value")
    print(f"  Baseline correct, RAG wrong : {baseline_only:>4}  ({baseline_only/n*100:.1f}%)  <- RAG regresses")
    print(f"  Both wrong                  : {both_wrong:>4}  ({both_wrong/n*100:.1f}%)")
