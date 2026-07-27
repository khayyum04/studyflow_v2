from __future__ import annotations

import json
import time

import anthropic

from .config import RAG_RESULTS_FILE
from .eval_baseline import build_meta, call_and_parse, format_options, load_eval_sample
from .retrieval import Retriever


def _format_user_message(chunks: list[dict], question: str, options: dict) -> str:
    context = "\n\n".join(c["document"] for c in chunks) if chunks else "(No context retrieved)"
    return f"Context:\n{context}\n\nQuestion:\n{question}\n\nOptions:\n{format_options(options)}\n\nAnswer:"


def run_eval(n: int = 100, seed: int = 0, k: int = 5) -> None:
    sample = load_eval_sample(n, seed)
    client = anthropic.Anthropic()
    retriever = Retriever()  # loaded once, reused for every question in the loop
    results = []

    for i, record in enumerate(sample, 1):
        chunks = retriever.retrieve(record["question"], k=k)
        user_message = _format_user_message(chunks, record["question"], record["options"])
        predicted = call_and_parse(client, user_message)
        correct = predicted == record["answer"]

        print(f"[{i}/{len(sample)}] {'✓' if correct else '✗'} {record['id']}")
        results.append({
            "id": record["id"],
            "question": record["question"],
            "options": record["options"],
            "correct_answer": record["answer"],
            "predicted_answer": predicted,
            "correct": correct,
            "retrieved_chunk_ids": [c["chunk_id"] for c in chunks],
        })

        if i < len(sample):
            time.sleep(0.5)

    meta = build_meta(results, mode="rag")
    meta["k"] = k
    RAG_RESULTS_FILE.write_text(
        json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nDone. Accuracy: {meta['accuracy'] * 100:.1f}% ({meta['correct']}/{meta['sample_size']})")
