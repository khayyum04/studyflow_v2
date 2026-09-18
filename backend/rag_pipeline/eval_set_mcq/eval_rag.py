from __future__ import annotations

import json
import os
import re
import time

from google import genai
from google.genai import types

from ..config import EVAL_SYSTEM_PROMPT, GEMINI_MODEL, GEMINI_RAG_RESULTS_FILE
from ..retrieval import Retriever
from .eval_common import ABCD_KEYS, build_meta, format_options, load_eval_sample

# Full RAG eval: retrieve textbook chunks for each question, then answer using
# ONLY that context (no Google Search — compare against eval_baseline.py to see
# how much retrieval alone is worth over the model's own knowledge).

ANSWER_RE = re.compile(r"\b([A-D])\b")


def _format_user_message(chunks: list[dict], question: str, options: dict) -> str:
    context = "\n\n".join(c["document"] for c in chunks) if chunks else "(No context retrieved)"
    return f"Context:\n{context}\n\nQuestion:\n{question}\n\nOptions:\n{format_options(options)}\n\nAnswer:"


def call_and_parse(client: genai.Client, user_message: str) -> str:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(system_instruction=EVAL_SYSTEM_PROMPT),
    )
    # Same letter-extraction fallback as eval_baseline.py — model rarely breaks
    # the "single letter" instruction, but this catches it instead of crashing.
    raw = (response.text or "").strip().upper()
    if raw in ABCD_KEYS:
        return raw
    match = ANSWER_RE.search(raw)
    return match.group(1) if match else "IDK"


def run_eval(n: int = 100, seed: int = 0, k: int = 5) -> None:
    sample = load_eval_sample(n, seed)
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    retriever = Retriever()  # loads the embedding model + BM25 index once, reused for every question below
    results = []

    for i, record in enumerate(sample, 1):
        chunks = retriever.retrieve(record["question"], k=k)
        user_message = _format_user_message(chunks, record["question"], record["options"])
        try:
            predicted = call_and_parse(client, user_message)
        except Exception as e:
            print(f"[{i}/{len(sample)}] ERROR {record['id']}: {e}")
            predicted = "IDK"
        correct = predicted == record["answer"]

        print(f"[{i}/{len(sample)}] {'✓' if correct else '✗'} {record['id']}")
        results.append({
            "id": record["id"],
            "question": record["question"],
            "options": record["options"],
            "correct_answer": record["answer"],
            "predicted_answer": predicted,
            "correct": correct,
            # chunk_ids + scores are recorded per-question so a wrong answer can
            # be diagnosed after the fact: was retrieval wrong (bad chunks) or
            # was generation wrong (right chunks, model still picked badly)?
            "retrieved_chunk_ids": [c["chunk_id"] for c in chunks],
            "retrieved_scores": [c["score"] for c in chunks],
        })

        if i < len(sample):
            time.sleep(0.5)

    meta = build_meta(results, mode="rag")
    meta["k"] = k
    GEMINI_RAG_RESULTS_FILE.write_text(
        json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nDone. Accuracy: {meta['accuracy'] * 100:.1f}% ({meta['correct']}/{meta['sample_size']})")
