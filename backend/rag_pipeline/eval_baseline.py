from __future__ import annotations

import json
import os
import re
import time

from google import genai
from google.genai import types

from .config import EVAL_SYSTEM_PROMPT, GEMINI_BASELINE_RESULTS_FILE, GEMINI_MODEL
from .eval_common import ABCD_KEYS, build_meta, format_options, load_eval_sample

ANSWER_RE = re.compile(r"\b([A-D])\b")


def _format_user_message(question: str, options: dict) -> str:
    return f"Question:\n{question}\n\nOptions:\n{format_options(options)}\n\nAnswer:"


def call_and_parse(client: genai.Client, user_message: str) -> str:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=EVAL_SYSTEM_PROMPT,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )
    raw = (response.text or "").strip().upper()
    if raw in ABCD_KEYS:
        return raw
    match = ANSWER_RE.search(raw)
    return match.group(1) if match else "IDK"


def run_baseline(n: int = 100, seed: int = 0) -> None:
    sample = load_eval_sample(n, seed)
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    results = []

    for i, record in enumerate(sample, 1):
        user_message = _format_user_message(record["question"], record["options"])
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
        })

        if i < len(sample):
            time.sleep(0.5)

    meta = build_meta(results, mode="baseline")
    GEMINI_BASELINE_RESULTS_FILE.write_text(
        json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nDone. Accuracy: {meta['accuracy'] * 100:.1f}% ({meta['correct']}/{meta['sample_size']})")
