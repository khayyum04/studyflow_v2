from __future__ import annotations

import json
import random
import time
from datetime import datetime

import anthropic

from .config import ANSWER_MODEL, BASELINE_RESULTS_FILE, EVAL_DATASET_FILE, EVAL_SYSTEM_PROMPT

ROMAN_KEYS = ["I", "II", "III", "IV"]
ABCD_KEYS = ["A", "B", "C", "D"]


def format_options(options: dict) -> str:
    # Most questions are plain A-D. Some pair Roman-numeral statements (I-IV)
    # with lettered combination options (e.g. A. "I dan II") — render the
    # statements first so the model sees what each letter is actually combining.
    lines = []
    roman_present = [k for k in ROMAN_KEYS if k in options]
    if roman_present:
        lines.append("Pernyataan:")
        for k in roman_present:
            lines.append(f"{k}. {options[k]}")
        lines.append("")

    for k in ABCD_KEYS:
        if k in options:
            lines.append(f"{k}. {options[k]}")

    return "\n".join(lines)


def call_and_parse(client: anthropic.Anthropic, user_message: str) -> str:
    response = client.messages.create(
        model=ANSWER_MODEL,
        max_tokens=16,
        system=EVAL_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    raw = response.content[0].text.strip().upper()
    return raw if raw in ABCD_KEYS else "IDK"


def load_eval_sample(n: int, seed: int) -> list[dict]:
    records = json.loads(EVAL_DATASET_FILE.read_text(encoding="utf-8"))
    return random.Random(seed).sample(records, min(n, len(records)))


def _format_user_message(question: str, options: dict) -> str:
    return f"Question:\n{question}\n\nOptions:\n{format_options(options)}\n\nAnswer:"


def build_meta(results: list[dict], mode: str) -> dict:
    correct_count = sum(1 for r in results if r["correct"])
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": ANSWER_MODEL,
        "mode": mode,
        "sample_size": len(results),
        "accuracy": round(correct_count / len(results), 4) if results else 0.0,
        "correct": correct_count,
        "incorrect": len(results) - correct_count,
    }


def run_baseline(n: int = 100, seed: int = 0) -> None:
    sample = load_eval_sample(n, seed)
    client = anthropic.Anthropic()
    results = []

    for i, record in enumerate(sample, 1):
        user_message = _format_user_message(record["question"], record["options"])
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
        })

        if i < len(sample):
            time.sleep(0.5)

    meta = build_meta(results, mode="baseline")
    BASELINE_RESULTS_FILE.write_text(
        json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nDone. Accuracy: {meta['accuracy'] * 100:.1f}% ({meta['correct']}/{meta['sample_size']})")
