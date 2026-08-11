from __future__ import annotations

import json
import random
from datetime import datetime

from ..config import EVAL_DATASET_FILE, GEMINI_MODEL

ROMAN_KEYS = ["I", "II", "III", "IV"]
ABCD_KEYS = ["A", "B", "C", "D"]


def format_options(options: dict) -> str:
    # Most SPM MCQs are plain A-D. Some pair Roman-numeral statements (I-IV)
    # with lettered combination answers (e.g. A. "I dan II") — we render the
    # statements first so the model sees what each letter is actually combining,
    # instead of just seeing "A. I dan II" with no idea what I/II say.
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


def load_eval_sample(n: int, seed: int) -> list[dict]:
    # random.Random(seed).sample (not a plain slice) — eval_dataset.json is
    # grouped consecutively by exam paper, so slicing the first n would bias
    # the sample toward whichever papers happen to sit at the top of the file.
    # Same (n, seed) always returns the same questions, so baseline/RAG/gemini
    # runs stay comparable against each other.
    records = json.loads(EVAL_DATASET_FILE.read_text(encoding="utf-8"))
    return random.Random(seed).sample(records, min(n, len(records)))


def build_meta(results: list[dict], mode: str) -> dict:
    # "mode" distinguishes which eval produced this file (baseline / rag) —
    # eval_analyze.py doesn't use it directly, but it's what you check first
    # when a results JSON shows up and you're not sure which script wrote it.
    correct_count = sum(1 for r in results if r["correct"])
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": GEMINI_MODEL,
        "mode": mode,
        "sample_size": len(results),
        "accuracy": round(correct_count / len(results), 4) if results else 0.0,
        "correct": correct_count,
        "incorrect": len(results) - correct_count,
    }
