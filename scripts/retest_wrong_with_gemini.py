"""
Re-run the 21 questions RAG got wrong through Gemini instead of Claude,
using the EXACT same retrieved context (no re-retrieval, no web search) —
isolates whether a different generation model resolves distractor confusion
that Haiku couldn't, given retrieval already found the right chunks.

Setup:
    (GEMINI_API_KEY is loaded from .env at the repo root)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai

load_dotenv()

ROOT = Path(__file__).parent.parent
RAG_RESULTS_PATH = ROOT / "data" / "rag_results.json"
CHUNKS_PATH = ROOT / "data" / "chunks" / "textbook_chunks.json"

GEMINI_MODEL = "gemini-3.5-flash-lite"
ABCD_KEYS = ["A", "B", "C", "D"]
ROMAN_KEYS = ["I", "II", "III", "IV"]
ANSWER_RE = re.compile(r"\b([A-D])\b")

# Same system prompt as backend/rag_pipeline/config.py's EVAL_SYSTEM_PROMPT
EVAL_SYSTEM_PROMPT = (
    "You are an expert on Malaysian history (Sejarah) at the SPM level.\n"
    "Answer the following multiple choice question.\n"
    "You MUST respond with only a single letter: A, B, C, or D.\n"
    "Always pick the best available option even if you are uncertain. Never refuse to answer.\n"
    "Do not explain your answer. Do not include any other text."
)


def format_options(options: dict) -> str:
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


def _format_user_message(chunks: list[str], question: str, options: dict) -> str:
    context = "\n\n".join(chunks) if chunks else "(No context retrieved)"
    return f"Context:\n{context}\n\nQuestion:\n{question}\n\nOptions:\n{format_options(options)}\n\nAnswer:"


def call_and_parse(client: genai.Client, user_message: str) -> str:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_message,
        config={"system_instruction": EVAL_SYSTEM_PROMPT},
    )
    raw = (response.text or "").strip().upper()
    if raw in ABCD_KEYS:
        return raw
    match = ANSWER_RE.search(raw)
    return match.group(1) if match else "IDK"


def main():
    data = json.loads(RAG_RESULTS_PATH.read_text(encoding="utf-8"))
    wrong = [r for r in data["results"] if not r["correct"]]
    chunk_lookup = {c["chunk_id"]: c["content_md"] for c in json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))}

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    fixed = 0

    for i, r in enumerate(wrong, 1):
        chunks = [chunk_lookup[cid] for cid in r["retrieved_chunk_ids"] if cid in chunk_lookup]
        user_message = _format_user_message(chunks, r["question"], r["options"])
        new_predicted = call_and_parse(client, user_message)
        now_correct = new_predicted == r["correct_answer"]
        fixed += now_correct

        status = "FIXED" if now_correct else "still wrong"
        print(
            f"[{i}/{len(wrong)}] {r['id']}: correct={r['correct_answer']}  "
            f"haiku_predicted={r['predicted_answer']}  gemini_predicted={new_predicted}  -> {status}"
        )

    print(f"\n{fixed}/{len(wrong)} previously-wrong questions fixed by switching generation model to {GEMINI_MODEL}")


if __name__ == "__main__":
    main()
