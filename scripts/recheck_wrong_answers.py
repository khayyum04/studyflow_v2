"""
Re-run every wrongly-answered question from data/rag_results.json through
Claude with the web_search tool enabled, to see whether grounding fixes them.

Setup:
    pip install anthropic python-dotenv
    (ANTHROPIC_API_KEY is loaded from .env at the repo root)
"""

from __future__ import annotations

import json
import os
import re

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

MODEL = "claude-haiku-4-5"
RAG_RESULTS_PATH = "data/rag_results.json"
OUTPUT_PATH = "data/recheck_wrong_answers_web_search.json"

ANSWER_RE = re.compile(r"JAWAPAN:\s*([A-Z])", re.IGNORECASE)


def build_prompt(question: str, options: dict) -> str:
    options_text = "\n".join(f"{key}: {value}" for key, value in options.items())
    return (
        "Berikut adalah soalan aneka pilihan Sejarah SPM Malaysia.\n\n"
        f"Soalan: {question}\n\n"
        f"Pilihan:\n{options_text}\n\n"
        "Selepas itu, berikan jawapan akhir dalam format tepat ini pada baris baharu:\n"
        "JAWAPAN: <huruf>"
    )


def ask_grounded(question: str, options: dict) -> tuple[str | None, str]:
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": build_prompt(question, options)}],
        tools=[
            {
                "type": "web_search_20260209",
                "name": "web_search",
                "allowed_callers": ["direct"],
                "max_uses": 3,
            }
        ],
    )

    text = "\n".join(block.text for block in response.content if block.type == "text")
    match = ANSWER_RE.search(text)
    predicted = match.group(1).upper() if match else None
    return predicted, text


def main():
    with open(RAG_RESULTS_PATH) as f:
        data = json.load(f)

    wrong = [r for r in data["results"] if not r["correct"]]
    print(f"Re-checking {len(wrong)} wrongly-answered questions with web search enabled...\n")

    results = []
    fixed = 0
    for i, item in enumerate(wrong, 1):
        print(f"[{i}/{len(wrong)}] {item['id']}")
        try:
            new_predicted, raw_text = ask_grounded(item["question"], item["options"])
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({**item, "web_search_predicted": None, "web_search_correct": False, "error": str(e)})
            continue

        now_correct = new_predicted == item["correct_answer"]
        fixed += now_correct
        status = "FIXED" if now_correct else "still wrong"
        print(
            f"  correct={item['correct_answer']}  old_predicted={item['predicted_answer']}  "
            f"new_predicted={new_predicted}  -> {status}"
        )

        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "correct_answer": item["correct_answer"],
                "old_predicted_answer": item["predicted_answer"],
                "web_search_predicted": new_predicted,
                "web_search_correct": now_correct,
                "raw_response": raw_text,
            }
        )

    print("\n" + "=" * 60)
    print(f"Fixed by web search: {fixed}/{len(wrong)}")
    print("=" * 60)

    with open(OUTPUT_PATH, "w") as f:
        json.dump({"fixed": fixed, "total": len(wrong), "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
