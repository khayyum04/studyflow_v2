import json
import sys
from pathlib import Path

DATASET_FILE = Path(__file__).parent / "eval_dataset.json"
VALID = {"A", "B", "C", "D"}


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 data/fix_answer.py <paper_key>")
        sys.exit(1)
    paper_key = sys.argv[1]

    records = json.loads(DATASET_FILE.read_text(encoding="utf-8"))
    targets = sorted(
        (r for r in records if r["source"]["paper_key"] == paper_key),
        key=lambda r: r["question_number"],
    )
    print(f"Found {len(targets)} questions for paper_key={paper_key}\n")

    for r in targets:
        new_answer = ""
        while new_answer not in VALID:
            new_answer = input(f"Q{r['question_number']}: ").strip().upper()
        r["answer"] = new_answer

    DATASET_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(targets)} answers.")


if __name__ == "__main__":
    main()
