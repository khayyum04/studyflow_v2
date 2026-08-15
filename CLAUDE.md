# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Keep this file current.** When a commit changes anything documented here — a command, a flag, a
file's location, a config split, a design decision described in Architecture — update the relevant
section in the same commit (or immediately after, before moving on). This applies to you whether
you're Claude Code making the change right now or a future session picking this repo back up: don't
wait to be asked. A CLAUDE.md that quietly drifts out of sync is worse than no CLAUDE.md, since it
actively misleads instead of just being silent.

## What this is

StudyFlow is a RAG-based exam-prep tool for Malaysian SPM Sejarah (History). Two independent
pipelines live under `backend/`, both driven by one Click CLI:

- `backend/textbook_extraction/` — OCRs textbook page images into section-level markdown via
  Claude's vision + Batch API.
- `backend/rag_pipeline/` — chunks that markdown, embeds it into ChromaDB, retrieves with a
  hybrid BM25+vector search, and generates/evaluates answers with Gemini.

`frontend/` and `backend/trial_papers_extraction/` exist but are empty — not yet built.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # registers the `studyflow` console command (see below)
```

Requires a `.env` file at the repo root with `ANTHROPIC_API_KEY` (extraction) and `GEMINI_API_KEY`
(eval + generation). `cli.py` loads it via `python-dotenv` before importing anything that needs it.

## Commands

With the venv activated, use the `studyflow` console command (declared in `pyproject.toml`'s
`[project.scripts]`, installed via `pip install -e .` above):

```bash
studyflow <command>
```

Without the venv/editable-install set up, the equivalent module form always works (relative
imports require running it as a module — don't run `cli.py` directly):

```bash
python3 -m backend.rag_pipeline.cli <command>
```

Pipeline, in order:

```bash
extract [--textbook sejarah_t4|sejarah_t5] [--section sejarah_t4/bab04/sec_4_2] [--dry-run] [--force]
chunk [--textbook sejarah_t4|sejarah_t5]
embed
```

`extract` submits to the Anthropic Batch API and is resumable — a crash mid-run is picked back up
from `backend/textbook_extraction/logs/.batch_state.json` rather than resubmitting. `--dry-run`
shows what's pending without calling the API. `embed` does a full rebuild of the Chroma collection
every run (cheap, avoids stale entries after re-chunking).

Query/ask the pipeline directly:

```bash
query "<text>" [--k 5]   # prints top-k retrieved chunks with fused scores
ask "<text>" [--k 5]     # retrieval + Gemini-generated answer with a single cited source
```

Eval harness (`backend/rag_pipeline/eval_set_mcq/`):

```bash
baseline [--n 100] [--seed 0]        # no-context, Gemini + Google Search grounding
eval [--n 100] [--seed 0] [--k 5]    # full RAG: retrieval + Gemini generation
analyze                               # compares the two most recent result files
```

Use the **same `--n`/`--seed`** on `baseline` and `eval` — they need to sample the identical
question set for `analyze` to produce a meaningful comparison. `analyze` diffs whatever is
currently in `data/baseline_gemini_results.json` and `data/rag_results_gemini.json`, so an eval
run silently overwrites the previous result file — copy it aside first if you need to keep a
specific run for comparison later. (`data/baseline_results.json` and `data/rag_results.json` are
the older Claude-era results, kept for reference but no longer written or read by any command.)

There is no test suite in this repo.

## Architecture

### Two config.py files, split by concern

`backend/textbook_extraction/config.py` owns everything about *what textbooks exist and where
their assets live* — the `TEXTBOOKS` dict (image dirs, output dirs, TOC files), the extraction
model, the vision prompt, batch-size limits. `backend/rag_pipeline/config.py` owns the downstream
RAG concerns — chunk/embedding file paths, the embedding model, the answer-generation model, both
system prompts, and eval result file paths.

`TEXTBOOKS` is the join point between the two packages: extraction writes `.md` files into
`TEXTBOOKS[id]["output_dir"]`, and `rag_pipeline/chunking.py` reads them back from the same path
(imported via `from ..textbook_extraction.config import TEXTBOOKS`). If you add a textbook or
change its layout, that's the one place to edit.

### Two LLM providers, two non-overlapping roles

Claude is used *only* for OCR extraction (`MODEL` in `textbook_extraction/config.py`, vision +
Batch API). Gemini is used *only* for eval scoring and student-facing generation (`GEMINI_MODEL`
in `rag_pipeline/config.py`). This split is deliberate, not incidental: Gemini replaced Claude
Haiku as the answer model after an eval comparison showed it scored higher at comparable cost
(`git log` has the commit with the numbers). Don't assume `anthropic` and `google-genai` are
interchangeable fallbacks for each other in this codebase — each is wired to exactly one stage.

Correspondingly there are two system prompts that never mix: `EVAL_SYSTEM_PROMPT` (strict
single-letter A/B/C/D, used only inside `eval_set_mcq/`) and `GENERATION_SYSTEM_PROMPT`
(conversational Bahasa Melayu tutor persona with citations, used only by `generation.py`).

### Retrieval: hybrid search via Reciprocal Rank Fusion

`retrieval.py`'s `Retriever` fuses two independent rankings — ChromaDB vector search and an
in-memory BM25 index — by rank position (RRF), not raw score, since cosine distance and BM25
scores aren't on comparable scales. The BM25 index is rebuilt from whatever `collection.get()`
returns rather than from a separate chunks file, so it can never drift out of sync with the vector
index. `Retriever` is intentionally a stateful class (not a function) because constructing it is
expensive (loads the sentence-transformers model, builds the BM25 index) and it needs to survive
across many `retrieve()` calls in a loop — one construction per eval run, not one per question.
`generation.answer_question()` currently constructs a fresh `Retriever` per call; that's a known
gap to fix before this is wrapped in a long-lived server process (FastAPI), not yet done.

### Chunking: heading-based, three passes

`chunking.py` splits each extracted `.md` section at markdown headings, merges runs under
`MERGE_THRESHOLD` words into a neighboring chunk, then recursively splits anything over
`SPLIT_THRESHOLD` words at the paragraph break closest to the midpoint (guarding against splitting
inside a markdown table). Page numbers are tracked through all three passes via `<!-- page N -->`
markers stamped by the extraction stage, then stripped from the final chunk content.

### Eval harness

`eval_set_mcq/eval_common.py` holds what both `eval_baseline.py` and `eval_rag.py` share:
`load_eval_sample` uses **seeded random sampling**, not a positional slice — `paper_1_dataset.json`
is grouped consecutively by exam paper, so slicing would bias the sample toward whichever papers
sit first in the file. `eval_rag.py` records `retrieved_chunk_ids` and `retrieved_scores` per
question specifically so a wrong answer can be diagnosed after the fact: was retrieval wrong
(bad/missing chunks), or was generation wrong (right chunks, model still picked badly)? That
distinction has driven most of the accuracy work on this project — check retrieval before
assuming a prompt or model change is needed.

### Tracked vs gitignored data

`data/paper_1_dataset.json` and `data/chunks/textbook_chunks.json` are tracked (small, hand-curated
or cheap to regenerate). `data/textbooks/`, `data/trial_papers/`, `data/chromadb/`, and
`data/practice.db` are gitignored (large or fully regenerable from tracked inputs). Eval result
files (`data/{baseline,rag}_results.json` and their `_gemini` counterparts) **are** tracked and
kept intentionally — they're the historical record of the Claude-era vs Gemini-era accuracy
comparison, not scratch output.
