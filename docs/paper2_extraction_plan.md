# Paper 2 (Kertas 2) Extraction Pipeline — Plan

Status: design only, nothing implemented yet.

## Goal

Build `data/paper_2_dataset.json`, an eval/grading dataset for SPM Sejarah Kertas 2
trial papers, analogous to the existing `data/paper_1_dataset.json` but structurally
different: Paper 2 has nested sub-parts, mark-scheme rubrics instead of single-letter
answers, and an optional-choice section (Bahagian B). The dataset is for grading only —
the student-facing "blank paper" view will render the original PDF directly, not
reconstruct the paper from this JSON. The two are joined only by paper_key / question
number / sub-part label.

## Record schema

One record per question (Bahagian A: questions 1–4, compulsory; Bahagian B: remaining
questions, students choose 3 of N to have graded — see grading note below). No
`is_choice_question` field: it's fully derivable from `bahagian`. No source-citation
field on stimuli: irrelevant for grading.

```json
{
  "id": "2025_johor_jb_k2_q1",
  "source": {
    "year": 2025,
    "state": "Johor",
    "district": "Johor Bahru",
    "paper": "K2",
    "paper_key": "2025_johor_jb"
  },
  "bahagian": "A",
  "question_number": 1,
  "question_stem": "Rajah berikut berkaitan dengan sistem pentadbiran Kesultanan Melayu Melaka.",
  "stimulus": {
    "type": "diagram",
    "description": "Carta organisasi Pembesar Empat Lipatan: Sultan di puncak, diikuti X, kemudian tiga jawatan sejajar - Penghulu Bendahari, Y, dan Laksamana.",
    "data": { "Sultan": "Sultan", "X": "?", "row_2": ["Penghulu Bendahari", "Y (?)", "Laksamana"] }
  },
  "sub_parts": [
    {
      "label": "a",
      "question_text": "Nyatakan jawatan pembesar berikut: X, Y",
      "marks": 2,
      "mark_scheme": [
        { "point_id": "F1", "point_text": "X: Bendahara" },
        { "point_id": "F2", "point_text": "Y: Temenggung" }
      ],
      "scoring_notes": null,
      "model_answers": null
    },
    {
      "label": "b",
      "question_text": "Jelaskan peranan Pembesar Empat Lipatan.",
      "marks": 4,
      "mark_scheme": [
        { "point_id": "F1", "group": "Bendahara", "point_text": "Ketua pentadbir" },
        { "point_id": "F1", "group": "Penghulu Bendahari", "point_text": "Menguruskan perbendaharaan secara teratur" }
      ],
      "scoring_notes": "Fakta boleh diambil daripada mana-mana pembesar; markah dikira sehingga had 4 markah.",
      "model_answers": null
    }
  ]
}
```

Field notes:
- `stimulus` is nullable — omitted when a question has no diagram/photo/table/excerpt.
- `mark_scheme[].group` is set when the skema organizes points under a subheading
  (e.g. separate fact lists per office-holder); omitted for flat lists.
- `scoring_notes` carries skema caveats verbatim ("jawapan umum tidak diterima",
  "KBAT terkawal", "mana-mana jawapan munasabah") — null when the skema states none.
- `model_answers` is an array (not a single string) because some skema entries give
  multiple worked examples ("Contoh 1", "Contoh 2") — null when the skema is a pure
  fact list with no worked example.

Bahagian B grading note (not a schema field — belongs in the future grading engine):
students may attempt more than 3 Bahagian B questions; nothing is deducted for extra
attempts. The examiner grades every attempted question and keeps only the top 3 by
marks awarded toward the total. The dataset does not need to encode this — it's a
paper-level scoring policy applied after all attempted sub-parts are graded.

## Pipeline architecture

New module: `backend/trial_papers_extraction/` (currently an empty stub per
`CLAUDE.md` — this is its first real use).

### 1. Discovery & pairing (manifest step)

`data/trial_papers/` has ~1000 PDFs with inconsistent naming (numeric prefixes,
"SOALAN"/"SKEMA"/"Jawapan"/"RALAT"/"Set_N" tokens, no fixed state/district folder
depth). Rather than pairing soalan↔skema fully automatically and trusting it, this
step produces a reviewable manifest:

- Walk `data/trial_papers/<year>/Sejarah/<folder>/`, filter to K2 files only.
- Normalize filenames (strip numeric prefix, role tokens, "RALAT", "Set_N") to find
  soalan/skema pairs within the same folder + set label.
- Prefer "RALAT" (errata) versions over the original when both exist for the same role.
- **Papers whose only answer file is marked "Tiada Jawapan" (no scheme) are recorded
  as unpairable and skipped entirely — never sent to extraction.**
- Output: `data/trial_papers_manifest.json`, one entry per resolved (or flagged)
  paper: `{ paper_key, year, state, district, soalan_path, skema_path, status }`
  where `status` is one of `paired`, `unpairable_no_scheme`, `ambiguous` (needs a
  human to pick between candidates). Reviewed once by hand before the first real run.

### 2. Extraction request

- One Batch API request per paper (not per page) — both PDFs sent as native
  `document` content blocks (Claude reads PDFs directly; no PNG conversion needed,
  unlike `textbook_extraction`, since these are short, self-contained documents).
- Forced structured output via `tool_choice` against a JSON schema mirroring the
  record shape above (array of questions, each with nested sub_parts) — not
  free-text JSON parsing.
- Prompt instructs: match sub-parts between soalan and skema by label; describe
  stimuli in enough detail to be self-contained (structured `data` for
  diagrams/org-charts with blanks, not just prose); extract every mark_scheme point
  verbatim and in order; set `group` when the skema subheads its point list; capture
  scoring caveats verbatim into `scoring_notes`; capture every worked example into
  `model_answers`; preserve Bahasa Melayu exactly, fixing only unambiguous OCR
  artifacts.
- Pilot on a handful of known-good papers (e.g. the Johor JB 2025 paper already
  reviewed by hand) and diff against manual extraction before running at scale.

#### Forced-schema tool definition (drafted)

```json
{
  "name": "record_paper2_questions",
  "description": "Structured extraction of one SPM Sejarah Kertas 2 trial paper, joining the question paper with its marking scheme.",
  "input_schema": {
    "type": "object",
    "properties": {
      "questions": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "bahagian": { "type": "string", "enum": ["A", "B"] },
            "question_number": { "type": "integer" },
            "question_stem": { "type": "string" },
            "stimulus": {
              "type": ["object", "null"],
              "properties": {
                "type": { "type": "string", "enum": ["diagram", "image", "table", "text_excerpt", "song_lyrics", "poster", "stamp"] },
                "description": { "type": "string" },
                "data": { "type": ["object", "null"] }
              },
              "required": ["type", "description"]
            },
            "sub_parts": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "label": { "type": "string" },
                  "question_text": { "type": "string" },
                  "marks": { "type": "integer" },
                  "mark_scheme": {
                    "type": "array",
                    "items": {
                      "type": "object",
                      "properties": {
                        "point_id": { "type": "string" },
                        "group": { "type": ["string", "null"] },
                        "point_text": { "type": "string" }
                      },
                      "required": ["point_id", "point_text"]
                    }
                  },
                  "scoring_notes": { "type": ["string", "null"] },
                  "model_answers": { "type": ["array", "null"], "items": { "type": "string" } }
                },
                "required": ["label", "question_text", "marks", "mark_scheme"]
              }
            }
          },
          "required": ["bahagian", "question_number", "question_stem", "sub_parts"]
        }
      }
    },
    "required": ["questions"]
  }
}
```

`source`/`id` fields are deliberately left off the model's output and stitched on
afterward from the manifest — no reason to make the model guess metadata (year,
state, district, paper_key) that's already known from the file path.

Called via forced tool use (`tool_choice: {"type": "tool", "name": "record_paper2_questions"}`),
not free-text JSON parsing.

#### Prompt text (drafted)

> You are extracting a Malaysian SPM Sejarah (History) Kertas 2 trial exam paper into
> structured JSON, joining the question paper (first document) with its marking
> scheme (second document, "Skema Pemarkahan"). Call `record_paper2_questions` with
> one entry per question (Bahagian A: questions 1–4, compulsory; Bahagian B: the
> remaining questions, students choose which to answer — do not encode that choice,
> just extract every question present).
>
> For each question: capture `question_stem` as the introductory text before the
> lettered sub-parts (or an empty string if the question goes straight into
> sub-parts). If a stimulus (diagram, photo, table, excerpt, poster, song, stamp) is
> present, describe it in enough detail that the question is answerable without
> seeing the image — for diagrams with labeled/blank nodes (e.g. an org chart with
> "X" or "Y" placeholders), represent the full structure in `data` as a nested
> object, not just prose. Do not invent details you can't actually see; if something
> is illegible, say so in the description rather than guessing.
>
> For each sub_part, match it to the skema by its label (a/b/c, and nested i/ii
> where present) — the skema will show the same label with a "PEMARKAHAN" column.
> Extract `marks` from the bracketed value (e.g. "[4 markah]"). Extract every
> scoring point listed (F1, F2, F3...) into `mark_scheme`, preserving order and
> exact wording — do not summarize or merge points. If the skema groups points
> under a subheading (e.g. separate fact lists per office-holder, per era, per
> cause category), set `group` to that subheading's exact text on every point under
> it; omit `group` entirely when the skema is a flat list.
>
> Capture any scoring caveats verbatim in `scoring_notes` — phrases like "jawapan
> umum tidak diterima," "KBAT terkawal," "mana-mana jawapan munasabah," "jawapan
> pasang siap tidak diterima" all belong here. Set `scoring_notes` to null when the
> skema states none.
>
> If the skema includes one or more worked example answers (usually labeled "Contoh
> jawapan," "Contoh 1," "Contoh 2"), include every one of them verbatim in
> `model_answers`. Set `model_answers` to null when the skema is a pure fact list
> with no worked example.
>
> Preserve all Bahasa Melayu text exactly as written, including minor stylistic
> inconsistencies — do not paraphrase or correct grammar. Fix only unambiguous
> OCR-type artifacts (e.g. a clearly broken character) where the surrounding
> context makes the intended word certain.

### 3. Validation pass

Runs after extraction, before writing to the dataset file:
- Pydantic schema validation (types, required fields).
- Per sub-part: mark_scheme point count doesn't imply more than `marks` (accounting
  for grouped points where any combination up to the cap is valid).
- Bahagian A: exactly 4 questions, marks summing to 40. Bahagian B: marks per
  question summing to 20.
- Sub-part labels align between soalan and skema (catches a bad pairing or an
  incomplete skema).
- Failures are flagged for manual review, not silently written or dropped.

### 4. Resumability

Same pattern as `textbook_extraction/batch.py`: a state file (e.g.
`backend/trial_papers_extraction/logs/.batch_state.json`) tracks submitted batch IDs
and paper_keys so a crash mid-run resumes instead of resubmitting.

## CLI

```
python3 -m backend.rag_pipeline.cli extract-paper2 \
  [--year 2025] [--state Johor] [--district "Johor Bahru"] \
  [--force] [--append] [--dry-run]
```

- `--district` requires `--state`; `--state` requires `--year` (same validation
  spirit as `parse_section_arg`). No scope options = full corpus.
- Scope filters against the normalized `state`/`district` fields in the manifest,
  not literal folder depth (the real folder layout doesn't cleanly separate the two).
- **Papers with `status: unpairable_no_scheme` in the manifest are excluded from the
  resolved scope entirely** — they never appear as pending, skipped, or extracted;
  `--dry-run` reports them as a separate "unpairable (no scheme) — not processed"
  bucket so they're visible but clearly out of scope.
- Default (no flags): idempotent and additive. Skip paper_keys already present in
  `paper_2_dataset.json`, extract the rest of the resolved scope, merge new records
  in — existing entries (in or out of scope) are untouched. Deliberately different
  from `chunk`/`embed`'s full-rebuild-every-run pattern, since Batch API extraction
  is expensive and will run in scoped chunks across many sessions.
- `--force`: within scope, re-extract papers even if already present, replacing
  their existing entry.
- `--append`: only meaningful with `--force` — instead of replacing the old entry,
  keep both (new one gets a suffixed id, e.g. `_v2`) so two extraction attempts can
  be diffed before manually choosing which to keep. A no-op (with a warning) if used
  without `--force`.
- `--dry-run`: reports four buckets without calling the API — pending (will be
  extracted), already-done (will be skipped), will-be-replaced (in scope + already
  done + `--force` set), and unpairable/no-scheme (excluded, not processed).

## Confirmed decisions

- Bahagian B's top-3-of-N grading policy is explicitly out of scope for this
  extraction pipeline. It belongs entirely to the future grading engine, which will
  grade every attempted Bahagian B sub-part and keep only the top 3 by marks
  awarded — the dataset schema doesn't need to know about it.

## Open items / not yet decided

- **Ambiguous pairing resolution mechanism.** When discovery can't confidently
  resolve a soalan↔skema pair (tied candidates after normalization, or a combined
  answer file like `..._K1_K2 Jawapan.pdf` covering both papers at once), it's
  flagged `status: "ambiguous"` in the manifest with its candidate paths, and
  excluded from extraction until a human resolves it. Two ways to feed that
  resolution back in:
  - *Separate override file* (e.g. `data/trial_papers_manifest_overrides.json`) —
    small, hand-curated, maps paper_key → chosen skema_path (or an explicit
    "exclude" verdict). Discovery reads it on every run and applies it on top of
    auto-detection, so regenerating the manifest never clobbers a manual decision.
    Matches the tracked-vs-gitignored split CLAUDE.md already uses elsewhere in
    this repo. Recommended.
  - *CLI command that mutates the manifest directly*
    (`extract-paper2 resolve-pairing --paper-key ... --skema-path ...`) — one file
    instead of two, but the manifest stops being purely regeneratable; re-running
    discovery from scratch would need to avoid overwriting manual fixes, a subtler
    bug surface than the override-file approach.
  - Decision pending.
