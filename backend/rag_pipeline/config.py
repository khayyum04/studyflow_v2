from pathlib import Path

# Resolve the project root (studyflow_v2/) from this file's location
ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data" / "textbooks"
_TOCS_DIR = Path(__file__).parent / "textbook_extraction" / "sejarah_toc"

# Where the chunking stage writes its output
CHUNKS_FILE = ROOT / "data" / "chunks" / "textbook_chunks.json"

# Where the embedding stage persists its ChromaDB collection
CHROMA_DIR = ROOT / "data" / "chromadb"
EMBEDDING_MODEL = "ibm-granite/granite-embedding-97m-multilingual-r2"
COLLECTION_NAME = "sejarah_chunks"

# One entry per textbook: where to find the PNG pages, where to write the .md files, and the TOC
TEXTBOOKS = {
    "sejarah_t4": {
        "images_dir": DATA_DIR / "sejarah_t4",
        "output_dir": DATA_DIR / "sejarah_t4_md",
        "toc_file": _TOCS_DIR / "sejarah_t4_toc.json",
        "form": "Form 4",
    },
    "sejarah_t5": {
        "images_dir": DATA_DIR / "sejarah_t5",
        "output_dir": DATA_DIR / "sejarah_t5_md",
        "toc_file": _TOCS_DIR / "sejarah_t5_toc.json",
        "form": "Form 5",
    },
}

# Claude model to use for all extraction requests
MODEL = "claude-haiku-4-5"

# Claude model used for answering user questions (generation) and eval scoring —
# kept separate from MODEL (extraction) even though the value currently matches
ANSWER_MODEL = "claude-haiku-4-5"

# Where eval questions/results live
EVAL_DATASET_FILE = ROOT / "data" / "eval_dataset.json"
BASELINE_RESULTS_FILE = ROOT / "data" / "baseline_results.json"
RAG_RESULTS_FILE = ROOT / "data" / "rag_results.json"
GEMINI_BASELINE_RESULTS_FILE = ROOT / "data" / "baseline_gemini_results.json"

# Gemini model used for the Google-Search-grounded baseline eval
# (gemini-2.5-flash is deprecated for new API keys — 404s at call time; use the flash-lite tier)
GEMINI_MODEL = "gemini-3.5-flash-lite"

# System prompt for answering a student's free-form question with retrieved context
GENERATION_SYSTEM_PROMPT = """You are an SPM (Sijil Pelajaran Malaysia) Sejarah tutor.

Rules:
- Answer based only on the provided textbook context.
- If the context is insufficient, say you do not have enough information.
- Use clear, casual Bahasa Melayu unless the student asks in English — not too formal, easy to understand.
- Keep the answer exam-focused and easy to understand.
- Cite the chapter, section, and page number(s) at the end of your answer.
- Some retrieved text may contain minor OCR or extraction mistakes; correct obvious \
spelling/formatting errors only when the surrounding context makes the intended meaning \
clear. If the meaning is uncertain, say the source is unclear instead of guessing."""

# System prompt for strict single-letter MCQ scoring (used identically by the
# baseline and RAG eval — the only difference between the two is what's in the
# user turn, not the system prompt)
EVAL_SYSTEM_PROMPT = (
    "You are an expert on Malaysian history (Sejarah) at the SPM level.\n"
    "Answer the following multiple choice question.\n"
    "You MUST respond with only a single letter: A, B, C, or D.\n"
    "Always pick the best available option even if you are uncertain. Never refuse to answer.\n"
    "Do not explain your answer. Do not include any other text."
)

# How often (in seconds) to ask the Batch API whether our job is done
POLL_INTERVAL = 30

# Tracks the active batch(es) so a crashed run can be resumed without resubmitting
BATCH_STATE_FILE = Path(__file__).parent / "logs" / ".batch_state.json"

# The Batch API rejects a submission over 256MB total; stay well under that so
# per-request JSON overhead and estimation error can't push us over the edge
MAX_BATCH_BYTES = 180 * 1024 * 1024

# The vision prompt sent to Claude alongside each section's pages
MARKDOWN_PROMPT = """You are reading pages from a Malaysian secondary school Sejarah (History) textbook. These pages are given by their sections (divided into sub-topics in each chapter)
Convert the full content of these pages to well-structured Markdown.

Include:
- All headings, subheadings, and their hierarchy (use the best of your ability to determine this from font size, bolding, etc.)
- All body text, dates, names, and facts exactly as written
- Tables as Markdown tables
- Illustrations or diagrams described briefly in square brackets, e.g. [Image: map of Malaya]
- Bahasa Melayu terms exactly as written (avoid any typos)
- Every page marker must be immediately followed by the actual content that appears 
  on that page, even if it's a continuation of a sentence or paragraph from the 
  previous page. Never place a page marker without content following it. The first page marker should appear at the very start of the output before any content.
  
  Example:
  ...end of sentence on page 153.
  
  <!-- page 154 -->
  
  Continuation of content that started on page 153, now appearing on page 154...
  
  Do not use placeholder text like "[Continuing from previous page]" — output the 
  actual textbook content.

Exclude (do not mention at all):
- QR codes or any links to external resources
- Publisher or ministry logos (e.g. KPM logo)
- Source citations / bibliographic references (e.g. "Sumber: ...")
- Page numbers, headers, and footers
- structured activity blocks with a header label "AKTIVITI", containing tasks for students to complete
- identified by a circular logo/emblem icon (no text header). These contain reflection or critical thinking questions directed at the student.

If something is unreadable, write [UNREADABLE].
Output only Markdown."""
