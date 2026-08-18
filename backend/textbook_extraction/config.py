from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data" / "textbooks"

TEXTBOOKS = {
    "sejarah_t4": {
        "images_dir": DATA_DIR / "sejarah_t4",
        "output_dir": DATA_DIR / "sejarah_t4_md",
        "form": "Form 4",
    },
    "sejarah_t5": {
        "images_dir": DATA_DIR / "sejarah_t5",
        "output_dir": DATA_DIR / "sejarah_t5_md",
        "form": "Form 5",
    },
}

MODEL = "claude-haiku-4-5" # Claude model to use for all extraction requests

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
