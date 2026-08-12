from pathlib import Path

# Resolve the project root (studyflow_v2/) from this file's location
ROOT = Path(__file__).parent.parent.parent

# Where the chunking stage writes its output. Chunking reads its source .md
# files from backend/textbook_extraction/config.py's TEXTBOOKS[...]["output_dir"]
# — extraction and chunking are separate pipeline stages joined by that directory.
CHUNKS_FILE = ROOT / "data" / "chunks" / "textbook_chunks.json"

# Where the embedding stage persists its ChromaDB collection
CHROMA_DIR = ROOT / "data" / "chromadb"
EMBEDDING_MODEL = "ibm-granite/granite-embedding-97m-multilingual-r2"
COLLECTION_NAME = "sejarah_chunks"

# Where eval questions/results live
PAPER_1_DATASET_FILE = ROOT / "data" / "paper_1_dataset.json"
BASELINE_RESULTS_FILE = ROOT / "data" / "baseline_results.json"
RAG_RESULTS_FILE = ROOT / "data" / "rag_results.json"
GEMINI_BASELINE_RESULTS_FILE = ROOT / "data" / "baseline_gemini_results.json"
GEMINI_RAG_RESULTS_FILE = ROOT / "data" / "rag_results_gemini.json"

# Gemini model used for all answer generation: MCQ eval scoring (baseline and
# RAG) and student-facing generation.py — outperformed Claude Haiku on eval
# (gemini-2.5-flash is deprecated for new API keys — 404s at call time)
GEMINI_MODEL = "gemini-3.6-flash"

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
