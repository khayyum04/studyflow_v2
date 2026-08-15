from __future__ import annotations

import os
import re
from dataclasses import dataclass

from google import genai
from google.genai import types

from .config import GEMINI_MODEL, GENERATION_SYSTEM_PROMPT
from .retrieval import Retriever

NO_CONTEXT_ANSWER = "Maaf, saya tidak mempunyai maklumat yang mencukupi untuk menjawab soalan ini."

# The model doesn't reliably write a bare number despite the prompt asking for one —
# seen "RUJUKAN: 1", "RUJUKAN: Source 1", "RUJUKAN: [Source 1]". Rather than chase every
# phrasing, just take the first number after "RUJUKAN:" on that line, whatever wraps it.
CITATION_RE = re.compile(r"RUJUKAN:.*?(\d+)\]?", re.IGNORECASE)

GENERATION_SYSTEM_PROMPT += (
    "\n- End your answer with a new line: 'RUJUKAN: <N>', where N is the "
    "[Source N] label of the single context block your answer is most based on. "
    "Pick exactly one, even if you drew on more than one."
)


@dataclass(frozen=True)
class Source:
    chapter_title: str
    section_title: str
    page: int          # current_page of the one chunk the model cited


@dataclass(frozen=True)
class Answer:
    text: str
    source: Source | None
    query: str


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk["metadata"]
        parts.append(
            f"[Source {i}] {meta['chapter_title']} > {meta['section_title']} "
            f"(m.s. {meta['page_start']}-{meta['page_end']})\n{chunk['document']}"
        )
    return "\n\n".join(parts)


def generate_answer(query: str, chunks: list[dict]) -> Answer:
    if not chunks:
        return Answer(text=NO_CONTEXT_ANSWER, source=None, query=query)

    context = _format_context(chunks)  # unchanged, still shows page_start-page_end per [Source N]
    user_message = f"Konteks:\n{context}\n\nSoalan pelajar:\n{query}\n\nJawapan:"

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=GENERATION_SYSTEM_PROMPT,
            # gemini-3.6-flash spends ~650-700 tokens on hidden "thinking" before the
            # visible answer, regardless of thinking_budget (tested — not a reliable
            # cap for this model). 1024 left too little room and silently truncated
            # mid-answer (finish_reason="MAX_TOKENS"); this leaves real headroom.
            max_output_tokens=4096,
        ),
    )
    raw = (response.text or "").strip()

    m = CITATION_RE.search(raw)
    idx = int(m.group(1)) - 1 if m and 0 <= int(m.group(1)) - 1 < len(chunks) else 0  # fallback: rank-1 chunk
    cited = chunks[idx]["metadata"]
    text = CITATION_RE.sub("", raw).strip()  # strip the marker out of the student-facing text

    source = Source(chapter_title=cited["chapter_title"], section_title=cited["section_title"], page=cited["current_page"])
    return Answer(text=text, source=source, query=query)


def answer_question(query: str, k: int = 5) -> Answer:
    retriever = Retriever()
    chunks = retriever.retrieve(query, k=k)
    return generate_answer(query, chunks)
