from __future__ import annotations

import os
import re
from collections import Counter
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
    "Pick exactly one, even if you drew on more than one. if there is none, say 'RUJUKAN: TIADA' instead."
)


@dataclass(frozen=True)
class Source:
    chapter_title: str
    section_title: str
    page: int          # current_page of the cited chunk; 0 = section-level fallback,
                        # see page_start/page_end instead of trusting this as a pinpoint page
    page_start: int
    page_end: int
    form: str          # "Form 4"/"Form 5" — which textbook this came from, needed to
                        # locate its page images (see backend/api/images.py)


@dataclass(frozen=True)
class Answer:
    text: str
    source: Source | None
    query: str


def _most_common_section(chunks: list[dict]) -> dict:
    # Groups retrieved chunks by section and returns a representative chunk's metadata
    # for whichever section appeared most often. Counter.most_common() ties toward
    # whichever key it saw first, and chunks arrive pre-sorted by retrieval rank, so a
    # tie naturally favors the higher-ranked section — no separate tie-break needed.
    counts = Counter((c["metadata"]["chapter_title"], c["metadata"]["section_title"]) for c in chunks)
    target = counts.most_common(1)[0][0]
    return next(
        c["metadata"] for c in chunks
        if (c["metadata"]["chapter_title"], c["metadata"]["section_title"]) == target
    )


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
    text = CITATION_RE.sub("", raw).strip()  # strip the marker out of the student-facing text

    m = CITATION_RE.search(raw)
    idx = int(m.group(1)) - 1 if m else -1
    if 0 <= idx < len(chunks):
        cited = chunks[idx]["metadata"]
        source = Source(
            chapter_title=cited["chapter_title"],
            section_title=cited["section_title"],
            page=cited["current_page"],
            page_start=cited["page_start"],
            page_end=cited["page_end"],
            form=cited["form"],
        )
    else:
        # Model didn't give a usable citation. Rather than arbitrarily picking the
        # rank-1 chunk, use whichever section appeared most often among the retrieved
        # chunks — page=0 signals this is section-level, not a specific cited page.
        fallback = _most_common_section(chunks)
        source = Source(
            chapter_title=fallback["chapter_title"],
            section_title=fallback["section_title"],
            page=0,
            page_start=fallback["page_start"],
            page_end=fallback["page_end"],
            form=fallback["form"],
        )

    return Answer(text=text, source=source, query=query)


def answer_question(query: str, k: int = 5, retriever: Retriever | None = None) -> Answer:
    # The CLI has no retriever to reuse across calls, so it leaves this as None and
    # gets a fresh one each time. The API passes in the one built once at server
    # startup (see backend/api/main.py's lifespan) instead of paying that cost per request.
    retriever = retriever or Retriever()
    chunks = retriever.retrieve(query, k=k)
    return generate_answer(query, chunks)
