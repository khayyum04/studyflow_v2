from __future__ import annotations

import os
from dataclasses import dataclass

from google import genai
from google.genai import types

from .config import GEMINI_MODEL, GENERATION_SYSTEM_PROMPT
from .retrieval import Retriever

NO_CONTEXT_ANSWER = "Maaf, saya tidak mempunyai maklumat yang mencukupi untuk menjawab soalan ini."


@dataclass(frozen=True)
class Source:
    chapter_title: str
    section_title: str
    page_start: int
    page_end: int


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Source]
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


def _build_sources(chunks: list[dict]) -> list[Source]:
    # Dedup down to unique sections — several chunks often come from the same section
    seen = set()
    sources = []
    for chunk in chunks:
        meta = chunk["metadata"]
        key = (meta["chapter_title"], meta["section_title"])
        if key in seen:
            continue
        seen.add(key)
        sources.append(Source(
            chapter_title=meta["chapter_title"],
            section_title=meta["section_title"],
            page_start=meta["page_start"],
            page_end=meta["page_end"],
        ))
    return sources


def generate_answer(query: str, chunks: list[dict]) -> Answer:
    if not chunks:
        return Answer(text=NO_CONTEXT_ANSWER, sources=[], query=query)

    context = _format_context(chunks)
    user_message = f"Konteks:\n{context}\n\nSoalan pelajar:\n{query}\n\nJawapan:"

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=GENERATION_SYSTEM_PROMPT,
            max_output_tokens=1024,
        ),
    )
    text = (response.text or "").strip()
    return Answer(text=text, sources=_build_sources(chunks), query=query)


def answer_question(query: str, k: int = 5) -> Answer:
    retriever = Retriever()
    chunks = retriever.retrieve(query, k=k)
    return generate_answer(query, chunks)
