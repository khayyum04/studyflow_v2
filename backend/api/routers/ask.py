from __future__ import annotations

from fastapi import APIRouter, Depends

from ...rag_pipeline.generation import answer_question
from ...rag_pipeline.retrieval import Retriever
from ..dependencies import get_retriever
from ..images import page_image_url, section_image_urls
from ..schemas import AskRequest, AskResponse, SourceOut

router = APIRouter()


# Plain `def`, not `async def` — answer_question() makes blocking calls (ChromaDB
# query, Gemini API request) via fully synchronous clients. FastAPI runs plain `def`
# path functions in a thread pool automatically, so a blocking call inside doesn't
# stall the server. An `async def` here would freeze the whole event loop — and every
# other concurrent request with it — for the duration of each blocking call.
@router.post("/ask", response_model=AskResponse)
def ask_question(body: AskRequest, retriever: Retriever = Depends(get_retriever)) -> AskResponse:
    answer = answer_question(body.question, retriever=retriever)

    source = None
    if answer.source:
        s = answer.source
        source = SourceOut(
            chapter_title=s.chapter_title,
            section_title=s.section_title,
            page=s.page,
            page_start=s.page_start,
            page_end=s.page_end,
            page_image=page_image_url(s.form, s.page) if s.page else None,
            section_images=section_image_urls(s.form, s.page_start, s.page_end),
        )
    return AskResponse(text=answer.text, source=source, query=answer.query)
