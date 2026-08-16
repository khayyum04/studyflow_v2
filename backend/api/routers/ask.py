from __future__ import annotations

from fastapi import APIRouter, Depends

from ...rag_pipeline.generation import answer_question
from ...rag_pipeline.retrieval import Retriever
from ..dependencies import get_retriever
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
        source = SourceOut(
            chapter_title=answer.source.chapter_title,
            section_title=answer.source.section_title,
            page=answer.source.page,
        )
    return AskResponse(text=answer.text, source=source, query=answer.query)
