from __future__ import annotations

from fastapi import Request

from ..rag_pipeline.retrieval import Retriever


def get_retriever(request: Request) -> Retriever:
    # The actual Retriever is built once in main.py's lifespan handler, before the
    # server starts accepting requests — this just hands back that already-built
    # instance from app.state for each request that depends on it.
    return request.app.state.retriever
