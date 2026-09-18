from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from jwt import PyJWKClient

from ..rag_pipeline.retrieval import Retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once, before the server starts accepting requests
    # See generation.answer_question()'s retriever
    # parameter for why this matters (avoids reloading the embedding model per call).
    print("Loading retriever...")
    app.state.retriever = Retriever()
    print("Retriever ready.")

    print("Loading Supabase JWKS client...")
    app.state.jwks_client = PyJWKClient(os.environ["SUPABASE_JWKS_URL"])
    print("JWKS client ready — server is now accepting requests.")
    yield
