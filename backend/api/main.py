from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

# Each process needs its own .env load — cli.py's load_dotenv() call doesn't apply
# here, this is a separate entrypoint (run via `uvicorn backend.api.main:app`).
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from .lifespan import lifespan
from .routers import ask

app = FastAPI(title="StudyFlow API", lifespan=lifespan)
app.include_router(ask.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
