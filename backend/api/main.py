from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Each process needs its own .env load — cli.py's load_dotenv() call doesn't apply
# here, this is a separate entrypoint (run via `uvicorn backend.api.main:app`).
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from ..textbook_extraction.config import TEXTBOOKS
from .images import IMAGES_URL_PREFIX
from .lifespan import lifespan
from .routers import ask

app = FastAPI(title="StudyFlow API", lifespan=lifespan)
app.include_router(ask.router)

# One mount per textbook so only the page PNGs are exposed
# images.py builds URLs matching these exact mount points.
for _textbook_id, _cfg in TEXTBOOKS.items():
    app.mount(f"{IMAGES_URL_PREFIX}/{_textbook_id}", StaticFiles(directory=_cfg["images_dir"]))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
