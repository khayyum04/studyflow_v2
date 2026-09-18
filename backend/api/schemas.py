from __future__ import annotations

from pydantic import BaseModel

# Deliberately separate from generation.py's Answer/Source dataclasses — these are the
# API's public contract. Changing an internal dataclass shouldn't silently change what
# clients receive; the router explicitly maps one to the other.


class AskRequest(BaseModel):
    question: str


class SourceOut(BaseModel):
    chapter_title: str
    section_title: str
    page: int                  # 0 = section-level fallback, see page_start/page_end instead
    page_start: int
    page_end: int
    page_image: str | None     # URL for the single selected page; None when page == 0
    section_images: list[str]  # URLs for page_start..page_end (only pages whose PNG exists)


class AskResponse(BaseModel):
    text: str
    source: SourceOut | None
    query: str
