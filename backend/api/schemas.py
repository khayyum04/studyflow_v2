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
    page: int


class AskResponse(BaseModel):
    text: str
    source: SourceOut | None
    query: str
