from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel

from .config import CHUNKS_FILE, TEXTBOOKS

SUBJECT = "Sejarah"  # only subject in the corpus today

MERGE_THRESHOLD = 50           # chunks under this many words get merged into a neighbor
LAST_CHUNK_MERGE_THRESHOLD = 150  # the last chunk in a file has no "next", so it gets a higher bar
SPLIT_THRESHOLD = 500          # chunks over this many words get split at a paragraph break

# Matches a heading line with 1-3 leading '#' (not 4+, which is out of scope for splitting)
HEADING_RE = re.compile(r"^(#{1,3})(?!#)\s+(.+?)\s*$")

# Extraction pipeline stamps `<!-- page N -->` immediately before that page's content begins
# (case varies between "page" and "PAGE" across extracted files, so match case-insensitively)
PAGE_RE = re.compile(r"<!--\s*page\s+(\d+)\s*-->", re.IGNORECASE)
PAGE_LINE_RE = re.compile(r"^\s*<!--\s*page\s+\d+\s*-->\s*$\n?", re.IGNORECASE | re.MULTILINE)


class Chunk(BaseModel):
    chunk_id: str
    form: str
    subject: str
    chapter_id: int
    chapter_title: str
    section_id: str
    section_title: str
    current_page: int
    page_start: int
    page_end: int
    content_md: str
    word_count: int


@dataclass
class _Candidate:
    # One heading-delimited piece of a section file, tracked through all three passes
    heading_line: str
    parent_heading: str | None
    body: str  # everything after the heading line (not yet stripped/joined)
    start_page: int  # page number in effect where this candidate's content begins

    @property
    def content(self) -> str:
        return f"{self.heading_line}\n\n{self.body}"

    @property
    def word_count(self) -> int:
        return len(self.content.split())


def _parse_file(path: Path) -> tuple[dict, str]:
    # Splits the YAML frontmatter block from the markdown body
    # maxsplit=2 so any "---" horizontal rules later in the body are left untouched
    text = path.read_text(encoding="utf-8")
    _, frontmatter_raw, body = text.split("---\n", 2)
    return yaml.safe_load(frontmatter_raw), body


def _split_headings(body: str, default_page: int) -> list[_Candidate]:
    # Pass 1: walk line by line, starting a new candidate at every #/##/### heading.
    # Tracks a heading stack so each candidate knows its immediate parent (one level up).
    # Also tracks the most recent `<!-- page N -->` marker so each candidate can record
    # the page its content starts on.
    candidates: list[_Candidate] = []
    stack: dict[int, str] = {}
    current: dict | None = None
    current_page = default_page

    def close_current() -> None:
        if current is None:
            return
        body_text = "\n".join(current["lines"]).strip()
        if body_text:  # skip headings immediately followed by another heading (no content)
            candidates.append(_Candidate(current["heading_line"], current["parent"], body_text, current["start_page"]))

    for line in body.split("\n"):
        m = HEADING_RE.match(line)
        if m:
            close_current()
            level = len(m.group(1))
            stack = {lvl: h for lvl, h in stack.items() if lvl < level}
            current = {
                "heading_line": line.strip(),
                "parent": stack.get(level - 1),
                "lines": [],
                "start_page": current_page,
            }
            stack[level] = current["heading_line"]
            continue
        pm = PAGE_RE.match(line.strip())
        if pm:
            current_page = int(pm.group(1))
        if current is not None:
            current["lines"].append(line)

    close_current()
    return candidates


def _merge_small(candidates: list[_Candidate]) -> list[_Candidate]:
    # Pass 2: merge chunks under the word threshold into a neighbor.
    chunks = list(candidates)
    i = 0
    while i < len(chunks):
        c = chunks[i]
        is_last = i == len(chunks) - 1

        if is_last:
            if c.word_count < LAST_CHUNK_MERGE_THRESHOLD and len(chunks) > 1:
                prev = chunks[i - 1]
                # prev comes first in the document, so its start_page is already the earliest
                chunks[i - 1] = _Candidate(prev.heading_line, prev.parent_heading, f"{prev.body}\n\n{c.content}", prev.start_page)
                del chunks[i]
            else:
                i += 1
        elif c.word_count < MERGE_THRESHOLD:
            nxt = chunks[i + 1]
            # c comes first in the document, so its start_page is the earliest of the two
            chunks[i + 1] = _Candidate(nxt.heading_line, nxt.parent_heading, f"{c.content}\n\n{nxt.body}", c.start_page)
            del chunks[i]
            # don't advance i — re-examine the newly merged chunk in case it's still small
        else:
            i += 1
    return chunks


def _table_guarded_split(body: str, split_at: int) -> int:
    # If split_at lands inside a markdown table, push it to just after the table ends
    lines = body.split("\n")
    offsets = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1

    line_idx = 0
    for i, off in enumerate(offsets):
        if off <= split_at:
            line_idx = i
        else:
            break

    if "|" not in lines[line_idx]:
        return split_at
    j = line_idx
    while j < len(lines) and "|" in lines[j]:
        j += 1
    return offsets[j] if j < len(offsets) else len(body)


def _split_one(c: _Candidate) -> list[_Candidate]:
    # Pass 3 (per-candidate): split at the paragraph break closest to the midpoint,
    # recursing on both halves so nothing over the threshold survives
    if c.word_count <= SPLIT_THRESHOLD:
        return [c]

    boundaries = [m.start() for m in re.finditer(r"\n\n", c.body)]
    if not boundaries:
        return [c]  # nothing to split on — leave it oversized

    midpoint = len(c.body) / 2
    split_at = min(boundaries, key=lambda b: abs(b - midpoint))
    split_at = _table_guarded_split(c.body, split_at)

    first_body = c.body[:split_at].strip()
    second_body = c.body[split_at:].strip()
    if not first_body or not second_body:
        return [c]

    # If a page marker falls within the first half, the second half starts on that later page
    page_matches = PAGE_RE.findall(first_body)
    second_start_page = int(page_matches[-1]) if page_matches else c.start_page

    first = _Candidate(c.heading_line, c.parent_heading, first_body, c.start_page)
    second = _Candidate(c.heading_line, c.parent_heading, second_body, second_start_page)
    return _split_one(first) + _split_one(second)


def _split_large(candidates: list[_Candidate]) -> list[_Candidate]:
    result: list[_Candidate] = []
    for c in candidates:
        result.extend(_split_one(c))
    return result


def _finalize(c: _Candidate) -> str:
    # Prepends the immediate parent heading (one level only) so the chunk is self-contained
    content = c.content
    if c.parent_heading:
        content = f"{c.parent_heading}\n\n{content}"
    # Page markers have done their job (tracked via start_page); strip them from the
    # user/LLM-facing content and collapse the blank line they leave behind
    content = PAGE_LINE_RE.sub("", content)
    return re.sub(r"\n{3,}", "\n\n", content).strip()


def _chunk_file(path: Path, textbook_id: str) -> list[Chunk]:
    frontmatter, body = _parse_file(path)
    candidates = _split_headings(body, frontmatter["page_start"])
    candidates = _merge_small(candidates)
    candidates = _split_large(candidates)

    bab_str = path.parent.name
    sec_str = path.stem
    chunks = []
    for i, c in enumerate(candidates):
        content_md = _finalize(c)
        chunks.append(Chunk(
            chunk_id=f"{textbook_id}_{bab_str}_{sec_str}_{i:02d}",
            form=frontmatter["form"],
            subject=SUBJECT,
            chapter_id=frontmatter["bab"],
            chapter_title=frontmatter["chapter_title"],
            section_id=str(frontmatter["section"]),
            section_title=frontmatter["section_title"],
            current_page=c.start_page,
            page_start=frontmatter["page_start"],
            page_end=frontmatter["page_end"],
            content_md=content_md,
            word_count=len(content_md.split()),
        ))
    return chunks


def run_chunking(textbook_ids: list[str] | None = None) -> None:
    # Main entry point — chunks every section file for the given textbooks (or all of them)
    # and fully regenerates CHUNKS_FILE from that scope
    ids = textbook_ids or list(TEXTBOOKS.keys())
    all_chunks: list[Chunk] = []
    for tid in ids:
        paths = sorted(TEXTBOOKS[tid]["output_dir"].glob("bab*/sec_*.md"))
        print(f"Chunking {tid} ({len(paths)} sections)...")
        for path in paths:
            all_chunks.extend(_chunk_file(path, tid))

    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_FILE.write_text(
        json.dumps([c.model_dump() for c in all_chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {len(all_chunks)} chunks to {CHUNKS_FILE}")
