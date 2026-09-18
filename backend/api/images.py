from __future__ import annotations

from ..textbook_extraction.config import TEXTBOOKS

# Matches the per-textbook mounts main.py sets up via StaticFiles — one prefix per
# textbook_id, e.g. /images/sejarah_t4/page_100.png serves
# TEXTBOOKS["sejarah_t4"]["images_dir"] / "page_100.png".
IMAGES_URL_PREFIX = "/images"

def _textbook_id_from_form(form: str) -> str | None:
    for textbook_id, config in TEXTBOOKS.items():
        if config["form"] == form:
            return textbook_id
    return None


def page_image_url(form: str, page: int) -> str | None:
    # None when the textbook can't be resolved, or that page's PNG doesn't exist on
    # disk (data/textbooks/ is gitignored — only present where extraction actually ran).
    textbook_id = _textbook_id_from_form(form)
    if textbook_id is None:
        return None
    images_dir = TEXTBOOKS[textbook_id]["images_dir"]
    if not (images_dir / f"page_{page:03d}.png").exists():
        return None
    return f"{IMAGES_URL_PREFIX}/{textbook_id}/page_{page:03d}.png"


def section_image_urls(form: str, page_start: int, page_end: int) -> list[str]:
    return [
        url for p in range(page_start, page_end + 1)
        if (url := page_image_url(form, p)) is not None
    ]
