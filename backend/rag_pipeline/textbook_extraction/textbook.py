from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..config import TEXTBOOKS


# Represents a single section of a textbook (e.g. "Bab 4, Section 4.2")
# Everything the pipeline needs to know about one chunk of content lives here
@dataclass
class TextbookSection:
    textbook_id: str
    form: str
    bab: int
    chapter_title: str
    section_id: str  # e.g. "4.2"
    title: str
    page_start: int
    page_end: int

    @property
    def bab_str(self) -> str:
        # Zero-padded chapter folder name, e.g. bab04
        return f"bab{self.bab:02d}"

    @property
    def sec_str(self) -> str:
        # Safe filename for this section, e.g. sec_4_2
        return f"sec_{self.section_id.replace('.', '_')}"

    @property
    def custom_id(self) -> str:
        # Unique ID used to label this request inside the Anthropic batch
        # '-' as separator: safe since textbook IDs use '_', never '-'
        return f"{self.textbook_id}-{self.bab_str}-{self.sec_str}"

    @property
    def output_path(self) -> Path:
        # Where the extracted .md file will be written
        return TEXTBOOKS[self.textbook_id]["output_dir"] / self.bab_str / f"{self.sec_str}.md"

    @property
    def image_paths(self) -> list[Path]:
        # Returns the PNG files for every page in this section (skips any missing files)
        images_dir: Path = TEXTBOOKS[self.textbook_id]["images_dir"]
        return [
            images_dir / f"page_{p:03d}.png"
            for p in range(self.page_start, self.page_end + 1)
            if (images_dir / f"page_{p:03d}.png").exists()
        ]

    def is_extracted(self) -> bool:
        # True if we've already written the output .md file for this section
        return self.output_path.exists()


def load_sections(textbook_ids: list[str] | None = None) -> list[TextbookSection]:
    # Reads the TOC JSON files and returns a flat list of every section across both textbooks
    # Pass textbook_ids to restrict to one textbook, e.g. ["sejarah_t4"]
    ids = textbook_ids or list(TEXTBOOKS.keys())
    sections: list[TextbookSection] = []
    for tid in ids:
        cfg = TEXTBOOKS[tid]
        with open(cfg["toc_file"]) as f:
            toc = json.load(f)
        for chapter in toc["chapters"]:
            for sec in chapter["sections"]:
                sections.append(TextbookSection(
                    textbook_id=tid,
                    form=cfg["form"],
                    bab=chapter["bab"],
                    chapter_title=chapter["title"],
                    section_id=sec["section"],
                    title=sec["title"],
                    page_start=sec["page_start"],
                    page_end=sec["page_end"],
                ))
    return sections


def parse_section_arg(arg: str) -> TextbookSection:
    # Converts a CLI string like "sejarah_t4/bab04/sec_4_2" into a TextbookSection object
    parts = arg.split("/")
    if len(parts) != 3:
        raise ValueError(f"Section arg must be 'textbook/babNN/sec_X_Y', got: {arg!r}")
    textbook_id, bab_str, sec_str = parts
    bab = int(bab_str.removeprefix("bab"))
    section_id = sec_str.removeprefix("sec_").replace("_", ".", 1)
    for s in load_sections([textbook_id]):
        if s.bab == bab and s.section_id == section_id:
            return s
    raise ValueError(f"Section not found in TOC: {arg!r}")
