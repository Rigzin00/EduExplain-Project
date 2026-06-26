"""
extraction/base.py
==================
Defines the stable contract between PDF extraction backends and the rest of
the pipeline.  Nothing downstream should ever import from a specific extractor
module.  Only ExtractionResult and ExtractorBase should cross this boundary.

Adding a new backend (Docling, Gemini, etc.) requires:
  1. Subclassing ExtractorBase
  2. Registering it in factory.py
  3. Zero changes to normaliser.py, parser.py, or the downstream pipeline.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data contract — the only thing that crosses the seam
# ---------------------------------------------------------------------------

@dataclass
class PageData:
    """
    One page of extracted content.

    `text`     — plain text for the structural parsing layer.
    `markdown` — richer representation when the backend produces it
                 (Marker, Docling). May be None for plain-text backends.
    `equations`— LaTeX strings extracted from this page.  Empty list when
                 the backend does not support equation extraction.
    `page_no`  — 1-indexed.
    """
    page_no: int
    text: str
    markdown: Optional[str] = None
    equations: list[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """
    Everything the extraction layer returns to the rest of the pipeline.

    Backends fill in what they can; consumers check for None / empty lists
    before using optional fields.  The `pages` list is always required.
    """
    pages: list[PageData]

    # Optional enrichments — populated by capable backends
    language: Optional[str] = None          # e.g. "en", "hi"
    has_equations: bool = False             # True when equations were found
    has_tables: bool = False                # True when tables were found
    extraction_backend: str = "unknown"

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def as_flat_pages(self) -> list[dict]:
        """
        Compatibility shim — returns the list[{page_no, text}] format that
        the existing parse_numbered_ncert / parse_class89 functions expect.
        This lets you keep those parsers completely unchanged.
        """
        return [{"page_no": p.page_no, "text": p.text} for p in self.pages]
    def to_json(self, path: str) -> None:
        """Serialize extraction result to a JSON file."""
        import json
        data = {
            "extraction_backend": self.extraction_backend,
            "language": self.language,
            "has_equations": self.has_equations,
            "has_tables": self.has_tables,
            "pages": [
                {
                    "page_no": p.page_no,
                    "text": p.text,
                    "markdown": p.markdown,
                    "equations": p.equations,
                }
                for p in self.pages
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "ExtractionResult":
        """Load an ExtractionResult that was previously serialized."""
        import json
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        pages = [
            PageData(
                page_no=p["page_no"],
                text=p["text"],
                markdown=p.get("markdown"),
                equations=p.get("equations", []),
            )
            for p in data["pages"]
        ]
        return cls(
            pages=pages,
            language=data.get("language"),
            has_equations=data.get("has_equations", False),
            has_tables=data.get("has_tables", False),
            extraction_backend=data.get("extraction_backend", "unknown"),
        )

# ---------------------------------------------------------------------------
# Abstract base — every backend implements this interface
# ---------------------------------------------------------------------------

class ExtractorBase(ABC):
    """
    One method to implement.  Return an ExtractionResult.  That is all.

    Backends handle their own setup (imports, model loading, etc.) inside
    __init__ so that importing base.py never triggers heavy dependencies.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    @abstractmethod
    def extract(self, pdf_path: str) -> ExtractionResult:
        """
        Extract text (and optionally structured content) from a PDF.

        Parameters
        ----------
        pdf_path : str
            Absolute or relative path to the PDF file.

        Returns
        -------
        ExtractionResult
            Normalised extraction result.  Never raises on a missing page —
            log a warning and return what could be extracted.
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers available to all backends
    # ------------------------------------------------------------------

    @staticmethod
    def is_valid_pdf(pdf_path: str) -> bool:
        import os
        return os.path.isfile(pdf_path) and pdf_path.lower().endswith(".pdf")

    @staticmethod
    def detect_two_column_layout(page_width: float, text_blocks: list) -> bool:
        """
        Heuristic: if block x-coordinates cluster into two groups separated
        by a gap > 35% of page width, the page is two-column.
        Used by backends that expose block-level coordinates.
        """
        if not text_blocks:
            return False
        xs = [b.get("x0", 0) for b in text_blocks]
        midpoint = page_width * 0.35
        left = sum(1 for x in xs if x < midpoint)
        right = sum(1 for x in xs if x >= midpoint)
        # Both columns need meaningful content
        return left > 2 and right > 2
