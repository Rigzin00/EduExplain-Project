"""
extraction/pdfplumber_extractor.py
===================================
Your original extraction logic, unchanged, wrapped behind ExtractorBase.

The only difference from the original code:
  - extract_pages_class10 / extract_pages_class11_12 are now private methods
  - extract() is the single public entry point
  - clean_text() is imported from normaliser.py to avoid duplication

This extractor is the reference implementation.  It is intentionally simple
and does not attempt equation extraction or layout classification.
"""

from __future__ import annotations

import re
from typing import Optional

from .base import ExtractorBase, ExtractionResult, PageData


class PdfPlumberExtractor(ExtractorBase):
    """
    Extraction backend using pdfplumber.

    config options (passed as dict):
        two_column_classes : list[int]  — class numbers that use two-column
                             layout.  Defaults to [11, 12].
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self._two_column_classes: list[int] = self.config.get(
            "two_column_classes", [11, 12]
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(self, pdf_path: str, class_no: int = 10) -> ExtractionResult:  # type: ignore[override]
        """
        Extract pages from a PDF.

        Parameters
        ----------
        pdf_path : str
        class_no : int
            Used to decide two-column vs single-column strategy.
            Pass it via the extract() call rather than storing it on the
            extractor instance so one extractor object can handle all classes.
        """
        try:
            import pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber is not installed. Run: pip install pdfplumber"
            )

        if class_no in self._two_column_classes:
            pages = self._extract_two_column(pdf_path)
        else:
            pages = self._extract_single_column(pdf_path)

        return ExtractionResult(
            pages=pages,
            extraction_backend="pdfplumber",
        )

    # ------------------------------------------------------------------
    # Private extraction strategies
    # ------------------------------------------------------------------

    def _extract_single_column(self, pdf_path: str) -> list[PageData]:
        """Single-column extraction — original extract_pages_class10 logic."""
        import pdfplumber

        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                text = _clean_text(text)
                pages.append(PageData(page_no=i + 1, text=text))
        return pages

    def _extract_two_column(self, pdf_path: str) -> list[PageData]:
        """
        Two-column extraction — original extract_pages_class11_12 logic.
        Splits each page at the 50% width mark, concatenates left + right.
        """
        import pdfplumber

        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                w, h = page.width, page.height
                divider = w * 0.50
                left_text = page.crop((0, 0, divider, h)).extract_text() or ""
                right_text = page.crop((divider, 0, w, h)).extract_text() or ""
                text = left_text + "\n" + right_text
                pages.append(PageData(page_no=i + 1, text=text))
        return pages


# ---------------------------------------------------------------------------
# Module-level clean_text — duplicated here to keep this module self-contained
# during migration.  Once normaliser.py exists, replace with:
#   from ..normaliser import clean_text as _clean_text
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Normalize extracted PDF text (NCERT-specific artifacts)."""
    if not text:
        return ""
    text = re.sub(r'(.)\1{3,}', lambda m: m.group(1), text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'Reprint \d{4}-\d{2,4}', '', text)
    text = re.sub(r'Chapter-\d+\.indd.*', '', text)
    text = re.sub(r'\d+\s+Science\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Science\s+\d+\s*$', '', text, flags=re.MULTILINE)
    return text.strip()
