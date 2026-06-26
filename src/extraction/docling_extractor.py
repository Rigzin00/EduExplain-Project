"""
extraction/docling_extractor.py  (stub — not yet implemented)
=============================================================
Template for adding a new extraction backend.

Copy this file, replace "Docling" with your backend name,
implement extract(), and register in factory.py.  That is the entire
process.  No other file needs to change.
"""

from __future__ import annotations

from typing import Optional

from .base import ExtractorBase, ExtractionResult, PageData


class DoclingExtractor(ExtractorBase):
    """
    Extraction backend using the Docling library (IBM Research).

    Docling excels at:
      - Scientific PDFs with complex layouts
      - Table extraction into structured form
      - Multi-language documents

    Installation:
        pip install docling

    Config options:
        ocr : bool — enable OCR for scanned pages. Default False.
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self._ocr: bool = self.config.get("ocr", False)

    def extract(self, pdf_path: str, **kwargs) -> ExtractionResult:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError:
            raise ImportError(
                "docling is not installed. Run: pip install docling"
            )

        converter = DocumentConverter()
        result = converter.convert(pdf_path)

        pages: list[PageData] = []

        # Docling returns a document object with pages.
        # Adapt to PageData here.
        for page in result.document.pages:
            text = page.export_to_text()   # adjust to actual Docling API
            pages.append(PageData(
                page_no=page.page_no,
                text=text,
                markdown=None,
                equations=[],
            ))

        return ExtractionResult(
            pages=pages,
            extraction_backend="docling",
        )


# To activate:
# 1. Implement extract() above using the real Docling API.
# 2. Uncomment this line in factory.py:
#      "docling": DoclingExtractor,
