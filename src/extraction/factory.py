"""
extraction/factory.py
======================
The only import path the rest of the system needs.

To add a new backend:
  1. Write YourExtractor(ExtractorBase) in a new file.
  2. Add one line to _REGISTRY below.
  3. Done.  No other file changes required.

Usage:
    from extraction.factory import ExtractorFactory

    extractor = ExtractorFactory.create("marker")
    result = extractor.extract("chapter01.pdf")

    # Switch to pdfplumber without changing any other code:
    extractor = ExtractorFactory.create("pdfplumber", config={"two_column_classes": [11, 12]})
    result = extractor.extract("chapter01.pdf", class_no=11)
"""

from __future__ import annotations

from typing import Optional

from .base import ExtractorBase


def _get_registry() -> dict[str, type[ExtractorBase]]:
    """
    Lazy registry: imports only happen when the backend is requested.
    This means importing factory.py does NOT trigger marker, docling, etc.
    unless they are actually used.
    """
    from .pdfplumber_extractor import PdfPlumberExtractor
    from .marker_extractor import MarkerExtractor
    from .precomputed_extractor import PrecomputedExtractor
    registry: dict[str, type[ExtractorBase]] = {
        "pdfplumber": PdfPlumberExtractor,
        "marker": MarkerExtractor,
        "precomputed": PrecomputedExtractor,
        # Future backends — add here, no other changes needed:
        # "docling":       DoclingExtractor,
        # "pymupdf4llm":   PyMuPDF4LLMExtractor,
    }
    return registry


class ExtractorFactory:
    """
    Creates extractor instances by name.

    All configuration is passed through the `config` dict so that callers
    do not need to know which constructor arguments each backend expects.
    """

    @staticmethod
    def create(
        backend: str = "pdfplumber",
        config: Optional[dict] = None,
    ) -> ExtractorBase:
        registry = _get_registry()
        key = backend.lower().strip()
        if key not in registry:
            available = ", ".join(sorted(registry.keys()))
            raise ValueError(
                f"Unknown extraction backend: '{backend}'. "
                f"Available: {available}"
            )
        return registry[key](config=config)

    @staticmethod
    def available_backends() -> list[str]:
        return sorted(_get_registry().keys())
