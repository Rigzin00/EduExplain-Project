"""
extraction/marker_extractor.py
================================
Extraction backend using Marker 1.x.

API changes from 0.x to 1.x:
  OLD (0.x, broken):  from marker.convert import convert_single_pdf
                      from marker.models import load_all_models
  NEW (1.x, correct): from marker.converters.pdf import PdfConverter
                      from marker.models import create_model_dict

This file is used when running locally with a GPU (HPC) via:
    ExtractorFactory.create("marker")

For Colab extraction, the standalone notebook (ncert_marker_extraction.py)
runs the same core logic and produces ExtractionResult-compatible JSON.
Those JSONs are loaded locally via:
    ExtractorFactory.create("precomputed", config={"result_path": "..."})

Installation:
    pip install marker-pdf      # installs 1.x

Config options (passed as dict to ExtractorFactory.create):
    langs          : list[str]  — language hints. Default ["en"].
    batch_multiplier: int       — Marker batch size multiplier. Default 2.
    use_gpu        : bool       — hint only; Marker auto-detects GPU. Default False.
"""

from __future__ import annotations

import re
from typing import Optional

from .base import ExtractorBase, ExtractionResult, PageData


class MarkerExtractor(ExtractorBase):

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self._langs: list[str] = self.config.get("langs", ["en"])
        self._batch_multiplier: int = self.config.get("batch_multiplier", 2)
        self._converter = None   # lazy-loaded on first extract() call

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(self, pdf_path: str, **kwargs) -> ExtractionResult:
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.config.parser import ConfigParser
        except ImportError:
            raise ImportError(
                "marker-pdf 1.x is not installed.\n"
                "Run: pip install marker-pdf\n"
                "Note: marker.convert (0.x API) no longer exists in 1.x."
            )

        if self._converter is None:
            marker_config = {
                "output_format": "json",
                "langs": self._langs,
                "disable_image_extraction": True,
            }
            cp = ConfigParser(marker_config)
            self._converter = PdfConverter(
                config=cp.generate_config_dict(),
                artifact_dict=create_model_dict(),
                processor_list=cp.get_processors(),
                renderer=cp.get_renderer(),
            )

        rendered = self._converter(pdf_path)
        return self._rendered_to_extraction_result(rendered, pdf_path)

    # ------------------------------------------------------------------
    # Private: convert Marker 1.x JSON output → ExtractionResult
    # ------------------------------------------------------------------

    def _rendered_to_extraction_result(
        self, rendered, pdf_path: str
    ) -> ExtractionResult:
        """
        Convert Marker's JSON-format rendered output to ExtractionResult.

        Uses JSON output_format (not markdown) because:
        - Each list item in rendered.children IS exactly one page.
          No heuristic splitting needed. Page boundaries are authoritative.
        - block_type="Page" children give clean reading-order text per page.
        - Equation blocks are explicitly typed — no regex needed.
        - PageHeader / PageFooter blocks can be filtered out cleanly.
        """
        page_blocks: list[dict] = rendered.children
        metadata = rendered.metadata or {}
        page_stats: list[dict] = metadata.get("page_stats", [])

        extraction_method_by_page: dict[int, str] = {
            int(ps["page_id"]): ps.get("text_extraction_method", "unknown")
            for ps in page_stats
            if "page_id" in ps
        }

        pages: list[PageData] = []
        has_equations = False
        has_tables = False

        for page_block in page_blocks:
            block_id: str = page_block.get("id", "")
            m = re.search(r"/page/(\d+)/", block_id)
            page_idx = int(m.group(1)) if m else len(pages)
            page_no = page_idx + 1  # 1-indexed

            children = page_block.get("children") or []
            plain_text = _blocks_to_plain_text(children)
            equations  = _extract_equations_from_tree(page_block)
            page_has_table = _tree_contains_table(children)

            if equations:
                has_equations = True
            if page_has_table:
                has_tables = True

            pages.append(PageData(
                page_no=page_no,
                text=plain_text,
                markdown=None,
                equations=equations,
            ))

        language = metadata.get("language", None)

        return ExtractionResult(
            pages=pages,
            language=language,
            has_equations=has_equations,
            has_tables=has_tables,
            extraction_backend="marker",
        )


# ---------------------------------------------------------------------------
# Shared helpers (also used by the Colab notebook inline copy)
# ---------------------------------------------------------------------------

_SKIP_BLOCK_TYPES = {"PageHeader", "PageFooter"}
_TABLE_BLOCK_TYPES = {"Table", "TableGroup"}
_EQUATION_BLOCK_TYPES = {"Equation", "TextInlineMath"}


def _blocks_to_plain_text(children: list[dict]) -> str:
    """
    Convert a page's children to plain text, stripping HTML tags.
    PageHeader and PageFooter blocks are excluded.
    """
    lines: list[str] = []

    def _walk(node: dict):
        if node.get("block_type") in _SKIP_BLOCK_TYPES:
            return
        html = node.get("html", "")
        if html:
            text = re.sub(r"</(p|div|h[1-6]|li|tr|td|th)>", "\n", html)
            text = re.sub(r"<br\s*/?>", "\n", text)
            text = re.sub(r"<[^>]+>", "", text)
            text = re.sub(r"&amp;", "&", text)
            text = re.sub(r"&lt;", "<", text)
            text = re.sub(r"&gt;", ">", text)
            text = re.sub(r"&nbsp;", " ", text)
            text = text.strip()
            if text:
                lines.append(text)
        else:
            for child in (node.get("children") or []):
                _walk(child)

    for block in children:
        _walk(block)

    return "\n".join(lines)


def _extract_equations_from_tree(node: dict) -> list[str]:
    """Recursively collect LaTeX from Equation and TextInlineMath blocks."""
    seen: set[str] = set()
    results: list[str] = []

    def _walk(n: dict):
        if n.get("block_type") in _EQUATION_BLOCK_TYPES:
            html = n.get("html", "")
            latex = re.sub(r"<[^>]+>", "", html).strip()
            if latex and latex not in seen:
                seen.add(latex)
                results.append(latex)
        for child in (n.get("children") or []):
            _walk(child)

    _walk(node)
    return results


def _tree_contains_table(children: list[dict]) -> bool:
    def _walk(node: dict) -> bool:
        if node.get("block_type") in _TABLE_BLOCK_TYPES:
            return True
        return any(_walk(c) for c in (node.get("children") or []))
    return any(_walk(c) for c in children)