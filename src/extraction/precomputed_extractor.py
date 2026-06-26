"""
extraction/precomputed_extractor.py
=====================================
Loads a previously-serialized ExtractionResult JSON instead of running
any PDF library. Used when extraction was done on a remote machine (Colab,
HPC) and parsing runs locally.

Usage:
    extractor = ExtractorFactory.create(
        "precomputed",
        config={"result_path": "results/class11_ch01.json"}
    )
    result = extractor.extract("ignored.pdf")
"""
from __future__ import annotations
from typing import Optional
from .base import ExtractorBase, ExtractionResult


class PrecomputedExtractor(ExtractorBase):
    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self._result_path: str = self.config.get("result_path", "")

    def extract(self, pdf_path: str, **kwargs) -> ExtractionResult:
        if not self._result_path:
            raise ValueError(
                "PrecomputedExtractor requires config={'result_path': '...'}"
            )
        return ExtractionResult.from_json(self._result_path)