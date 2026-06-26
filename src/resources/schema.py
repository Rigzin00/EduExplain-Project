"""
schema.py
=========
Defines the Resource dataclass — the single data contract shared across
dataset creation, retrieval, query generation, and evaluation.

Nothing downstream should construct resource dicts by hand.
Always go through Resource.to_dict() to serialise and
Resource.from_dict() to deserialise.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Resource:
    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    document_id: str          # e.g. "NCERT_SCI_11"
    resource_id: str          # e.g. "TXT_11_CH04_S4_2_007"

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    resource_type: str        # controlled — see resource_types.py

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------
    class_no: int
    subject: str
    chapter_no: int
    chapter_title: str
    section: Optional[str]
    subsection: Optional[str]
    page_start: int
    page_end: int

    # ------------------------------------------------------------------
    # Content
    # Text resources: text is populated, others are null.
    # Image resources: image_path, image_caption, description populated.
    # Table resources: description populated, image_path optionally set.
    # ------------------------------------------------------------------
    text: Optional[str] = None
    image_path: Optional[str] = None
    image_caption: Optional[str] = None
    description: Optional[str] = None

    # ------------------------------------------------------------------
    # Annotations (populated in later pipeline stages)
    # ------------------------------------------------------------------
    summary: Optional[str] = None
    concepts: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Structural flags (populated at build time, never later)
    # ------------------------------------------------------------------
    has_equation: bool = False

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Resource":
        return cls(
            document_id=d["document_id"],
            resource_id=d["resource_id"],
            resource_type=d["resource_type"],
            class_no=d["class_no"],
            subject=d["subject"],
            chapter_no=d["chapter_no"],
            chapter_title=d["chapter_title"],
            section=d.get("section"),
            subsection=d.get("subsection"),
            page_start=d["page_start"],
            page_end=d["page_end"],
            text=d.get("text"),
            image_path=d.get("image_path"),
            image_caption=d.get("image_caption"),
            description=d.get("description"),
            summary=d.get("summary"),
            concepts=d.get("concepts", []),
            has_equation=d.get("has_equation", False),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """
        Returns a list of validation error messages.
        Empty list means the resource is valid.
        Call after construction during testing; skip in production hot paths.
        """
        errors = []

        from src.resources.resource_types import VALID_RESOURCE_TYPES
        if self.resource_type not in VALID_RESOURCE_TYPES:
            errors.append(
                f"Invalid resource_type '{self.resource_type}'. "
                f"Must be one of: {sorted(VALID_RESOURCE_TYPES)}"
            )

        # Text resources must have text
        if self.resource_type not in ("diagram", "table"):
            if not self.text or not self.text.strip():
                errors.append(
                    f"resource_id={self.resource_id}: "
                    f"text resource has empty text"
                )

        # Image resources must have image_path
        if self.resource_type == "diagram":
            if not self.image_path:
                errors.append(
                    f"resource_id={self.resource_id}: "
                    f"diagram resource has no image_path"
                )

        if self.page_end < self.page_start:
            errors.append(
                f"resource_id={self.resource_id}: "
                f"page_end {self.page_end} < page_start {self.page_start}"
            )

        return errors