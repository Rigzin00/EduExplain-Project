"""
resource_builder.py
====================
Transforms parsed chunk dicts (from parsed_extraction/) into Resource
objects conforming to the benchmark schema.

Design principles:
  - One builder class with a narrow public interface: build_from_chunks()
  - Image and table resources are added via register_handler(), not by
    modifying this file (see question 4 answer below)
  - No retrieval logic, no annotation logic, no IO logic in this file

Usage:
    from resource_builder import ResourceBuilder

    builder = ResourceBuilder(
        document_id="NCERT_SCI_11",
        raw_extraction=raw_extraction_dict,   # from raw_marker_extraction/
    )
    resources = builder.build_from_chunks(parsed_chunks)

    # Serialise
    import json
    with open("resources/NCERT_SCI_11_CH04.json", "w") as f:
        json.dump([r.to_dict() for r in resources], f, indent=2, ensure_ascii=False)
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from src.resources.schema import Resource
from src.resources.resource_types import (
    map_chunk_type,
    get_id_prefix,
    is_text_resource,
)


# ------------------------------------------------------------------
# Type alias for resource handler functions
# A handler takes a raw chunk dict and a ResourceBuilder instance
# and returns a Resource (or None to skip).
# This is the extension point for image and table resources.
# ------------------------------------------------------------------
HandlerFn = Callable[[dict, "ResourceBuilder"], Optional[Resource]]


class ResourceBuilder:
    """
    Converts parsed chunk dicts → Resource objects.

    Parameters
    ----------
    document_id : str
        Canonical textbook identifier, e.g. "NCERT_SCI_11".
        Used as the document_id field on every emitted resource.

    raw_extraction : dict
        The full extraction JSON from raw_marker_extraction/.
        Used to look up per-page equation lists for has_equation detection.
        Pass an empty dict if you do not have extraction data available.
    """

    def __init__(
        self,
        document_id: str,
        raw_extraction: Optional[dict] = None,
    ):
        self.document_id = document_id
        self._equations_by_page: dict[int, list] = {}

        if raw_extraction:
            for page in raw_extraction.get("pages", []):
                pg = page.get("page_no")
                eqs = page.get("equations", [])
                if pg is not None:
                    self._equations_by_page[pg] = eqs

        # Registry for custom handlers (image, table, etc.)
        # Maps resource_type string → handler function
        self._handlers: dict[str, HandlerFn] = {}

        # Counter per (chapter, section) for unique ID generation
        self._id_counters: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_from_chunks(self, chunks: list[dict]) -> list[Resource]:
        """
        Convert a list of parsed chunk dicts to Resource objects.
        Chunks that fail validation are logged and skipped.

        Parameters
        ----------
        chunks : list[dict]
            Output of ncert_parser_v2.py — list of chunk dicts.

        Returns
        -------
        list[Resource]
            One Resource per valid chunk.
        """
        resources = []
        for chunk in chunks:
            resource = self._build_one(chunk)
            if resource is None:
                continue
            errors = resource.validate()
            if errors:
                print(f"[SKIP] {resource.resource_id}: {errors}")
                continue
            resources.append(resource)
        return resources

    def register_handler(self, resource_type: str, handler: HandlerFn) -> None:
        """
        Register a custom handler for a resource type.

        When a chunk maps to resource_type, the registered handler is called
        instead of the default _build_text_resource() path.

        This is the extension point for image and table resources — you do
        not need to modify this file to add them.

        Example:
            def image_handler(chunk, builder):
                return Resource(
                    document_id=builder.document_id,
                    resource_id=builder.make_resource_id(chunk, "diagram"),
                    resource_type="diagram",
                    ...
                )
            builder.register_handler("diagram", image_handler)
        """
        self._handlers[resource_type] = handler

    # ------------------------------------------------------------------
    # Core transformation
    # ------------------------------------------------------------------

    def _build_one(self, chunk: dict) -> Optional[Resource]:
        resource_type = map_chunk_type(chunk.get("chunk_type", ""))

        # Delegate to a registered handler if one exists for this type
        if resource_type in self._handlers:
            return self._handlers[resource_type](chunk, self)

        # Default path: text resource
        if is_text_resource(resource_type):
            return self._build_text_resource(chunk, resource_type)

        # Non-text type with no handler registered — skip for now
        print(
            f"[SKIP] chunk_type='{chunk.get('chunk_type')}' maps to "
            f"resource_type='{resource_type}' but no handler is registered. "
            f"Register one via builder.register_handler('{resource_type}', fn)."
        )
        return None

    def _build_text_resource(
        self, chunk: dict, resource_type: str
    ) -> Resource:
        return Resource(
            document_id=self.document_id,
            resource_id=self.make_resource_id(chunk, resource_type),
            resource_type=resource_type,
            class_no=chunk["class_no"],
            subject=chunk["subject"],
            chapter_no=chunk["chapter_no"],
            chapter_title=chunk["chapter_title"],
            section=chunk.get("section"),
            subsection=chunk.get("subsection"),
            page_start=chunk["page_start"],
            page_end=chunk["page_end"],
            text=chunk.get("text"),
            image_path=None,
            image_caption=None,
            description=None,
            summary=None,
            concepts=[],
            has_equation=self._detect_has_equation(chunk),
        )

    # ------------------------------------------------------------------
    # Resource ID generation
    # Public so that custom handlers can call it
    # ------------------------------------------------------------------

    def make_resource_id(self, chunk: dict, resource_type: str) -> str:
        """
        Generate a unique, human-readable resource ID.

        Format: {PREFIX}_{class}_{CH}{chapter}_{section}_{index:03d}
        Example: TXT_11_CH04_S4_2_007
        """
        prefix = get_id_prefix(resource_type)
        class_no = chunk.get("class_no", 0)
        chapter_no = chunk.get("chapter_no", 0)

        # Normalise section_no: "4.2" → "S4_2", None → "SX"
        section_no = chunk.get("section_no") or chunk.get("section") or "X"
        section_slug = re.sub(r"[^a-zA-Z0-9]", "_", str(section_no)).strip("_")
        section_slug = f"S{section_slug}" if section_slug[0].isdigit() else section_slug

        # Counter key ensures uniqueness within section
        counter_key = f"{class_no}_{chapter_no}_{section_slug}_{prefix}"
        self._id_counters[counter_key] = self._id_counters.get(counter_key, 0) + 1
        idx = self._id_counters[counter_key]

        return f"{prefix}_{class_no:02d}_CH{chapter_no:02d}_{section_slug}_{idx:03d}"

    # ------------------------------------------------------------------
    # has_equation detection
    # ------------------------------------------------------------------

    def _detect_has_equation(self, chunk: dict) -> bool:
        """
        A resource has an equation if:
        (a) the raw extraction reported equations on any page in the chunk's
            page range, OR
        (b) the chunk text contains LaTeX-like patterns.

        (a) is preferred when raw extraction data is available.
        (b) is a fallback for chunks built without extraction data.
        """
        
        page_start = chunk.get("page_start", 0)
        page_end = chunk.get("page_end", page_start)

        # (a) Check extraction equations by page
        if self._equations_by_page:
            for page_no in range(page_start, page_end + 1):
                if self._equations_by_page.get(page_no):
                    return True
            return False

        # (b) Regex fallback
        text = chunk.get("text", "") or ""
        return bool(re.search(
            r"\$.*?\$|\\frac|\\sum|\\int|\\sqrt|=\s*\d", text
        ))
