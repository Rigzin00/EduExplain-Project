"""
process_book.py
===============
Master Orchestrator for the NIOS Physics Textbook Dataset Pipeline.

For every chapter folder found inside the Physics/ input directory, this script:
  1. Auto-detects the raw text JSON and images JSON (handles inconsistent naming)
  2. Runs the NIOS parser (chunking)
  3. Runs the Resource Builder (schema validation + enrichment)
  4. Runs the Multimodal Merger (attaches images as standalone diagram resources)
  5. Saves a per-chapter multimodal JSON to data/final/

Usage:
    python process_book.py
    python process_book.py --input Physics --output data/final --class-no 12
"""

import argparse
import json
import os
import sys
import re
import glob

# ── Make sure the src package is importable ──────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.resources.resource_builder import ResourceBuilder
from src.resources import resource_types

# ── Register NIOS-specific chunk type mappings ────────────────────────────────
resource_types.CHUNK_TYPE_TO_RESOURCE_TYPE.update({
    "content":           "explanation",
    "terminal_exercise": "exercise",
    "answers":           "explanation",
})


# =============================================================================
# FILE AUTO-DETECTION
# =============================================================================

def find_text_json(chapter_dir: str) -> str | None:
    """Find the main text extraction JSON in a chapter folder.
    Prefers files NOT containing 'image' in their name."""
    candidates = glob.glob(os.path.join(chapter_dir, "*.json"))
    # Exclude image JSON files
    text_files = [f for f in candidates if "image" not in os.path.basename(f).lower()]
    if text_files:
        return text_files[0]   # Take the first match
    return None


def find_images_json(chapter_dir: str) -> str | None:
    """Find the images metadata JSON in a chapter folder."""
    candidates = glob.glob(os.path.join(chapter_dir, "*.json"))
    image_files = [f for f in candidates if "image" in os.path.basename(f).lower()]
    if image_files:
        return image_files[0]
    return None


# =============================================================================
# STAGE 1 — CHUNKING  (calls nios_parser functions directly)
# =============================================================================

def run_chunker(text_json_path: str, chapter_no: int) -> list[dict]:
    """Run the NIOS parser and return parsed chunks as a list of dicts."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "nios_parser", os.path.join(ROOT, "nios_parser.py")
    )
    nios_parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nios_parser)

    # Use the exact public API from nios_parser.py
    pages, raw_data = nios_parser.load_extraction_json(text_json_path)
    chapter_title   = nios_parser.infer_chapter_title(pages)

    chunks = nios_parser.parse_nios(
        pages,
        subject="Physics",
        source="NIOS",
        chapter_no=chapter_no,
        chapter_title=chapter_title,
    )
    return chunks


# =============================================================================
# STAGE 2 — RESOURCE BUILDER
# =============================================================================

def run_resource_builder(
    chunks: list[dict],
    raw_extraction: dict,
    document_id: str,
    class_no: int,
) -> list[dict]:
    """Convert parsed chunks to validated Resource dicts."""
    # Inject required fields that the NIOS parser doesn't embed
    for chunk in chunks:
        if "class_no" not in chunk:
            chunk["class_no"] = class_no
        if chunk.get("chapter_no") is None:
            chunk["chapter_no"] = 0

    builder = ResourceBuilder(document_id=document_id, raw_extraction=raw_extraction)
    resources = builder.build_from_chunks(chunks)
    return [r.to_dict() for r in resources]


# =============================================================================
# STAGE 3 — MULTIMODAL MERGE
# =============================================================================

def run_multimodal_merge(resources: list[dict], images: list[dict]) -> list[dict]:
    """Attach images as independent 'diagram' Resources next to text resources."""
    # Build page → context map from text resources
    context_map: dict[int, dict] = {}
    for res in resources:
        for p in range(res.get("page_start", 0), res.get("page_end", 0) + 1):
            if p not in context_map:
                context_map[p] = res

    image_resources = []
    for img in images:
        img_page = img.get("page_no")
        ctx = context_map.get(img_page, resources[0] if resources else {})

        img_raw_id = re.sub(r"[^0-9]", "", img.get("image_id", "000")).zfill(3)
        c_no  = ctx.get("class_no", 12)
        ch_no = ctx.get("chapter_no", 0)

        image_resources.append({
            "document_id":   ctx.get("document_id", "NIOS_PHY"),
            "resource_id":   f"IMG_{c_no:02d}_CH{ch_no:02d}_{img_raw_id}",
            "resource_type": "diagram",
            "class_no":      c_no,
            "subject":       ctx.get("subject", "Physics"),
            "chapter_no":    ch_no,
            "chapter_title": ctx.get("chapter_title", ""),
            "section":       ctx.get("section"),
            "subsection":    ctx.get("subsection"),
            "page_start":    img_page,
            "page_end":      img_page,
            "text":          None,
            "image_path":    img.get("image_path"),
            "image_caption": img.get("caption"),
            "description":   None,
            "summary":       None,
            "concepts":      [],
            "has_equation":  False,
        })

    return resources + image_resources


# =============================================================================
# CHAPTER PROCESSOR
# =============================================================================

def process_chapter(chapter_dir: str, chapter_no: int, output_dir: str, class_no: int) -> bool:
    """Run the full pipeline for a single chapter. Returns True on success."""
    sep = "=" * 58
    print(f"\n{sep}")
    print(f"  CHAPTER {chapter_no:02d}  |  {chapter_dir}")
    print(sep)

    # ── Detect files ──────────────────────────────────────────────────────────
    text_json   = find_text_json(chapter_dir)
    images_json = find_images_json(chapter_dir)

    if not text_json:
        print(f"  [!] SKIPPED: No text JSON found in {chapter_dir}")
        return False

    print(f"  [*] Text JSON    : {os.path.basename(text_json)}")
    print(f"  [*] Images JSON  : {os.path.basename(images_json) if images_json else 'NOT FOUND (no diagrams)'}")

    # ── Load raw extraction (for equation detection) ──────────────────────────
    with open(text_json, "r", encoding="utf-8") as f:
        raw_extraction = json.load(f)

    # ── Load images ───────────────────────────────────────────────────────────
    images = []
    if images_json and os.path.exists(images_json):
        with open(images_json, "r", encoding="utf-8") as f:
            images = json.load(f)

    # ── Stage 1: Chunking ─────────────────────────────────────────────────────
    print(f"  [1/3] Chunking...")
    try:
        chunks = run_chunker(text_json, chapter_no=chapter_no)
    except Exception as e:
        print(f"  [!] Chunking FAILED: {e}")
        return False
    print(f"        >> {len(chunks)} chunks produced")

    # ── Stage 2: Resource Builder ─────────────────────────────────────────────
    print(f"  [2/3] Building resources...")
    document_id = f"NIOS_PHY_CH{chapter_no:02d}"
    try:
        resources = run_resource_builder(chunks, raw_extraction, document_id, class_no)
    except Exception as e:
        print(f"  [!] Resource Builder FAILED: {e}")
        return False
    print(f"        >> {len(resources)} text resources compiled")

    # ── Stage 3: Multimodal Merge ─────────────────────────────────────────────
    print(f"  [3/3] Merging images...")
    final_dataset = run_multimodal_merge(resources, images)
    diagram_count = len(final_dataset) - len(resources)
    print(f"        >> {diagram_count} diagram resources added")
    print(f"        >> {len(final_dataset)} total records in final dataset")

    # ── Save output ───────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"ch{chapter_no:02d}_multimodal.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)

    print(f"  [+] Saved >> {output_path}")
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NIOS Physics Full-Book Dataset Pipeline")
    parser.add_argument("--input",    "-i", default="Physics",   help="Root folder containing 'chapter N' subdirectories")
    parser.add_argument("--output",   "-o", default="data/final", help="Output directory for per-chapter multimodal JSONs")
    parser.add_argument("--class-no", "-c", type=int, default=12, help="Class number to embed in every resource")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"[!] Input directory '{args.input}' not found. Exiting.")
        sys.exit(1)

    # Discover and sort chapter directories
    chapter_dirs = sorted(
        [d for d in os.listdir(args.input) if re.match(r"chapter\s+\d+$", d, re.IGNORECASE)],
        key=lambda d: int(re.search(r"\d+", d).group())
    )

    if not chapter_dirs:
        print(f"[!] No 'chapter N' folders found inside '{args.input}'. Exiting.")
        sys.exit(1)

    print(f"\n{'#' * 60}")
    print(f"  NIOS PHYSICS FULL-BOOK PIPELINE")
    print(f"  Found {len(chapter_dirs)} chapter(s) to process")
    print(f"  Output >> {os.path.abspath(args.output)}")
    print(f"{'#' * 60}")

    success, failed = [], []
    for ch_dir_name in chapter_dirs:
        chapter_no = int(re.search(r"\d+", ch_dir_name).group())
        chapter_dir = os.path.join(args.input, ch_dir_name)
        ok = process_chapter(chapter_dir, chapter_no, args.output, args.class_no)
        (success if ok else failed).append(chapter_no)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'#' * 60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  [OK]  Success : {len(success)} chapters >> {success}")
    if failed:
        print(f"  [FAIL] Failed : {len(failed)} chapters >> {failed}")
    print(f"  Output saved to: {os.path.abspath(args.output)}/")
    print(f"{'#' * 60}\n")


if __name__ == "__main__":
    main()
