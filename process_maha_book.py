"""
process_maha_book.py
===============
Master Orchestrator for the Maharashtra State Board Textbook Dataset Pipeline.

Usage:
    python process_maha_book.py --input Maharashtra --output data/final
"""

import argparse
import json
import os
import sys
import re
import glob

# Ensure src package is importable
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.resources.resource_builder import ResourceBuilder
from src.resources import resource_types
from src.extraction.maha_parser import MahaParser

def find_text_json(chapter_dir: str) -> str | None:
    candidates = glob.glob(os.path.join(chapter_dir, "*.json"))
    text_files = [f for f in candidates if "image" not in os.path.basename(f).lower() and not os.path.basename(f).startswith(".")]
    return text_files[0] if text_files else None

def find_images_json(chapter_dir: str) -> str | None:
    candidates = glob.glob(os.path.join(chapter_dir, "*.json"))
    image_files = [f for f in candidates if "image" in os.path.basename(f).lower()]
    return image_files[0] if image_files else None

def run_chunker(text_json_path: str, chapter_no: int, subject: str, class_no: int) -> list[dict]:
    mp = MahaParser(
        subject=subject,
        source="MAHA",
        chapter_no=chapter_no,
        chapter_title=""
    )
    pages, _meta = mp.load_extraction_json(text_json_path)
    mp.chapter_title = mp.infer_chapter_title(pages)
    pages = mp.preprocess_pages(pages)
    chunks = mp.process_pages_to_chunks(pages)
    chunks = mp.merge_short_chunks(chunks)
    return chunks

def run_resource_builder(chunks: list[dict], raw_extraction: dict, document_id: str, class_no: int) -> list[dict]:
    for chunk in chunks:
        if "class_no" not in chunk:
            chunk["class_no"] = class_no
        if chunk.get("chapter_no") is None:
            chunk["chapter_no"] = 0

    builder = ResourceBuilder(document_id=document_id, raw_extraction=raw_extraction)
    resources = builder.build_from_chunks(chunks)
    return [r.to_dict() for r in resources]

def run_multimodal_merge(resources: list[dict], images: list[dict]) -> list[dict]:
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
            "document_id":   ctx.get("document_id"),
            "resource_id":   f"IMG_{c_no:02d}_CH{ch_no:02d}_{img_raw_id}",
            "resource_type": "diagram",
            "class_no":      c_no,
            "subject":       ctx.get("subject"),
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
            "figure_references": [],
            "has_equation":  False,
        })

    return resources + image_resources

def process_chapter(chapter_dir: str, output_dir: str) -> bool:
    # Path logic: Maharashtra/BIO/class11/chapter_01
    parts = Path(chapter_dir).parts
    if len(parts) < 4:
        return False
        
    subject_code = parts[-3].upper()
    class_str = parts[-2].lower()
    ch_str = parts[-1].lower()

    # Map subject
    sub_map = {"BIO": "Biology", "PHY": "Physics", "CHEM": "Chemistry"}
    subject = sub_map.get(subject_code, subject_code)
    
    class_no = int(re.sub(r"[^\d]", "", class_str) or 11)
    chapter_no = int(re.sub(r"[^\d]", "", ch_str) or 1)

    print(f"\n==========================================================")
    print(f"  CLASS {class_no} | {subject.upper()} | CH {chapter_no:02d} | {chapter_dir}")
    print(f"==========================================================")

    text_json = find_text_json(chapter_dir)
    images_json = find_images_json(chapter_dir)

    if not text_json:
        print(f"  [!] SKIPPED: No text JSON found in {chapter_dir}")
        return False

    print(f"  [*] Text JSON    : {os.path.basename(text_json)}")
    
    with open(text_json, "r", encoding="utf-8") as f:
        raw_extraction = json.load(f)

    images = []
    if images_json and os.path.exists(images_json):
        with open(images_json, "r", encoding="utf-8") as f:
            images = json.load(f)

    print(f"  [1/3] Chunking...")
    chunks = run_chunker(text_json, chapter_no, subject, class_no)
    print(f"        >> {len(chunks)} chunks produced")

    print(f"  [2/3] Building resources...")
    document_id = f"MAHA_{subject_code}_{class_no}"
    resources = run_resource_builder(chunks, raw_extraction, document_id, class_no)
    print(f"        >> {len(resources)} text resources compiled")

    print(f"  [3/3] Merging images...")
    final_dataset = run_multimodal_merge(resources, images)
    print(f"        >> {len(final_dataset) - len(resources)} diagram resources added")

    os.makedirs(output_dir, exist_ok=True)
    out_name = f"{subject_code.lower()}_{class_no}_ch{chapter_no:02d}.json"
    output_path = os.path.join(output_dir, out_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)

    print(f"  [+] Saved >> {output_path}")
    return True

from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="MAHA Textbook Full-Book Dataset Pipeline")
    parser.add_argument("--input", "-i", default="Maharashtra", help="Root folder (e.g. Maharashtra)")
    parser.add_argument("--output", "-o", default="data/final_maha", help="Output directory")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"[!] Directory '{args.input}' not found.")
        sys.exit(1)

    print(f"Scanning {args.input} for chapters...")
    chapter_dirs = []
    for root, dirs, files in os.walk(args.input):
        if any(f.endswith(".json") and "image" not in f.lower() for f in files):
            chapter_dirs.append(root)

    print(f"Found {len(chapter_dirs)} chapters to process.")
    
    success = 0
    for ch_dir in sorted(chapter_dirs):
        ok = process_chapter(ch_dir, args.output)
        if ok: success += 1

    print(f"\nPipeline complete! Successfully processed {success}/{len(chapter_dirs)} chapters.")

if __name__ == "__main__":
    main()
