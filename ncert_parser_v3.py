

import re
import json
import argparse
import os
from pathlib import Path
from typing import Optional

from cv2 import line

from src.extraction.factory import ExtractorFactory
from src.extraction.base import ExtractionResult


# ---------------------------------------------------------------------------
# Text cleaning helpers  (unchanged from v1)
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize extracted PDF text."""
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


def is_page_number_line(line: str) -> bool:
    return bool(re.match(r'^\s*\d{1,3}\s*$', line.strip()))


def remove_header_footer(lines: list[str]) -> list[str]:
    skip_patterns = [
        r'^Chemical Reactions and Equations\s*\d*$',
        r'^Science\s*$',
        r'^\d+\s+Science\s*$',
        r'^Science\s+\d+\s*$',
        r'^Reprint \d{4}',
        r'^Chapter-\d+\.indd',
        r'^Curiosity.*Grade \d+',
        r'^Exploration\|Grade \d+',
        r'^Chapter \d+.*World of Science',
        r'^\s*\d{1,3}\s*$',
    ]
    cleaned = []
    for line in lines:
        if not any(re.match(p, line.strip(), re.IGNORECASE) for p in skip_patterns):
            cleaned.append(line)
    return cleaned


# ---------------------------------------------------------------------------
# Section detection  (unchanged from v1)
# ---------------------------------------------------------------------------

SECTION_RE = re.compile(r'^(\d+\.\d+(?:\.\d+)?)\s+(.+?)\s*$')


def detect_numbered_section(line: str):
    line = line.strip()
    m = SECTION_RE.match(line)
    if m:
        no = m.group(1)
        title = " ".join(w.capitalize() for w in m.group(2).strip().split())
        return no, title
    return None


def count_dots(s: str) -> int:
    return s.count('.')


CLASS89_HEADING_PATTERNS = [
    r'^[A-Z][A-Z\s\(\)\-\/]{4,60}$',
    r'^[A-Z][a-zA-Z\s\(\)\-\/]{4,60}$',
]
CLASS89_HEADING_EXCLUSIONS = [
    r'^Dear Young Scientists', r'^Happy', r'^Write it here',
    r'^Probe and ponder', r'^Pause and Ponder', r'^Ready to Go Beyond',
    r'^Meet a Scientist', r'^Threads of Curiosity', r'^Next Level Up',
    r'^Activity \d+', r'^Example \d+', r'^Answer',
]


def is_class89_heading(line: str, prev_blank: bool, next_blank: bool) -> bool:
    line = line.strip()
    if len(line) < 3 or len(line) > 80:
        return False
    if not (prev_blank or next_blank):
        return False
    for excl in CLASS89_HEADING_EXCLUSIONS:
        if re.match(excl, line, re.IGNORECASE):
            return False
    for pat in CLASS89_HEADING_PATTERNS:
        if re.match(pat, line):
            return True
    return False


# ---------------------------------------------------------------------------
# Chunk type classifiers  
# ---------------------------------------------------------------------------


EXAMPLE_START_RE = re.compile(    
    r'^\s*.*?Example\s+\d+(?:\.\d+)*',
    re.IGNORECASE
)

ACTIVITY_START_RE = re.compile(
    r'^Activity\s+\d+[\.\d]*|^ACTIVITY\s+\d+[\.\d]*', re.IGNORECASE
)
EXERCISES_HEADER_RE = re.compile(
    r'^EXERCISES?\s*$', re.IGNORECASE
)
INTEXT_QUESTION_RE = re.compile(
    r'^(?:IN[\s\-]?TEXT\s+QUESTIONS?|QUESTIONS?)\s*$', re.IGNORECASE
)
SUMMARY_HEADER_RE = re.compile(
    r'^(?:SUMMARY|WHAT\s+YOU\s+HAVE\s+LEARNT|KEY\s+POINTS)\s*$',
    re.IGNORECASE
)
EXERCISE_QUESTION_RE = re.compile(
    r'^\s*\d+\.\d+\b'
)
EXERCISE_QUESTION_SIMPLE_RE = re.compile(
    r'^\d+[\.\)]\s+[A-Z]'      # "1. A body..." fallback
)

def is_example_heading(line):
    return bool(EXAMPLE_START_RE.match(line.strip()))

def is_activity_heading(line):
    return bool(ACTIVITY_START_RE.match(line.strip()))

def is_exercise_heading(line):
    return bool(EXERCISES_HEADER_RE.match(line.strip()))

def is_intext_question_heading(line):
    return bool(INTEXT_QUESTION_RE.match(line.strip()))

def is_summary_heading(line):
    return bool(SUMMARY_HEADER_RE.match(line.strip()))





# ---------------------------------------------------------------------------
# Page-to-line flattening  (unchanged from v1)
# ---------------------------------------------------------------------------

def pages_to_lines(pages: list[dict]) -> list[dict]:
    lines = []
    for page in pages:
        for line in page["text"].split('\n'):
            lines.append({"page_no": page["page_no"], "text": line})
    return lines


# ---------------------------------------------------------------------------
# Parsers  (unchanged from v1 — they receive flat page dicts as before)
# ---------------------------------------------------------------------------

import re

# ── Boundary detection patterns ───────────────────────────────────────




def parse_numbered_ncert(pages, class_no, chapter_no, chapter_title):
    lines = pages_to_lines(pages)
    lines = [l for l in lines if not is_page_number_line(l["text"])]
    #for testing
    #for line_obj in lines:
    #    line = line_obj["text"]
    #    if "Example" in line:
    #        print(repr(line))
    
    

    chunks = []
    chunk_id_counter = [0]

    # ── Metadata state (unchanged from your current parser) ───────────
    current_section_no    = None
    current_section_title = None
    current_subsection_no = None
    current_subsection_title = None

    # ── Accumulator state ─────────────────────────────────────────────
    current_lines      = []
    current_page_start = pages[0]["page_no"] if pages else 1
    current_page_end   = current_page_start

    # ── Parser mode state machine ─────────────────────────────────────
    # Possible modes:
    #   "explanation"     — default, between semantic blocks
    #   "example"         — inside an Example N.N block
    #   "activity"        — inside an Activity N.N block
    #   "exercise"        — inside the back-of-chapter exercises
    #   "intext_question" — inside an in-text question block
    #   "summary"         — inside the chapter summary
    current_mode = "explanation"

    # ── flush_chunk ───────────────────────────────────────────────────
    def flush_chunk():
        nonlocal current_page_start

        text = clean_text('\n'.join(l["text"] for l in current_lines))
        if len(text.strip()) < 30:
            current_lines.clear()
            return

        # Type is determined by mode, not by text scanning —
        # except for "explanation" mode where we fall back to
        chunk_type = current_mode
        #print("FLUSHING:", current_mode, "LEN=", len(text))

        chunk_id_counter[0] += 1
        sec_str = (current_section_no or "intro").replace(".", "_")
        chunk_id = (
            f"sci_{class_no:02d}_ch{chapter_no:02d}"
            f"_s{sec_str}_{chunk_id_counter[0]}"
        )

        chunks.append({
            "chunk_id":       chunk_id,
            "class_no":       class_no,
            "subject":        "Science",
            "chapter_no":     chapter_no,
            "chapter_title":  chapter_title,
            "section":        current_section_title,
            "section_no":     current_section_no,
            "subsection":     current_subsection_title,
            "subsection_no":  current_subsection_no,
            "chunk_type":     chunk_type,
            "text":           text,
            "page_start":     current_page_start,
            "page_end":       current_page_end,
        })
        current_lines.clear()
        current_page_start = current_page_end
        #print("FLUSHING:", current_mode, "LEN=", len(text))

    # ── Main loop ─────────────────────────────────────────────────────
    for line_obj in lines:
        line     = line_obj["text"].strip()
        page_no  = line_obj["page_no"]
        current_page_end = page_no

        # ── Priority 1: structural boundaries ─────────────────────────
        # These always flush, regardless of current mode.

        detected_section = None

        if current_mode != "exercise":
            detected_section = detect_numbered_section(line)
        #remove
        if detected_section:
            print("SECTION DETECTED:", repr(line))
        if detected_section:
            flush_chunk()
            current_mode = "explanation"
            no, title = detected_section
            if count_dots(no) == 1:
                current_section_no    = no
                current_section_title = title
                current_subsection_no    = None
                current_subsection_title = None
            elif count_dots(no) == 2:
                current_subsection_no    = no
                current_subsection_title = title
            current_page_start = page_no
            # Do not append the heading line itself — it's metadata
            continue
        #remember to remove
        
        # ── Priority 2: semantic block starts ──────────────────────────
        # These flush whatever came before, then start a new typed block.

        if is_summary_heading(line):
            flush_chunk()
            current_mode = "summary"
            current_page_start = page_no
            # Don't append the header line — it's structural noise
            continue

        if is_exercise_heading(line):
            flush_chunk()
            current_mode = "exercise"
            current_page_start = page_no
            continue

        if is_intext_question_heading(line):
            flush_chunk()
            current_mode = "intext_question"
            current_page_start = page_no
            continue

        if is_example_heading(line) and current_mode != "exercise":
            #print("FOUND EXAMPLE:", line)
            flush_chunk()
            current_mode = "example"
            current_page_start = page_no
            current_lines.append(line_obj)  # keep the "Example N.N" line
            continue

        if is_activity_heading(line) and current_mode != "exercise":
            flush_chunk()
            current_mode = "activity"
            current_page_start = page_no
            current_lines.append(line_obj)  # keep the "Activity N.N" line
            continue

        # ── Priority 3: within-exercise question boundaries ────────────
        # Only active when already in exercise mode.
        # Each question becomes its own resource.

        if current_mode == "exercise":
            is_question = (
                EXERCISE_QUESTION_RE.match(line) or
                EXERCISE_QUESTION_SIMPLE_RE.match(line)
            )
            if is_question and current_lines:
                flush_chunk()
                current_page_start = page_no
        
        
        # ── Default: accumulate ────────────────────────────────────────
        current_lines.append(line_obj)

    # Flush whatever remains at end of chapter
    flush_chunk()

    #chunks = merge_short_chunks(chunks, min_chars=30)
    return chunks


def parse_class89(pages: list[dict], class_no: int, chapter_no: int,
                  chapter_title: str) -> list[dict]:
    lines = pages_to_lines(pages)
    raw_texts = [l["text"] for l in lines]
    cleaned_texts = remove_header_footer(raw_texts)
    cleaned_lines = [
        {"page_no": lines[i]["page_no"], "text": cleaned_texts[i]}
        for i in range(len(cleaned_texts))
    ]

    chunks = []
    chunk_id_counter = [0]
    current_heading = None
    current_lines = []
    current_page_start = pages[0]["page_no"] if pages else 1
    current_page_end = current_page_start

    def make_chunk_id():
        chunk_id_counter[0] += 1
        heading_slug = re.sub(r'\W+', '_', (current_heading or "intro"))[:30].lower()
        return f"sci_{class_no:02d}_ch{chapter_no:02d}_{heading_slug}_{chunk_id_counter[0]}"

    def flush_chunk():
        text = clean_text('\n'.join(l["text"] for l in current_lines))
        if len(text.strip()) < 30:
            return
        chunk_type = classify_chunk_type(current_heading or "", text)
        chunks.append({
            "chunk_id": make_chunk_id(),
            "class_no": class_no,
            "subject": "Science",
            "chapter_no": chapter_no,
            "chapter_title": chapter_title,
            "section": current_heading,
            "section_no": None,
            "subsection": None,
            "subsection_no": None,
            "chunk_type": chunk_type,
            "text": text,
            "page_start": current_page_start,
            "page_end": current_page_end,
        })
        current_lines.clear()

    n = len(cleaned_lines)
    for i, line_obj in enumerate(cleaned_lines):
        line = line_obj["text"]
        page_no = line_obj["page_no"]
        prev_blank = (i == 0) or (cleaned_lines[i - 1]["text"].strip() == "")
        next_blank = (i == n - 1) or (cleaned_lines[i + 1]["text"].strip() == "")
        if is_class89_heading(line, prev_blank, next_blank):
            flush_chunk()
            current_heading = line.strip()
            current_page_start = page_no
            current_page_end = page_no
        else:
            current_lines.append(line_obj)
            current_page_end = page_no

    flush_chunk()
    return chunks


# ---------------------------------------------------------------------------
# Dispatcher  — the only section that changed from v1
# ---------------------------------------------------------------------------

def parse_chapter(
    pdf_path: str,
    class_no: int,
    chapter_no: int,
    chapter_title: str,
    backend: str = "pdfplumber",
    extractor_config: Optional[dict] = None,
) -> list[dict]:
    """
    Main entry point.  Dispatches extraction to the chosen backend,
    then dispatches parsing to the correct structural parser.

    Parameters
    ----------
    backend : str
        "pdfplumber" (default) or "marker" or any registered backend name.
        Change this one string to switch the entire extraction layer.
    extractor_config : dict, optional
        Passed through to the backend's constructor.
    """
    extractor = ExtractorFactory.create(backend, config=extractor_config)

    # pdfplumber needs class_no to decide column strategy.
    # Marker resolves layout automatically and ignores this kwarg.
    result: ExtractionResult = extractor.extract(pdf_path, class_no=class_no)

    if not result.pages:
        print(f"  [WARN] No text extracted from {pdf_path}")
        return []

    # as_flat_pages() converts ExtractionResult → list[{page_no, text}]
    # This is the compatibility shim — the parsers below never changed.
    flat_pages = result.as_flat_pages()

    if class_no in (10, 11, 12):
        chunks = parse_numbered_ncert(flat_pages, class_no, chapter_no, chapter_title)
    elif class_no in (8, 9):
        chunks = parse_class89(flat_pages, class_no, chapter_no, chapter_title)
    else:
        raise ValueError(f"Unsupported class: {class_no}")

    chunks = merge_short_chunks(chunks, min_chars=80)
    return chunks


def merge_short_chunks(chunks: list[dict], min_chars: int = 80) -> list[dict]:
    """Merge chunks that are too short into the preceding chunk."""
    if not chunks:
        return chunks
    merged = [chunks[0]]
    for chunk in chunks[1:]:
        if len(chunk["text"]) < min_chars and merged:
            merged[-1]["text"] += "\n\n" + chunk["text"]
            merged[-1]["page_end"] = chunk["page_end"]
        else:
            merged.append(chunk)
    return merged


# ---------------------------------------------------------------------------
# CLI  (unchanged from v1 — one new optional --backend flag added)
# ---------------------------------------------------------------------------

def save_chunks(chunks: list[dict], output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(chunks)} chunks → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse NCERT Science PDFs into structured JSON chunks."
    )
    parser.add_argument("--pdf", help="Path to a single chapter PDF")
    parser.add_argument("--class_no", type=int)
    parser.add_argument("--chapter_no", type=int)
    parser.add_argument("--chapter_title")
    parser.add_argument("--output", default="output/")
    parser.add_argument(
        "--backend", default="pdfplumber",
        choices=ExtractorFactory.available_backends(),
        help="Extraction backend to use (default: pdfplumber)",
    )
    parser.add_argument(
    "--result_path",
    help="Path to precomputed extraction JSON"
    )

    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--manifest")
    parser.add_argument("--input_dir", default=".")
    parser.add_argument("--output_dir", default="output/")

    args = parser.parse_args()

    if args.batch:
        if not args.manifest:
            print("ERROR: --manifest required for batch mode")
            return
        with open(args.manifest) as f:
            manifest = json.load(f)

        all_chunks = []
        for entry in manifest:
            pdf_path = os.path.join(args.input_dir, entry["file"])
            class_no = entry["class_no"]
            chapter_no = entry["chapter_no"]
            chapter_title = entry["chapter_title"]

            print(f"\nProcessing: Class {class_no} Ch{chapter_no} — {chapter_title}")
            if not os.path.exists(pdf_path):
                print(f"  [SKIP] File not found: {pdf_path}")
                continue

            chunks = parse_chapter(
                pdf_path, class_no, chapter_no, chapter_title,
                backend=args.backend,
            )
            all_chunks.extend(chunks)

            out_filename = f"sci_class{class_no}_ch{chapter_no:02d}.json"
            out_path = os.path.join(args.output_dir, f"class{class_no}", out_filename)
            save_chunks(chunks, out_path)

        save_chunks(all_chunks, os.path.join(args.output_dir, "all_chunks.json"))
        print(f"\nDone. Total chunks: {len(all_chunks)}")

    else:
        if not all([args.pdf, args.class_no, args.chapter_no, args.chapter_title]):
            print("ERROR: --pdf, --class_no, --chapter_no, --chapter_title all required")
            parser.print_help()
            return

        print(f"Parsing: Class {args.class_no} Ch{args.chapter_no} — {args.chapter_title}")
        print(f"  File: {args.pdf}  Backend: {args.backend}")

        extractor_config = None

        if args.backend == "precomputed":
            extractor_config = {
                "result_path": args.result_path
            }
        chunks = parse_chapter(
            args.pdf, args.class_no, args.chapter_no, args.chapter_title,
            backend=args.backend,extractor_config=extractor_config,
        )

        out_filename = f"sci_class{args.class_no}_ch{args.chapter_no:02d}.json"
        out_path = os.path.join(args.output, out_filename)
        save_chunks(chunks, out_path)

        print("\n--- Preview (first 3 chunks) ---")
        for chunk in chunks[:3]:
            print(json.dumps(
                {k: v for k, v in chunk.items() if k != "text"}, indent=2
            ))
            print(f"  text preview: {chunk['text'][:120]}...")
            print()


if __name__ == "__main__":
    main()


