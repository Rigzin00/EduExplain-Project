"""
nios_parser.py
~~~~~~~~~~~~~~
Parses precomputed Marker extraction JSONs (like Phy_NIOS.json) into
structured text chunks suitable for LLM training / RAG datasets.

Adapted from ncert_parser_v3.py to work with your local resource files
without needing a live PDF or Marker installation.

Usage:
    python nios_parser.py --input Phy_NIOS.json --output output/Phy_NIOS_chunks.json
    python nios_parser.py --input Phy_NIOS.json --output output/Phy_NIOS_chunks.json --subject Physics --source NIOS
"""

import re
import json
import argparse
import os
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

# NIOS-specific running headers that pollute chunk text
# (compiled once for efficiency)
_RUNNING_HEADER_RES = [
    re.compile(r'^MODULE\s*-\s*\d+\s*$', re.MULTILINE),
    re.compile(r'^MODULE\s*-\s*\d+\s+.+$', re.MULTILINE),   # "MODULE - 1 Units..."
    re.compile(r'^Motion,\s+Force\s+and\s+Energy\s*$', re.MULTILINE),
    re.compile(r'^Units,\s+Dimensions\s+and\s+Vectors\s*$', re.MULTILINE),
    re.compile(r'^Units\.\s+Dimensions\s+and\s+Vectors\s*$', re.MULTILINE),
]


def clean_text(text: str) -> str:
    """Normalize extracted text from Marker JSON."""
    if not text:
        return ""
    # Remove repeated characters (OCR artifacts)
    text = re.sub(r'(.)\1{3,}', lambda m: m.group(1), text)
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    # Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip NIOS running headers (module/chapter banners on each page)
    for pat in _RUNNING_HEADER_RES:
        text = pat.sub('', text)
    # Remove bare page numbers
    text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)
    # Clean up any extra blank lines left after removals
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def is_page_number_line(line: str) -> bool:
    return bool(re.match(r'^\s*\d{1,3}\s*$', line.strip()))


# ---------------------------------------------------------------------------
# Section / heading detection  (handles NIOS-style numbered sections)
# ---------------------------------------------------------------------------

# Matches: "1.1 PHYSICAL WORLD", "1.2.3 Significant Figures", "1.2.4. Derived Units"
# Requires:
#   - section number: 1-2 digits per part, optionally trailing dot (e.g. "1.2.4.")
#   - title: must begin with a letter (rejects raw numbers/symbols in sentences)
SECTION_RE = re.compile(r'^(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?\.?)\s+([A-Za-z].+?)\s*$')

# Matches chapter-level headings like "UNITS, DIMENSIONS AND VECTORS"
CHAPTER_HEADING_RE = re.compile(r'^[A-Z][A-Z0-9\s,\(\)\-\/]{5,80}$')

# Markers for special block types
INTEXT_QUESTION_RE = re.compile(
    r'^INTEXT\s+QUESTIONS?\s*[\d\.]*\s*$', re.IGNORECASE
)
TERMINAL_EXERCISE_RE = re.compile(
    r'^TERMINAL\s+EXERCISE\s*$', re.IGNORECASE
)
WHAT_YOU_LEARNED_RE = re.compile(
    r'^WHAT\s+YOU\s+HAVE\s+LEARNT\s*$', re.IGNORECASE
)
EXAMPLE_RE = re.compile(
    r'^Example\s+\d+[\.\d]*', re.IGNORECASE
)
OBJECTIVES_RE = re.compile(
    r'^OBJECTIVES\s*$', re.IGNORECASE
)
ANSWER_SECTION_RE = re.compile(
    r'^ANSWERS?\s+TO\s+', re.IGNORECASE
)


def detect_numbered_section(line: str) -> Optional[tuple]:
    """Return (section_number, title) if line is a numbered section heading."""
    line = line.strip()
    m = SECTION_RE.match(line)
    if m:
        no = m.group(1).rstrip('.')
        title = m.group(2).strip()
        return no, title
    return None


def count_dots(s: str) -> int:
    return s.count('.')


# ---------------------------------------------------------------------------
# Load precomputed JSON
# ---------------------------------------------------------------------------

def load_extraction_json(json_path: str) -> tuple[list[dict], dict]:
    """
    Load a Marker precomputed extraction JSON.
    Returns (pages, metadata) where metadata carries source_pdf, has_equations, etc.
    """
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    pages = data.get("pages", [])
    if not pages:
        raise ValueError(f"No pages found in {json_path}")

    print(f"  Loaded {len(pages)} pages from {json_path}")
    print(f"  Backend: {data.get('extraction_backend', 'unknown')}")
    print(f"  Has equations: {data.get('has_equations', False)}")
    return pages, data


def infer_chapter_title(pages: list[dict]) -> str:
    """
    Try to auto-detect the chapter title from the first page of text.
    Looks for the first all-caps line of 5+ words that is clearly a chapter heading.
    Falls back to an empty string if nothing is found.
    """
    if not pages:
        return ""
    first_text = pages[0].get("text", "")
    for raw_line in first_text.split("\n"):
        line = raw_line.strip()
        # Must be entirely uppercase letters, spaces, commas, hyphens
        if re.match(r'^[A-Z][A-Z0-9\s,\(\)\-\/]{8,80}$', line):
            # Skip generic module banners
            if re.match(r'^MODULE\s*-', line):
                continue
            return line.title()   # e.g. "UNITS, DIMENSIONS AND VECTORS" → title case
    return ""


def pages_to_lines(pages: list[dict]) -> list[dict]:
    """Flatten pages into a list of line dicts: {page_no, text}."""
    lines = []
    for page in pages:
        for line in page["text"].split('\n'):
            lines.append({"page_no": page["page_no"], "text": line})
    return lines


def get_equations_by_page(pages: list[dict]) -> dict[int, list[str]]:
    """Build a map of page_no → list of LaTeX equation strings."""
    eq_map = {}
    for page in pages:
        eqs = page.get("equations", [])
        if eqs:
            eq_map[page["page_no"]] = eqs
    return eq_map


# ---------------------------------------------------------------------------
# Core parser — handles NIOS-style numbered sections
# ---------------------------------------------------------------------------

def parse_nios(
    pages: list[dict],
    subject: str,
    source: str,
    chapter_no: int,
    chapter_title: str,
) -> list[dict]:
    """
    Parse NIOS-style pages into structured chunks.

    Section structure detected:
      - Numbered sections: "1.1 ...", "1.2.3 ..."
      - Special blocks: INTEXT QUESTIONS, TERMINAL EXERCISE,
                        WHAT YOU HAVE LEARNT, Examples, OBJECTIVES,
                        ANSWERS TO ...
    """
    lines = pages_to_lines(pages)
    # Filter out bare page-number lines
    lines = [l for l in lines if not is_page_number_line(l["text"])]

    eq_map = get_equations_by_page(pages)

    chunks = []
    chunk_id_counter = [0]

    # Section tracking
    current_section_no = None
    current_section_title = None
    current_subsection_no = None
    current_subsection_title = None

    # Accumulator
    current_lines: list[dict] = []
    current_page_start = pages[0]["page_no"] if pages else 1
    current_page_end = current_page_start
    current_mode = "content"   # content | intext_question | terminal_exercise | summary | example | objectives | answers

    def flush_chunk():
        nonlocal current_page_start

        text = clean_text('\n'.join(l["text"] for l in current_lines))
        if len(text.strip()) < 30:
            current_lines.clear()
            return

        # Collect equations from the LAST page of this chunk only.
        # Using the full range causes equations to bleed across chunk boundaries
        # because adjacent pages share the same equation list when equations span pages.
        # Using only current_page_end keeps each equation with the chunk it closes on.
        chunk_equations = list(eq_map.get(current_page_end, []))

        chunk_id_counter[0] += 1
        sec_str = (current_section_no or "intro").replace(".", "_")
        chunk_id = (
            f"{source.lower()}_{subject.lower()[:3]}"
            f"_ch{chapter_no:02d}"
            f"_s{sec_str}_{chunk_id_counter[0]}"
        )

        chunks.append({
            "chunk_id":       chunk_id,
            "source":         source,
            "subject":        subject,
            "chapter_no":     chapter_no,
            "chapter_title":  chapter_title,
            "section":        current_section_title,
            "section_no":     current_section_no,
            "subsection":     current_subsection_title,
            "subsection_no":  current_subsection_no,
            "chunk_type":     current_mode,
            "text":           text,
            "equations":      chunk_equations,
            "page_start":     current_page_start,
            "page_end":       current_page_end,
        })
        current_lines.clear()
        current_page_start = current_page_end

    # ── Main parsing loop ─────────────────────────────────────────────────────
    i = 0
    n = len(lines)
    while i < n:
        line_obj = lines[i]
        line = line_obj["text"].strip()
        page_no = line_obj["page_no"]
        current_page_end = page_no

        # ── Structural section boundary (numbered: 1.1, 1.2.3, …) ────────────
        detected = detect_numbered_section(line)
        if detected:
            flush_chunk()
            current_mode = "content"
            no, title = detected
            if count_dots(no) == 1:          # e.g. "1.3"
                current_section_no = no
                current_section_title = title
                current_subsection_no = None
                current_subsection_title = None
            elif count_dots(no) == 2:        # e.g. "1.3.1"
                current_subsection_no = no
                current_subsection_title = title
            current_page_start = page_no
            # Don't append heading — it becomes metadata
            i += 1
            continue

        # ── INTEXT QUESTIONS block ────────────────────────────────────────────
        if INTEXT_QUESTION_RE.match(line):
            flush_chunk()
            current_mode = "intext_question"
            current_page_start = page_no
            i += 1
            continue

        # ── TERMINAL EXERCISE block ───────────────────────────────────────────
        if TERMINAL_EXERCISE_RE.match(line):
            flush_chunk()
            current_mode = "terminal_exercise"
            current_page_start = page_no
            i += 1
            continue

        # ── WHAT YOU HAVE LEARNT (summary) ───────────────────────────────────
        if WHAT_YOU_LEARNED_RE.match(line):
            flush_chunk()
            current_mode = "summary"
            current_page_start = page_no
            i += 1
            continue

        # ── ANSWERS TO … block ───────────────────────────────────────────────
        if ANSWER_SECTION_RE.match(line):
            flush_chunk()
            current_mode = "answers"
            current_page_start = page_no
            current_lines.append(line_obj)
            i += 1
            continue

        # ── OBJECTIVES block ─────────────────────────────────────────────────
        if OBJECTIVES_RE.match(line):
            flush_chunk()
            current_mode = "objectives"
            current_page_start = page_no
            i += 1
            continue

        # ── Example block ────────────────────────────────────────────────────
        if EXAMPLE_RE.match(line) and current_mode not in ("terminal_exercise", "answers"):
            flush_chunk()
            current_mode = "example"
            current_page_start = page_no
            current_lines.append(line_obj)   # keep "Example N.N" line
            i += 1
            continue

        # ── Default: accumulate ───────────────────────────────────────────────
        current_lines.append(line_obj)
        i += 1

    flush_chunk()
    return chunks


# ---------------------------------------------------------------------------
# Merge short chunks
# ---------------------------------------------------------------------------

def merge_short_chunks(chunks: list[dict], min_chars: int = 80) -> list[dict]:
    """Merge chunks that are too short into the preceding chunk."""
    if not chunks:
        return chunks
    merged = [chunks[0]]
    for chunk in chunks[1:]:
        if len(chunk["text"]) < min_chars and merged:
            merged[-1]["text"] += "\n\n" + chunk["text"]
            merged[-1]["page_end"] = chunk["page_end"]
            # Merge equations too
            merged[-1]["equations"] = list(set(
                merged[-1]["equations"] + chunk["equations"]
            ))
        else:
            merged.append(chunk)
    return merged


# ---------------------------------------------------------------------------
# Save output
# ---------------------------------------------------------------------------

def save_chunks(chunks: list[dict], output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(chunks)} chunks → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parse precomputed Marker extraction JSONs into structured chunks."
    )
    parser.add_argument(
        "--input", default=None,
        help=(
            "Path to the precomputed extraction JSON (e.g. Phy_NIOS.json). "
            "If omitted, the script auto-detects the first .json file in the "
            "current directory (excluding the output/ folder)."
        )
    )
    parser.add_argument(
        "--output", default="output/chunks.json",
        help="Output path for the chunks JSON (default: output/chunks.json)"
    )
    parser.add_argument(
        "--subject", default="Physics",
        help="Subject name to embed in chunk metadata (default: Physics)"
    )
    parser.add_argument(
        "--source", default="NIOS",
        help="Source/board name (default: NIOS)"
    )
    parser.add_argument(
        "--chapter_no", type=int, default=1,
        help="Chapter number (default: 1)"
    )
    parser.add_argument(
        "--chapter_title", default="",
        help="Chapter title (default: inferred from filename)"
    )
    parser.add_argument(
        "--min_chars", type=int, default=80,
        help="Minimum characters per chunk before merging (default: 80)"
    )
    parser.add_argument(
        "--preview", type=int, default=3,
        help="Number of chunks to preview after parsing (default: 3)"
    )

    args = parser.parse_args()

    # ── Auto-detect input if not given ────────────────────────────────────────
    if args.input is None:
        candidates = [
            str(p) for p in Path(".").glob("*.json")
            if p.parent.name != "output"
        ]
        if not candidates:
            print("ERROR: No .json files found in the current directory.")
            print("       Please specify one with: --input <file.json>")
            return
        args.input = sorted(candidates)[0]
        print(f"[Auto-detected input] {args.input}")

    # Set a smarter default output name based on the input filename
    if args.output == "output/chunks.json":
        stem = Path(args.input).stem
        args.output = f"output/{stem}_chunks.json"

    # Auto-detect chapter title from content if not provided on CLI
    pages, _meta = load_extraction_json(args.input)
    if args.chapter_title:
        chapter_title = args.chapter_title
    else:
        chapter_title = infer_chapter_title(pages)
        if chapter_title:
            print(f"[Auto-detected title] {chapter_title}")
        else:
            chapter_title = Path(args.input).stem.replace("_", " ")
            print(f"[Fallback title] {chapter_title}")

    print(f"\nParsing: {args.source} {args.subject} Ch{args.chapter_no} — {chapter_title}")
    print(f"  Input : {args.input}")
    print(f"  Output: {args.output}")

    # Parse
    chunks = parse_nios(
        pages,
        subject=args.subject,
        source=args.source,
        chapter_no=args.chapter_no,
        chapter_title=chapter_title,
    )

    # Merge short chunks
    chunks = merge_short_chunks(chunks, min_chars=args.min_chars)

    # Save
    save_chunks(chunks, args.output)

    # Preview
    if args.preview > 0:
        print(f"\n--- Preview (first {args.preview} chunks) ---")
        for chunk in chunks[:args.preview]:
            meta = {k: v for k, v in chunk.items() if k not in ("text", "equations")}
            print(json.dumps(meta, indent=2))
            print(f"  text preview : {chunk['text'][:150]}...")
            if chunk["equations"]:
                print(f"  equations    : {chunk['equations'][:2]}")
            print()

    # Stats
    types = {}
    for c in chunks:
        t = c["chunk_type"]
        types[t] = types.get(t, 0) + 1
    print("Chunk type breakdown:")
    for t, cnt in sorted(types.items(), key=lambda x: -x[1]):
        print(f"  {t:<22} {cnt}")
    print(f"\nTotal chunks: {len(chunks)}")


if __name__ == "__main__":
    main()
