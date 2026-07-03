"""
maha_parser.py
~~~~~~~~~~~~~~
Maharashtra State Board parser.
Inherits base IO / cleaning from BaseParser and adds:
  - Maharashtra-specific running-header removal
  - Strict OCR artifact cleaning  ()), www, Kev:, meaninaful …)
  - Inline special-block detection (mid-line markers are caught before flush)
  - Section / heading detection for "1.1 Title" pattern
  - Exercise sub-type splitting (mcq / exercise / project)
  - Figure reference extraction from text
"""

import re
from typing import Optional
from src.extraction.base_parser import BaseParser

# ---------------------------------------------------------------------------
# Maharashtra running-header patterns (stripped from every page)
# ---------------------------------------------------------------------------
_MAHA_HEADER_RES = [
    re.compile(r'^Maharashtra\s*State\s*Bureau.*$', re.MULTILINE | re.IGNORECASE),
    re.compile(r'^Std\.\s*XI\b.*$', re.MULTILINE | re.IGNORECASE),
    re.compile(r'^Std\.\s*XII\b.*$', re.MULTILINE | re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Common OCR substitution corrections (order matters — longer first)
# ---------------------------------------------------------------------------
_OCR_CORRECTIONS = [
    # Duplicated / stray punctuation artefacts from box-borders
    (re.compile(r'\)\)+\s*'), ''),          # )) Can you tell? → Can you tell?
    (re.compile(r'\(\(+\s*'), ''),          # (( stray open parens
    # "www " prefix used by the textbook for Internet tasks — keep URL but strip bare www
    (re.compile(r'^www\s+', re.MULTILINE), ''),
    # Common OCR word-level substitutions
    (re.compile(r'\bKev\b'), 'Key'),
    (re.compile(r'\bmeaninaful\b', re.IGNORECASE), 'meaningful'),
    (re.compile(r'\btransperent\b', re.IGNORECASE), 'transparent'),
    (re.compile(r'\bformaledehyde\b', re.IGNORECASE), 'formaldehyde'),
    (re.compile(r'\bexsitu\b', re.IGNORECASE), 'ex-situ'),
    # Stray lone digit lines (page numbers)
    (re.compile(r'^\s*\d{1,3}\s*$', re.MULTILINE), ''),
]

# ---------------------------------------------------------------------------
# Section heading regex  "1.1 Title"  (Maharashtra format)
# ---------------------------------------------------------------------------
SECTION_RE = re.compile(
    r'^(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\s+([A-Za-z].+?)\s*$'
)

# ---------------------------------------------------------------------------
# Special semantic-block markers
# These are matched BOTH as standalone lines AND as inline substrings
# so that blocks buried mid-paragraph are also caught.
# ---------------------------------------------------------------------------

# --- Standalone (full-line) patterns ---
_BLOCK_TRIGGERS_FULL = [
    # (compiled_regex,  chunk_type)
    (re.compile(r'^Can\s+you\s+recall\s*\??\s*$',       re.IGNORECASE), 'recall'),
    (re.compile(r'^Can\s+you\s+tell\s*\??\s*$',         re.IGNORECASE), 'recall'),
    (re.compile(r'^Think\s+about\s+it\s*\??\s*$',       re.IGNORECASE), 'recall'),
    (re.compile(r'^Use\s+your\s+brain\s+power\s*\??\s*$', re.IGNORECASE), 'recall'),
    (re.compile(r'^Do\s+you\s+know\s*\??\s*$',          re.IGNORECASE), 'callout'),
    (re.compile(r'^Internet\s+my\s+friend\s*\??\s*$',   re.IGNORECASE), 'internet_task'),
    (re.compile(r'^Find\s+out\s*\??\s*$',               re.IGNORECASE), 'activity'),
    (re.compile(r'^Know\s+the\s+scientists?\s*$',        re.IGNORECASE), 'scientist_box'),
    (re.compile(r'^Exercises?\s*$',                      re.IGNORECASE), 'exercise'),
]

# --- Inline (search inside accumulated text) patterns ---
# When these appear mid-paragraph we split before that point.
_BLOCK_TRIGGERS_INLINE = [
    # (compiled_search_regex, chunk_type)
    (re.compile(r'(?:^|\n)\s*Can\s+you\s+recall\s*\??',   re.IGNORECASE), 'recall'),
    (re.compile(r'(?:^|\n)\s*Can\s+you\s+tell\s*\??',     re.IGNORECASE), 'recall'),
    (re.compile(r'(?:^|\n)\s*Think\s+about\s+it\s*\??',   re.IGNORECASE), 'recall'),
    (re.compile(r'(?:^|\n)\s*Use\s+your\s+brain\s+power\s*\??', re.IGNORECASE), 'recall'),
    (re.compile(r'(?:^|\n)\s*Do\s+you\s+know\s*\??',      re.IGNORECASE), 'callout'),
    (re.compile(r'(?:^|\n)\s*(?:www\s+)?Internet\s+my\s+friend', re.IGNORECASE), 'internet_task'),
]

# --- Exercise sub-section patterns ---
MCQ_RE        = re.compile(r'^(?:Q\.\s*\d+\.\s*)?(?:multiple\s+choice|choose\s+the\s+correct)', re.IGNORECASE)
SUBJECTIVE_RE = re.compile(r'^(?:Q\.\s*\d+\.\s*)?(?:answer\s+the\s+following|answer\s+in\s+brief|write\s+short\s+notes)', re.IGNORECASE)
PROJECT_RE    = re.compile(r'^(?:project|project\s+work)\s*$', re.IGNORECASE)


def _apply_ocr_corrections(text: str) -> str:
    """Apply all OCR word-substitution corrections."""
    for pat, replacement in _OCR_CORRECTIONS:
        text = pat.sub(replacement, text)
    return text


class MahaParser(BaseParser):

    # -----------------------------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------------------------

    def preprocess_pages(self, pages: list[dict]) -> list[dict]:
        pages = super().preprocess_pages(pages)
        for page in pages:
            text = page.get("text", "")
            # Strip Maharashtra running headers
            for pat in _MAHA_HEADER_RES:
                text = pat.sub('', text)
            # Remove non-ASCII noise (keep °, curly-quotes, em-dash)
            text = re.sub(
                r'[^\x00-\x7F\°\n\u201C\u201D\u2018\u2019\u2013\u2014]', ' ', text
            )
            # Apply targeted OCR corrections
            text = _apply_ocr_corrections(text)
            page["text"] = self.clean_text(text)
        return pages

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def detect_numbered_section(self, line: str) -> Optional[tuple]:
        """Return (section_no, title) for numbered section headings."""
        m = SECTION_RE.match(line.strip())
        if m:
            return m.group(1).rstrip('.'), m.group(2).strip()
        return None

    def find_figure_references(self, text: str) -> list[str]:
        """Extract figure references like 'Fig. 1.2' or 'Figure 3.4a' from text."""
        refs = re.findall(
            r'(?:Fig\.|Fig\s|Figure)\s*(\d+\.\d+[a-zA-Z]?)', text, re.IGNORECASE
        )
        return sorted(set(refs))

    def _detect_inline_block(self, text: str) -> Optional[tuple[int, str]]:
        """
        Scan accumulated text for an inline block trigger.
        Returns (char_position, chunk_type) of the FIRST match, or None.
        """
        best_pos = None
        best_type = None
        for pat, btype in _BLOCK_TRIGGERS_INLINE:
            m = pat.search(text)
            if m:
                pos = m.start()
                if best_pos is None or pos < best_pos:
                    best_pos = pos
                    best_type = btype
        if best_pos is not None:
            return best_pos, best_type
        return None

    # -----------------------------------------------------------------------
    # Main parsing loop
    # -----------------------------------------------------------------------

    def process_pages_to_chunks(self, pages: list[dict]) -> list[dict]:
        lines    = self.pages_to_lines(pages)
        eq_map   = self.get_equations_by_page(pages)
        chunks   = []
        chunk_id_counter   = [0]

        current_section_no    = None
        current_section_title = None
        current_subsection_no    = None
        current_subsection_title = None

        current_lines:      list[dict] = []
        current_page_start = pages[0]["page_no"] if pages else 1
        current_page_end   = current_page_start
        current_mode       = "content"
        in_exercise_mode   = False

        def flush_chunk(mode_override: str = None):
            nonlocal current_page_start
            raw_text = '\n'.join(l["text"] for l in current_lines)
            text = self.clean_text(raw_text)
            if len(text.strip()) < 30:
                current_lines.clear()
                return

            chunk_equations = list(eq_map.get(current_page_end, []))
            chunk_id        = self.generate_chunk_id(chunk_id_counter, current_section_no)
            figure_refs     = self.find_figure_references(text)
            mapped_type     = mode_override or current_mode
            if mapped_type == "content":
                mapped_type = "explanation"

            chunks.append({
                "chunk_id":       chunk_id,
                "source":         self.source,
                "subject":        self.subject,
                "chapter_no":     self.chapter_no,
                "chapter_title":  self.chapter_title,
                "section":        current_section_title,
                "section_no":     current_section_no,
                "subsection":     current_subsection_title,
                "subsection_no":  current_subsection_no,
                "chunk_type":     mapped_type,
                "text":           text,
                "equations":      chunk_equations,
                "page_start":     current_page_start,
                "page_end":       current_page_end,
                "figure_references": figure_refs,
            })
            current_lines.clear()
            current_page_start = current_page_end

        def set_mode(new_mode: str, page_no: int):
            nonlocal current_mode, current_page_start, in_exercise_mode
            flush_chunk()
            current_mode = new_mode
            current_page_start = page_no
            if new_mode == "exercise":
                in_exercise_mode = True

        # ── Main loop ────────────────────────────────────────────────────────
        for line_obj in lines:
            line    = line_obj["text"].strip()
            page_no = line_obj["page_no"]
            current_page_end = page_no

            # 1. Numbered section heading
            detected = self.detect_numbered_section(line)
            if detected and not in_exercise_mode:
                flush_chunk()
                current_mode = "content"
                no, title = detected
                if no.count('.') == 1:
                    current_section_no, current_section_title = no, title
                    current_subsection_no, current_subsection_title = None, None
                elif no.count('.') >= 2:
                    current_subsection_no, current_subsection_title = no, title
                current_page_start = page_no
                continue

            # 2. Full-line special-block trigger
            triggered = False
            for pat, btype in _BLOCK_TRIGGERS_FULL:
                if pat.match(line):
                    set_mode(btype, page_no)
                    triggered = True
                    break
            if triggered:
                continue

            # 3. Exercise sub-type splits
            if in_exercise_mode:
                if MCQ_RE.match(line):
                    set_mode("mcq", page_no)
                    continue
                elif SUBJECTIVE_RE.match(line):
                    set_mode("exercise", page_no)
                    continue
                elif PROJECT_RE.match(line):
                    set_mode("project", page_no)
                    continue

            # 4. Accumulate line
            current_lines.append(line_obj)

            # 5. INLINE block detection — scan accumulated text for embedded triggers
            #    Only applies outside exercise mode to avoid over-splitting Q&A text.
            if not in_exercise_mode and len(current_lines) >= 2:
                combined = '\n'.join(l["text"] for l in current_lines)
                hit = self._detect_inline_block(combined)
                if hit:
                    split_pos, btype = hit
                    # Split the raw text at the detected trigger position
                    before_text = combined[:split_pos].strip()
                    # Rebuild current_lines from before text only, flush it, then start new block
                    if before_text:
                        before_lines = []
                        for lobj in current_lines:
                            if before_text.find(lobj["text"]) != -1:
                                before_lines.append(lobj)
                        # Simpler approach: fabricate a synthetic line dict for before text
                        current_lines.clear()
                        current_lines.append({"page_no": page_no, "text": before_text})
                    else:
                        current_lines.clear()
                    flush_chunk()
                    current_mode = btype
                    current_page_start = page_no
                    # The current line itself starts the new block
                    current_lines.append(line_obj)

        flush_chunk()
        return chunks


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",      required=True)
    ap.add_argument("--output",     required=True)
    ap.add_argument("--subject",    default="Biology")
    ap.add_argument("--source",     default="MAHA")
    ap.add_argument("--chapter_no", type=int, default=1)
    args = ap.parse_args()

    mp = MahaParser(subject=args.subject, source=args.source,
                    chapter_no=args.chapter_no, chapter_title="")
    chunks = mp.parse(args.input, args.output)
    print(f"Done. {len(chunks)} chunks -> {args.output}")
