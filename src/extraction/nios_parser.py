import re
from typing import Optional
from src.extraction.base_parser import BaseParser

# NIOS-specific running headers
_RUNNING_HEADER_RES = [
    re.compile(r'^MODULE\s*-\s*\d+\s*$', re.MULTILINE),
    re.compile(r'^MODULE\s*-\s*\d+\s+.+$', re.MULTILINE),
    re.compile(r'^Motion,\s+Force\s+and\s+Energy\s*$', re.MULTILINE),
    re.compile(r'^Units,\s+Dimensions\s+and\s+Vectors\s*$', re.MULTILINE),
    re.compile(r'^Units\.\s+Dimensions\s+and\s+Vectors\s*$', re.MULTILINE),
]

SECTION_RE = re.compile(r'^(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?\.?)\s+([A-Za-z].+?)\s*$')
INTEXT_QUESTION_RE = re.compile(r'^INTEXT\s+QUESTIONS?\s*[\d\.]*\s*$', re.IGNORECASE)
TERMINAL_EXERCISE_RE = re.compile(r'^TERMINAL\s+EXERCISE\s*$', re.IGNORECASE)
WHAT_YOU_LEARNED_RE = re.compile(r'^WHAT\s+YOU\s+HAVE\s+LEARNT\s*$', re.IGNORECASE)
EXAMPLE_RE = re.compile(r'^Example\s+\d+[\.\d]*', re.IGNORECASE)
OBJECTIVES_RE = re.compile(r'^OBJECTIVES\s*$', re.IGNORECASE)
ANSWER_SECTION_RE = re.compile(r'^ANSWERS?\s+TO\s+', re.IGNORECASE)

class NiosParser(BaseParser):
    def preprocess_pages(self, pages: list[dict]) -> list[dict]:
        pages = super().preprocess_pages(pages)
        for page in pages:
            text = page.get("text", "")
            for pat in _RUNNING_HEADER_RES:
                text = pat.sub('', text)
            page["text"] = self.clean_text(text)
        return pages

    def detect_numbered_section(self, line: str) -> Optional[tuple]:
        line = line.strip()
        m = SECTION_RE.match(line)
        if m:
            no = m.group(1).rstrip('.')
            title = m.group(2).strip()
            return no, title
        return None

    def process_pages_to_chunks(self, pages: list[dict]) -> list[dict]:
        lines = self.pages_to_lines(pages)
        eq_map = self.get_equations_by_page(pages)

        chunks = []
        chunk_id_counter = [0]

        current_section_no = None
        current_section_title = None
        current_subsection_no = None
        current_subsection_title = None

        current_lines: list[dict] = []
        current_page_start = pages[0]["page_no"] if pages else 1
        current_page_end = current_page_start
        current_mode = "content"

        def flush_chunk():
            nonlocal current_page_start
            text = self.clean_text('\n'.join(l["text"] for l in current_lines))
            if len(text.strip()) < 30:
                current_lines.clear()
                return

            chunk_equations = list(eq_map.get(current_page_end, []))
            chunk_id = self.generate_chunk_id(chunk_id_counter, current_section_no)

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
                "chunk_type":     current_mode,
                "text":           text,
                "equations":      chunk_equations,
                "page_start":     current_page_start,
                "page_end":       current_page_end,
                "figure_references": []
            })
            current_lines.clear()
            current_page_start = current_page_end

        for line_obj in lines:
            line = line_obj["text"].strip()
            page_no = line_obj["page_no"]
            current_page_end = page_no

            detected = self.detect_numbered_section(line)
            if detected:
                flush_chunk()
                current_mode = "content"
                no, title = detected
                if no.count('.') == 1:
                    current_section_no, current_section_title = no, title
                    current_subsection_no, current_subsection_title = None, None
                elif no.count('.') == 2:
                    current_subsection_no, current_subsection_title = no, title
                current_page_start = page_no
                continue

            if INTEXT_QUESTION_RE.match(line):
                flush_chunk()
                current_mode = "intext_question"
                current_page_start = page_no
                continue

            if TERMINAL_EXERCISE_RE.match(line):
                flush_chunk()
                current_mode = "terminal_exercise"
                current_page_start = page_no
                continue

            if WHAT_YOU_LEARNED_RE.match(line):
                flush_chunk()
                current_mode = "summary"
                current_page_start = page_no
                continue

            if ANSWER_SECTION_RE.match(line):
                flush_chunk()
                current_mode = "answers"
                current_page_start = page_no
                current_lines.append(line_obj)
                continue

            if OBJECTIVES_RE.match(line):
                flush_chunk()
                current_mode = "objectives"
                current_page_start = page_no
                continue

            if EXAMPLE_RE.match(line) and current_mode not in ("terminal_exercise", "answers"):
                flush_chunk()
                current_mode = "example"
                current_page_start = page_no
                current_lines.append(line_obj)
                continue

            current_lines.append(line_obj)

        flush_chunk()
        return chunks

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subject", default="Physics")
    parser.add_argument("--source", default="NIOS")
    parser.add_argument("--chapter_no", type=int, default=1)
    args = parser.parse_args()
    
    np = NiosParser(
        subject=args.subject,
        source=args.source,
        chapter_no=args.chapter_no,
        chapter_title=""
    )
    np.parse(args.input, args.output)
    print(f"Parsed NIOS chapter into {args.output}")
