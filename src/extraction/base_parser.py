import re
import json
import os
from typing import Optional

class BaseParser:
    def __init__(self, subject: str, source: str, chapter_no: int, chapter_title: str):
        self.subject = subject
        self.source = source
        self.chapter_no = chapter_no
        self.chapter_title = chapter_title

    def clean_text(self, text: str) -> str:
        """Base normalization of extracted text."""
        if not text:
            return ""
        # Remove repeated characters (OCR artifacts) like ---- or ====
        text = re.sub(r'(.)\1{3,}', lambda m: m.group(1), text)
        # Normalize whitespace (spaces and tabs, but leave newlines)
        text = re.sub(r'[ \t]+', ' ', text)
        # Collapse excessive blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Fix broken hyphenated words at line endings (e.g. "exam-\nple")
        text = re.sub(r'([A-Za-z]+)-\n([a-z]+)', r'\1\2\n', text)
        return text.strip()

    def preprocess_pages(self, pages: list[dict]) -> list[dict]:
        """Hook for board-specific page-level preprocessing."""
        # By default, applies base clean_text to each page and removes un-printable/unicode noise
        for page in pages:
            text = page.get("text", "")
            # Basic unicode clean (retain standard ascii and common punctuation)
            # We don't want to remove degree signs or standard scientific characters, so just basic strip
            page["text"] = self.clean_text(text)
        return pages

    def load_extraction_json(self, json_path: str) -> tuple[list[dict], dict]:
        """Load a Marker precomputed extraction JSON."""
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)

        pages = data.get("pages", [])
        if not pages:
            raise ValueError(f"No pages found in {json_path}")
        return pages, data

    def infer_chapter_title(self, pages: list[dict]) -> str:
        """Try to auto-detect chapter title from first page.

        Handles common formats:
          1. ALL-CAPS heading: "UNITS, DIMENSIONS AND VECTORS"  (NIOS style)
          2. Chapter-dot-title: "1. Living World"  (Maharashtra style single line)
          3. Chapter-dot newline Title: "1.\nUnits and Measurements" (Maharashtra style multi-line)
        """
        if not pages:
            return self.chapter_title
        first_text = pages[0].get("text", "")
        lines = [line.strip() for line in first_text.split("\n") if line.strip()]
        
        for i, line in enumerate(lines):
            # Format 1: All-caps heading (NIOS style)
            if re.match(r'^[A-Z][A-Z0-9\s,\(\)\-\/]{8,80}$', line):
                if re.match(r'^MODULE\s*-', line):
                    continue
                return line.title()
                
            # Format 2: "N. Title" (Only matches top level chapter numbers, never 1.1)
            m = re.match(r'^\d{1,2}\.\s+(.{4,80})$', line)
            if m:
                extracted = m.group(1).strip()
                if "introduction" not in extracted.lower():
                    return extracted.title()
                    
            # Format 3: "N." followed by title on next line
            if re.match(r'^\d{1,2}\.$', line) and i + 1 < len(lines):
                next_line = lines[i+1]
                if 4 <= len(next_line) <= 80 and "introduction" not in next_line.lower():
                    return next_line.title()
                    
        # Fallback: if no format matched, often the first line is the naked title
        if lines:
            fallback = lines[0].strip()
            # If the first line is just a stray number, use the second line
            if re.match(r'^\d{1,2}\.?$', fallback) and len(lines) > 1:
                fallback = lines[1].strip()
                
            if 4 <= len(fallback) <= 100 and "introduction" not in fallback.lower():
                return fallback.title()
                
        return self.chapter_title or ""

    def pages_to_lines(self, pages: list[dict]) -> list[dict]:
        """Flatten pages into a list of line dicts."""
        lines = []
        for page in pages:
            for line in page["text"].split('\n'):
                # optionally clean isolated page numbers
                ln = line.strip()
                if not re.match(r'^\d{1,3}$', ln):
                    lines.append({"page_no": page["page_no"], "text": line})
        return lines

    def get_equations_by_page(self, pages: list[dict]) -> dict[int, list[str]]:
        """Map page_no -> equations."""
        eq_map = {}
        for page in pages:
            eqs = page.get("equations", [])
            if eqs:
                eq_map[page["page_no"]] = eqs
        return eq_map

    def merge_short_chunks(self, chunks: list[dict], min_chars: int = 80) -> list[dict]:
        """Merge short chunks with the previous chunk."""
        if not chunks:
            return chunks
        merged = [chunks[0]]
        for chunk in chunks[1:]:
            if len(chunk["text"]) < min_chars and merged:
                merged[-1]["text"] += "\n\n" + chunk["text"]
                merged[-1]["page_end"] = max(merged[-1]["page_end"], chunk["page_end"])
                # Merge equations safely
                merged[-1]["equations"] = list(set(
                    merged[-1].get("equations", []) + chunk.get("equations", [])
                ))
            else:
                merged.append(chunk)
        return merged

    def save_chunks(self, chunks: list[dict], output_path: str):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

    def parse(self, json_path: str, output_path: str, min_chars: int = 80):
        """Execute full parsing pipeline."""
        pages, _meta = self.load_extraction_json(json_path)
        if not self.chapter_title:
            self.chapter_title = self.infer_chapter_title(pages)
        
        pages = self.preprocess_pages(pages)
        chunks = self.process_pages_to_chunks(pages)
        chunks = self.merge_short_chunks(chunks, min_chars=min_chars)
        self.save_chunks(chunks, output_path)
        return chunks

    def process_pages_to_chunks(self, pages: list[dict]) -> list[dict]:
        """Must be implemented by board-specific subclass."""
        raise NotImplementedError("Subclasses must implement process_pages_to_chunks()")

    # ---- Helpers for block extraction ----
    def generate_chunk_id(self, chunk_id_counter: list, section_no: str) -> str:
        chunk_id_counter[0] += 1
        sec_str = (section_no or "intro").replace(".", "_")
        return (
            f"{self.source.lower()}_{self.subject.lower()[:3]}"
            f"_ch{self.chapter_no:02d}"
            f"_s{sec_str}_{chunk_id_counter[0]}"
        )

