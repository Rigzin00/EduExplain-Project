# NIOS Physics Textbook — Full Dataset Pipeline Documentation

> **Project:** NIOS Physics Textbook → LLM Training Dataset  
> **Final Output:** 30 per-chapter multimodal JSON files in `data/final/`  
> **Total Resources Generated:** 1,191 (text + diagram objects)

---

## Table of Contents

1. [Project Overview & Architecture](#1-project-overview--architecture)  
2. [Stage 0 — PDF Extraction (Google Colab)](#2-stage-0--pdf-extraction-google-colab)  
3. [Stage 1 — Chunking (`nios_parser.py`)](#3-stage-1--chunking-nios_parserpy)  
4. [Stage 2 — Resource Builder](#4-stage-2--resource-builder)  
5. [Stage 3 — Multimodal Merge](#5-stage-3--multimodal-merge)  
6. [Stage 4 — Master Orchestrator (`process_book.py`)](#6-stage-4--master-orchestrator-process_bookpy)  
7. [Known Issues & Fixes Applied](#7-known-issues--fixes-applied)  
8. [Final Output Format](#8-final-output-format)  
9. [How to Run the Full Pipeline](#9-how-to-run-the-full-pipeline)

---

## 1. Project Overview & Architecture

The goal of this project is to fully automate the conversion of NIOS Physics textbook PDFs into a structured, validated, multimodal JSON dataset ready for LLM training and RAG applications.

### Full Pipeline Flowchart

```
PDF (per chapter)
    │
    ▼
[Stage 0] Google Colab — Marker PDF Extractor
    │   Output: Phy_NIOS.json  +  Phy_NIOS_images.json  +  figures/
    │
    ▼
[Stage 1] nios_parser.py — Chunker
    │   Output: output/Phy_NIOS_chunks.json
    │
    ▼
[Stage 2] run_resource_builder.py — Schema Validator + Enricher 
    │   Output: output/Phy_NIOS_resources.json
    │
    ▼
[Stage 3] merge_multimodal.py — Image Attacher
    │   Output: output/Phy_NIOS_multimodal.json
    │
    ▼
[Stage 4] process_book.py — Master Orchestrator (runs ALL chapters)
         Output: data/final/ch01_multimodal.json
                 data/final/ch02_multimodal.json
                 ...
                 data/final/ch30_multimodal.json
```

### Directory Structure

```
dataset/
├── Physics/                    ← All 30 chapter folders uploaded here
│   ├── chapter 1/
│   │   ├── Phy_NIOS.json       ← Raw text extraction
│   │   ├── Phy_NIOS_images.json
│   │   └── figures/
│   ├── chapter 2/
│   │   ├── Phy__Ch2.json       ← (Inconsistent naming — auto-detected)
│   │   ├── Phy_NIOS_Ch2_images.json
│   │   └── ch2_figures/
│   └── ... (chapters 3–30)
│
├── src/
│   └── resources/
│       ├── schema.py           ← The Resource dataclass
│       ├── resource_builder.py ← Builds + validates Resource objects
│       └── resource_types.py   ← Type mappings and ID prefixes
│
├── nios_parser.py              ← Stage 1: Chunker
├── run_resource_builder.py     ← Stage 2: Resource Builder CLI
├── merge_multimodal.py         ← Stage 3: Multimodal Merger
├── process_book.py             ← Stage 4: Master Orchestrator
│
└── data/
    └── final/
        ├── ch01_multimodal.json
        ├── ch02_multimodal.json
        └── ...ch30_multimodal.json
```

---

## 2. Stage 0 — PDF Extraction (Google Colab)

**Script:** `untitled2.py`  
**Runs on:** Google Colab (uses GPU resources)  
**Tool used:** [Marker](https://github.com/VikParuchuri/marker) by VikParuchuri

### What it does
Converts each chapter's PDF into a structured JSON file containing:
- Per-page raw text
- Detected LaTeX equations
- Extracted image metadata

### BEFORE — Original config (fast but loses bullet lists)

```python
MARKER_CONFIG = {
    "output_format": "json",
    "langs": ["en"],
    "disable_image_extraction": True,   # Fast but DROPS bulleted lists rendered as images
    "force_ocr": False,
}
```

### AFTER — Updated config (full fidelity)

```python
MARKER_CONFIG = {
    "output_format": "json",
    "langs": ["en"],
    "disable_image_extraction": False,  # Extracts everything including lists
    "force_ocr": True,                  # Forces OCR on all blocks
}
```

### Pillow Bug Fix (added to Colab notebook)

When `disable_image_extraction=False`, Google Colab's upgraded Pillow library crashes with:
```
TypeError: function takes at most 16 arguments (17 given)
```

**Fix — add this to the top of your Colab notebook:**
```python
!pip install marker-pdf -q
!pip install "Pillow<10.3.0" -q   # ← This line fixes the crash
```

### Output Format (`Phy_NIOS.json`)

```json
{
  "pages": [
    {
      "page_no": 2,
      "text": "1.1 PHYSICAL WORLD AND MEASUREMENTS\n...",
      "equations": ["F = ma", "E = mc^2"],
      "_extraction_method": "surya"
    }
  ],
  "extraction_backend": "marker",
  "has_equations": true
}
```

### Missing Bullet Points — Root Cause & Fix

**Root Cause:** Surya OCR (inside Marker) falsely detects certain bulleted lists as "image/figure" blocks.  
With `disable_image_extraction: True`, these image blocks are silently deleted, causing whole lists (like the `(i)-(vii)` examples in Chapter 1) to vanish from the output JSON entirely.

**Temporary Fix:** If `disable_image_extraction` must stay `True` for performance, manually patch the missing content directly into the raw JSON:

```json
{
  "page_no": 2,
  "text": "...for example:\n(i) A falling apple led to the understanding of gravitation.\n(ii) Production of electrical energy...\n(iii) Receiving messages...\n(iv) Landing on the moon...\n(v) The study of outer space...\n(vi) Lasers and its numerous applications\n(vii) High speed computers, and many more.\n1.1.2 Nature of Physical Laws\n..."
}
```

---

## 3. Stage 1 — Chunking (`nios_parser.py`)

**Script:** `nios_parser.py`  
**Input:** Raw extraction JSON (e.g., `Phy_NIOS.json`)  
**Output:** `output/<name>_chunks.json`

### What it does
Splits the flat per-page text into semantic, section-level chunks.

### Key Features

| Feature | Implementation |
|---|---|
| Section detection | Regex: `^\d{1,2}\.\d{1,2}(?:\.\d{1,2})?\.?\s+[A-Za-z]` |
| Running header removal | Strips MODULE banners, page titles on every page |
| Special block detection | `INTEXT QUESTIONS`, `TERMINAL EXERCISE`, `WHAT YOU HAVE LEARNT`, `Example N`, `ANSWERS TO` |
| Chapter title auto-detection | Reads first ALL-CAPS line from page 1 |
| Equation attribution | Maps LaTeX equations to the correct page/chunk |

### BEFORE — Issues found in early output

1. Running page headers like *"Motion, Force and Energy"* were embedded mid-sentence in chunks.
2. Section numbers with trailing dots (e.g., `1.2.4.`) were not matched by the regex.
3. LaTeX equations from the wrong page were leaking into adjacent chunks.
4. `--input` was mandatory, requiring the user to type the filename every time.

### AFTER — Fixes applied

```python
# FIX 1: Running header patterns (strips module/chapter banners)
_RUNNING_HEADER_RES = [
    re.compile(r'^MODULE\s*-\s*\d+\s*$', re.MULTILINE),
    re.compile(r'^Motion,\s+Force\s+and\s+Energy\s*$', re.MULTILINE),
    re.compile(r'^Units,\s+Dimensions\s+and\s+Vectors\s*$', re.MULTILINE),
]

# FIX 2: Section regex now handles trailing dots
SECTION_RE = re.compile(r'^(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?\.?)\s+([A-Za-z].+?)\s*$')

# FIX 3: Equation attribution scoped to chunk page range only
# FIX 4: Auto-detection of first JSON in directory (--input now optional)
```

### Chunk Output Format

```json
{
  "chunk_id": "nios_phy_ch01_s1_1_1",
  "source": "NIOS",
  "subject": "Physics",
  "chapter_no": 1,
  "chapter_title": "Units, Dimensions And Vectors",
  "section": "PHYSICAL WORLD AND MEASUREMENTS",
  "section_no": "1.1",
  "subsection": "Physics: Scope and Excitement",
  "subsection_no": "1.1.1",
  "chunk_type": "content",
  "text": "The scope of Physics is very wide...",
  "equations": [],
  "page_start": 2,
  "page_end": 2
}
```

### Chunk Types Detected

| `chunk_type` | Trigger Pattern |
|---|---|
| `content` | Regular numbered section text |
| `example` | Lines starting with `Example N` |
| `intext_question` | `INTEXT QUESTIONS` header |
| `terminal_exercise` | `TERMINAL EXERCISE` header |
| `answers` | `ANSWERS TO ...` header |

### How to run

```bash
python nios_parser.py
# Auto-detects first .json file in the directory

python nios_parser.py --input Phy_NIOS.json --output output/Phy_NIOS_chunks.json
```

---

## 4. Stage 2 — Resource Builder

**Scripts:** `run_resource_builder.py` + `src/resources/resource_builder.py`  
**Input:** `output/<name>_chunks.json`  
**Output:** `output/<name>_resources.json`

### What it does
Upgrades raw chunk dicts into strictly validated `Resource` objects conforming to the benchmark schema defined in `src/resources/schema.py`.

- Assigns a unique, human-readable `resource_id` to every item
- Validates all required fields (crashes non-compliant items with a `[SKIP]` log)
- Detects `has_equation` flag from either the raw extraction or LaTeX regex fallback
- Maps NIOS-specific `chunk_type` strings to benchmark `resource_type` strings

### BEFORE — Problem: not compatible with NIOS chunks

The raw `resource_types.py` only knew about NCERT chunk types. NIOS chunks with `"chunk_type": "content"` or `"terminal_exercise"` would either crash or silently be skipped.

Also, chunks were missing the `class_no` field required by `schema.py`.

### AFTER — Fixed in `run_resource_builder.py`

```python
# Dynamically register NIOS-specific chunk types
resource_types.CHUNK_TYPE_TO_RESOURCE_TYPE.update({
    "content":           "explanation",
    "terminal_exercise": "exercise",
    "answers":           "explanation",
})

# Inject missing required fields before building
for chunk in parsed_chunks:
    if "class_no" not in chunk:
        chunk["class_no"] = args.class_no   # from --class-no CLI arg
    if chunk.get("chapter_no") is None:
        chunk["chapter_no"] = 0
```

### Resource ID Format

```
TXT_12_CH01_S1_1_001   ← explanation resource
EXP_12_CH01_S1_2_001   ← example resource
EX_12_CH01_S2_1_001    ← exercise resource
IMG_12_CH01_001        ← diagram resource
```

### Resource Output Format

```json
{
  "document_id": "NIOS_PHY_CH01",
  "resource_id": "TXT_12_CH01_S1_1_001",
  "resource_type": "explanation",
  "class_no": 12,
  "subject": "Physics",
  "chapter_no": 1,
  "chapter_title": "Units, Dimensions And Vectors",
  "section": "PHYSICAL WORLD AND MEASUREMENTS",
  "subsection": "Physics: Scope and Excitement",
  "page_start": 2,
  "page_end": 2,
  "text": "The scope of Physics is very wide...",
  "image_path": null,
  "image_caption": null,
  "description": null,
  "summary": null,
  "concepts": [],
  "has_equation": false
}
```

### How to run

```bash
# Default paths
python run_resource_builder.py

# Custom paths
python run_resource_builder.py \
    --input output/Phy_NIOS_chunks.json \
    --raw Phy_NIOS.json \
    --output output/Phy_NIOS_resources.json \
    --subject NIOS_PHY \
    --class-no 12
```

---

## 5. Stage 3 — Multimodal Merge (`merge_multimodal.py`)

**Script:** `merge_multimodal.py`  
**Input:** `output/<name>_resources.json` + `<name>_images.json`  
**Output:** `output/<name>_multimodal.json`

### What it does
Pairs text resources with their corresponding images (by page number overlap), then spawns each image as an **independent `diagram` Resource** object.

### BEFORE v1 — Wrong approach: `images: []` array

```json
{
  "resource_id": "TXT_12_CH01_S1_1_003",
  "text": "Technology is the application...",
  "image_path": null,
  "images": [                            ← Redundant custom array
    {
      "image_id": "IMG_001",
      "image_path": "/content/.../page_4_fig_1.png",
      "caption": null
    }
  ]
}
```
**Problem:** Introduced a non-schema field `images: []`. When a chunk had multiple images, they were nested — harder to extract individually.

### BEFORE v2 — Wrong approach: comma-separated paths

**Problem:** A text chunk with 2 images on the same page would produce:
```json
"image_path": "/path/fig_1.png, /path/fig_2.png"
```
This broke schema typing (field expects single path string, not CSV) and made individual image extraction impossible.

### AFTER — Correct approach: independent `diagram` Resources

Each image from `Phy_NIOS_images.json` now becomes its **own top-level Resource object**:

```json
{
  "document_id": "NIOS_PHY_CH01",
  "resource_id": "IMG_12_CH01_001",
  "resource_type": "diagram",
  "class_no": 12,
  "subject": "Physics",
  "chapter_no": 1,
  "chapter_title": "Units, Dimensions And Vectors",
  "section": "PHYSICAL WORLD AND MEASUREMENTS",
  "subsection": "Physics, Technology and Society",
  "page_start": 4,
  "page_end": 4,
  "text": null,
  "image_path": "/content/drive/MyDrive/openstax/NIOS/figures/page_4_fig_1.png",
  "image_caption": null,
  "description": null,
  "summary": null,
  "concepts": [],
  "has_equation": false
}
```

**Why this is correct:**
- Fully compliant with `schema.py` (no custom fields added)
- Every image is independently addressable by `resource_id`
- If a student or retriever needs just that one figure, they can fetch it in one query
- Multiple images on the same page each get their own clean resource entry

### How to run

```bash
# Default paths (chapter 1 only)
python merge_multimodal.py
```

---

## 6. Stage 4 — Master Orchestrator (`process_book.py`)

**Script:** `process_book.py`  
**Input:** `Physics/` folder (all 30 chapter subdirectories)  
**Output:** `data/final/ch01_multimodal.json` ... `ch30_multimodal.json`

### What it does
Fully automates all 3 pipeline stages (chunking → resource building → multimodal merge) for every chapter folder found in the `Physics/` directory. Produces one clean output file per chapter.

### Problem it solves: Inconsistent file naming

The uploaded chapters had completely inconsistent file names:

| Chapter | Text JSON | Images JSON |
|---|---|---|
| ch 1 | `Phy_NIOS.json` | `Phy_NIOS_images.json` |
| ch 2 | `Phy__Ch2.json` | `Phy_NIOS_Ch2_images.json` |
| ch 3–30 | `Phy_Ch3.json` | `Phy_Ch3_images.json` |

**Solution:** The orchestrator uses smart glob-based auto-detection:
```python
# Text JSON: any .json file WITHOUT "image" in its name
text_files = [f for f in candidates if "image" not in basename(f).lower()]

# Images JSON: any .json file WITH "image" in its name
image_files = [f for f in candidates if "image" in basename(f).lower()]
```

### How to run

```bash
# Default (processes all chapters in Physics/ folder)
python process_book.py

# Custom paths and class number
python process_book.py --input Physics --output data/final --class-no 12
```

### Console Output Example

```
############################################################
  NIOS PHYSICS FULL-BOOK PIPELINE
  Found 30 chapter(s) to process
  Output >> C:\...\data\final
############################################################

==========================================================
  CHAPTER 01  |  Physics\chapter 1
==========================================================
  [*] Text JSON    : Phy_NIOS.json
  [*] Images JSON  : Phy_NIOS_images.json
  [1/3] Chunking...
        >> 34 chunks produced
  [2/3] Building resources...
        >> 34 text resources compiled
  [3/3] Merging images...
        >> 20 diagram resources added
        >> 54 total records in final dataset
  [+] Saved >> data/final/ch01_multimodal.json
...

############################################################
  PIPELINE COMPLETE
  [OK]  Success : 30 chapters >> [1, 2, 3, ..., 30]
  Output saved to: C:\...\data\final/
############################################################
```

---

## 7. Known Issues & Fixes Applied

| # | Issue | Root Cause | Fix Applied |
|---|---|---|---|
| 1 | Missing bullet points in Ch.1 | Marker/Surya flagged lists as images and dropped them when `disable_image_extraction=True` | Manually patched into `Phy_NIOS.json`; Long-term: add `!pip install "Pillow<10.3.0"` to Colab and set `disable_image_extraction=False` |
| 2 | Running headers mid-sentence | Marker inserts chapter/module title on every page | `_RUNNING_HEADER_RES` regex list in `clean_text()` |
| 3 | Section `1.2.4.` not matched | SECTION_RE didn't allow trailing dot | Added `\.?` at end of section number group |
| 4 | Equation spilling into adjacent chunks | Equations were mapped by page, not by chunk | Scoped equation lookup inside chunk page range only |
| 5 | `class_no` missing from NIOS chunks | NIOS parser didn't embed class number (NIOS ≠ NCERT) | Injected via `--class-no` arg in `run_resource_builder.py` |
| 6 | NIOS chunk types unrecognised by Resource Builder | `resource_types.py` only knew NCERT types | Registered `content`, `terminal_exercise`, `answers` dynamically |
| 7 | Images as CSV string in `image_path` | Tried to force multiple images into a single string field | Dropped this approach — each image is now an independent `diagram` Resource |
| 8 | Windows encoding crash (`UnicodeEncodeError`) | Used `→` and `✓` unicode characters in print statements | Replaced all with pure-ASCII `>>` and `[OK]` |
| 9 | Pillow crash in Colab (`TypeError: 17 arguments`) | Latest Pillow version broke Marker's image encoder API | `!pip install "Pillow<10.3.0"` added to Colab notebook |
| 10 | Inconsistent chapter file naming | Files uploaded from different extraction runs with different names | Smart glob auto-detection filters by presence of "image" substring |

---

## 8. Final Output Format

Every object in `data/final/ch0N_multimodal.json` follows this exact schema:

```json
{
  "document_id":   "NIOS_PHY_CH01",
  "resource_id":   "TXT_12_CH01_S1_1_001",
  "resource_type": "explanation",
  "class_no":      12,
  "subject":       "Physics",
  "chapter_no":    1,
  "chapter_title": "Units, Dimensions And Vectors",
  "section":       "PHYSICAL WORLD AND MEASUREMENTS",
  "subsection":    "Physics: Scope and Excitement",
  "page_start":    2,
  "page_end":      2,
  "text":          "The scope of Physics is very wide...",
  "image_path":    null,
  "image_caption": null,
  "description":   null,
  "summary":       null,
  "concepts":      [],
  "has_equation":  false
}
```

### Resource Type Reference

| `resource_type` | `resource_id` prefix | Used for |
|---|---|---|
| `explanation` | `TXT` | Regular content, summaries, answers |
| `example` | `EXP` | Worked examples |
| `exercise` | `EX` | Intext questions, terminal exercises |
| `diagram` | `IMG` | Extracted figures/images |
| `activity` | `ACT` | Hands-on activities |
| `table` | `TBL` | Structured tabular data |

### Dataset Statistics

| Metric | Value |
|---|---|
| Total chapters processed | 30 |
| Total resources generated | 1,191 |
| Total dataset size | ~1.74 MB |
| Output files | `data/final/ch01_multimodal.json` through `data/final/ch30_multimodal.json` |

---

## 9. How to Run the Full Pipeline

### One-time setup

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### For a single chapter (manual mode)

```bash
# Step 1: Chunk
python nios_parser.py --input Phy_NIOS.json --output output/Phy_NIOS_chunks.json

# Step 2: Build resources
python run_resource_builder.py \
    --input output/Phy_NIOS_chunks.json \
    --raw Phy_NIOS.json \
    --output output/Phy_NIOS_resources.json

# Step 3: Merge images
python merge_multimodal.py
```

### For ALL 30 chapters (automatic mode) ← Recommended

```bash
python process_book.py
```

> [!NOTE]
> Make sure all chapter folders are inside the `Physics/` directory before running. Each folder must contain a text extraction `.json` file and optionally an `_images.json` file.

> [!TIP]
> To process chapters for a different class number, use: `python process_book.py --class-no 11`

> [!IMPORTANT]
> The `src/` package must be present (with `__init__.py` files) for the Resource Builder imports to work. Do not move or rename any files inside `src/resources/`.
