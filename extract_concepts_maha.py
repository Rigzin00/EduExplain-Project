"""
extract_concepts_maha.py
=========================
Stage 4: LLM concept extraction for the EduExplain dataset.

Supports both datasets:
  - data_resources_2/  (MAHA + NIOS multimodal resources with summaries)
  - data/Maha/         (MAHA raw pipeline output, summary always null)

SCHEMA (verified by auditing all 7,834 resources in data_resources_2):
  document_id, resource_id, resource_type, class_no, subject,
  chapter_no, chapter_title, section, subsection, page_start,
  page_end, text, image_path, image_caption, description,
  summary, concepts, figure_references, has_equation

SUMMARY AVAILABILITY (per resource type, full dataset audit):
  explanation   3206  -- summary ALWAYS present  <- use summary
  example        225  -- summary ALWAYS present  <- use summary
  recall         318  -- summary always NULL     <- use text
  callout        194  -- summary always NULL     <- use text
  internet_task   57  -- summary always NULL     <- use text
  activity        21  -- summary ALWAYS present  <- use summary
  scientist_box    6  -- summary always NULL     <- use text
  diagram       3563  -- no text, no summary     <- SKIP
  exercise       243  -- no summary, OCR noise   <- SKIP
  mcq              1  -- no summary              <- SKIP

Usage:
    # Validate on single NIOS chapter (has summaries)
    python extract_concepts_maha.py --input data_resources_2/bio_nios/ch01_multimodal.json

    # Full NIOS biology folder
    python extract_concepts_maha.py --input data_resources_2/bio_nios

    # Full cross-board dataset
    python extract_concepts_maha.py --input data_resources_2

    # MAHA raw pipeline output (no summaries)
    python extract_concepts_maha.py --input data/Maha/BIO/11

    # Limit for quick testing
    python extract_concepts_maha.py --input data_resources_2/bio_nios/ch01_multimodal.json --max_resources 5

    # Resume a crashed run
    python extract_concepts_maha.py --input data_resources_2 --resume

Output:
    vocab/raw_concepts.json             per-resource concept list
    vocab/extraction_log.json           run statistics and errors
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

# Fix for Windows: HuggingFace tries to create symlinks which require Developer Mode.
# This env var disables that requirement and falls back to file copies instead.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

# Model options (pick based on available disk space):
#   Qwen/Qwen2.5-1.5B-Instruct  ~3 GB  -- fits on low-disk machines, good quality
#   Qwen/Qwen2.5-3B-Instruct    ~6 GB  -- better quality, needs ~6 GB free
#   Qwen/Qwen2.5-7B-Instruct    ~15GB  -- best quality, needs ~16 GB free
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

OUTPUT_DIR          = Path("vocab")
RAW_CONCEPTS_PATH   = OUTPUT_DIR / "raw_concepts.json"
EXTRACTION_LOG_PATH = OUTPUT_DIR / "extraction_log.json"

# ------------------------------------------------------------------
# Resource type filtering
#
# Determined by auditing ALL 7,834 resources in data_resources_2:
#
#  Type           Total  Summary?   Action
#  explanation    3206   ALWAYS     INCLUDE  (use summary)
#  example         225   ALWAYS     INCLUDE  (use summary)
#  recall          318   NEVER      INCLUDE  (use text)
#  callout         194   NEVER      INCLUDE  (use text)
#  internet_task    57   NEVER      INCLUDE  (use text)
#  activity         21   ALWAYS     INCLUDE  (use summary)
#  scientist_box     6   NEVER      INCLUDE  (use text)
#  diagram        3563   NEVER      SKIP     (text=null, image-only)
#  exercise        243   NEVER      SKIP     (Q&A stems, OCR noise)
#  mcq               1   NEVER      SKIP     (multiple-choice question)
# ------------------------------------------------------------------

INCLUDE_TYPES = {
    "explanation",    # Main textbook prose        -- summary available
    "example",        # Worked examples             -- summary available
    "recall",         # Prior-knowledge recall boxes-- text only
    "callout",        # Sidebar key definitions     -- text only
    "activity",       # Activity descriptions       -- summary available
    "internet_task",  # Research task + context     -- text only
    "scientist_box",  # Scientist biography boxes   -- text only
}

SKIP_TYPES = {
    "diagram",    # image_path only, text=null, summary=null -- nothing to extract
    "exercise",   # Q&A stems, OCR garbled, no concept prose
    "mcq",        # Multiple-choice question stems
}

# Types where summary is reliably populated (use it preferentially)
SUMMARY_RELIABLE_TYPES = {
    "explanation",
    "example",
    "activity",
}

# Minimum character length for content to be worth processing
MIN_TEXT_LENGTH = 40


# ------------------------------------------------------------------
# Prompt
# ------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert science educator specializing in Indian school curriculum (Classes 11-12).
Your task is to identify the key scientific concepts, processes, organisms, mechanisms,
or technical terms present in the given textbook passage.
Return ONLY a valid JSON array of strings. No explanation. No markdown fences.\
"""


def build_user_prompt(resource: dict, content: str) -> str:
    section_line = ""
    if resource.get("section"):
        section_line = f"  Section     : {resource['section']}\n"
    if resource.get("subsection"):
        section_line += f"  Subsection  : {resource['subsection']}\n"

    has_eq = "yes" if resource.get("has_equation") else "no"

    return f"""\
Extract 3-8 key scientific concepts from the textbook passage below.

RESOURCE METADATA:
  Subject      : {resource.get('subject', '')}
  Class        : {resource.get('class_no', '')}
  Chapter      : {resource.get('chapter_title', '')}
{section_line}  Resource type: {resource.get('resource_type', '')}
  Has equation : {has_eq}

TEXTBOOK PASSAGE:
{content}

INSTRUCTIONS:
- Extract scientific concepts, biological processes, organisms, technical terms,
  chemical/physical phenomena, mechanisms, or named laws/principles.
- Accept terms from any science domain:
    Biology  : mitosis, chloroplast, transpiration, taxonomy, herbarium
    Chemistry: covalent bond, redox reaction, entropy, mole concept
    Physics  : angular momentum, electromagnetic induction, centripetal force
- Include named scientific tools and taxonomic/ecological concepts when they are
  the subject of the passage (e.g. herbarium, taxidermy, in-situ conservation).
- Ignore: chapter/section headings, page numbers, generic filler words
  (important, basic, various, understand), author names and institution names
  UNLESS a concept/discovery named after them is discussed, question stems,
  OCR artefacts (random characters, partial words).
- If the passage contains no scientific content, return [].
- Return ONLY a JSON array of strings.

Example output: ["taxonomy", "herbarium", "in-situ conservation", "biodiversity", "taxidermy"]

Your output:"""


# ------------------------------------------------------------------
# Text cleaning
# ------------------------------------------------------------------

def clean_text(raw: str, max_chars: int = 1500) -> str:
    """
    Light-touch cleaning of OCR textbook text before feeding to LLM.
    Preserves scientific terms while removing obvious noise.
    """
    if not raw:
        return ""
    # Collapse excessive whitespace/newlines
    text = re.sub(r"\n{3,}", "\n\n", raw)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Remove bare page numbers (e.g. "12" alone on a line)
    text = re.sub(r"(?m)^\s*\d{1,3}\s*$", "", text)
    return text.strip()[:max_chars]


def get_extraction_content(resource: dict) -> str | None:
    """
    Determine what text to feed to the LLM for concept extraction.

    Strategy is type-aware based on full dataset audit:

      Types with reliable summaries (explanation, example, activity):
        -> prefer summary (cleaner, more condensed, no OCR noise)
        -> fallback to text if summary is absent/short

      Types without summaries (recall, callout, internet_task, scientist_box):
        -> use text directly (summary is always null for these)
        -> image_caption as last resort

    Returns None if no usable content is found above MIN_TEXT_LENGTH.
    """
    rtype = resource.get("resource_type", "")

    if rtype in SUMMARY_RELIABLE_TYPES:
        # For these types, summary is the preferred source
        summary = (resource.get("summary") or "").strip()
        if len(summary) >= MIN_TEXT_LENGTH:
            return clean_text(summary)
        # Fallback: summary can still be null in older pipeline outputs (e.g. data/Maha)
        text = (resource.get("text") or "").strip()
        if len(text) >= MIN_TEXT_LENGTH:
            return clean_text(text)
    else:
        # For recall, callout, internet_task, scientist_box — use text directly
        text = (resource.get("text") or "").strip()
        if len(text) >= MIN_TEXT_LENGTH:
            return clean_text(text)
        # Image caption as last resort (for future pipeline stages)
        caption = (resource.get("image_caption") or "").strip()
        if len(caption) >= MIN_TEXT_LENGTH:
            return clean_text(caption)

    return None


# ------------------------------------------------------------------
# Resource loading
# ------------------------------------------------------------------

def load_resources_from_path(input_path: Path) -> list[dict]:
    """
    Load resources from:
      - a single .json file (chapter file), OR
      - a directory (recursively finds all .json files)
    """
    resources = []
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.rglob("*.json"))
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for r in data:
                    r["_source_file"] = str(fp)
                resources.extend(data)
            elif isinstance(data, dict):
                data["_source_file"] = str(fp)
                resources.append(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [WARN] Could not read {fp}: {e}")

    return resources


def filter_resources(resources: list[dict]) -> list[dict]:
    """
    Apply filtering to select resources suitable for concept extraction.
    """
    included = []
    skip_counts = {"no_resource_id": 0, "type_skipped": 0, "no_content": 0}

    for r in resources:
        rid = r.get("resource_id")
        if not rid:
            skip_counts["no_resource_id"] += 1
            continue

        rtype = r.get("resource_type", "")
        if rtype in SKIP_TYPES or rtype not in INCLUDE_TYPES:
            skip_counts["type_skipped"] += 1
            continue

        content = get_extraction_content(r)
        if content is None:
            skip_counts["no_content"] += 1
            continue

        included.append(r)

    print(f"  Filter: skipped {skip_counts['type_skipped']} by type, "
          f"{skip_counts['no_content']} with no content, "
          f"{skip_counts['no_resource_id']} with no ID. "
          f"Kept {len(included)}.")
    return included


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

def load_model(model_name: str):
    print(f"Loading model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")
    return tokenizer, model


def generate_response(
    tokenizer,
    model,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 200,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ------------------------------------------------------------------
# Parsing LLM output
# ------------------------------------------------------------------

def parse_concept_list(raw_output: str) -> list[str]:
    """
    Robustly extract a JSON array of strings from LLM output.
    Handles: clean JSON, markdown fences, leading/trailing commentary.
    """
    text = re.sub(r"```(?:json)?\s*", "", raw_output)
    text = re.sub(r"```", "", text)

    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        return []

    try:
        result = json.loads(match.group())
        if isinstance(result, list):
            seen = set()
            cleaned = []
            for c in result:
                if not isinstance(c, str):
                    continue
                c = c.strip()
                if c and c.lower() not in seen:
                    seen.add(c.lower())
                    cleaned.append(c)
            return cleaned
        return []
    except json.JSONDecodeError:
        return []


# ------------------------------------------------------------------
# Output record format
# ------------------------------------------------------------------

def format_output_record(resource: dict, concepts: list[str], raw_output: str) -> dict:
    return {
        "resource_id":   resource["resource_id"],
        "document_id":   resource.get("document_id"),
        "resource_type": resource.get("resource_type"),
        "subject":       resource.get("subject"),
        "class_no":      resource.get("class_no"),
        "chapter_no":    resource.get("chapter_no"),
        "chapter_title": resource.get("chapter_title"),
        "section":       resource.get("section"),
        "concepts":      concepts
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EduExplain concept extraction pipeline (MAHA + NIOS)"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to a single chapter JSON or a directory (e.g. data_resources_2/bio_nios)"
    )
    parser.add_argument(
        "--output", "-o",
        default=str(RAW_CONCEPTS_PATH),
        help="Output JSON file (default: vocab/raw_concepts.json)"
    )
    parser.add_argument(
        "--max_resources", type=int, default=None,
        help="Limit number of resources to process (for quick validation)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.0,
        help="Delay in seconds between LLM calls"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output, skipping already-processed resource_ids"
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Resume: load existing results ---
    if args.resume and output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        results: dict[str, dict] = {r["resource_id"]: r for r in existing}
        print(f"Resuming: {len(results)} resource_ids already processed.")
    else:
        results: dict[str, dict] = {}

    # --- Load & filter resources ---
    input_path = Path(args.input)
    print(f"\nLoading resources from: {input_path}")
    all_resources = load_resources_from_path(input_path)
    print(f"Total resources loaded : {len(all_resources)}")

    to_process = filter_resources(all_resources)

    # Skip already processed (resume)
    if results:
        to_process = [r for r in to_process if r["resource_id"] not in results]

    if args.max_resources:
        to_process = to_process[:args.max_resources]

    print(f"Resources to process   : {len(to_process)}\n")

    if not to_process:
        print("Nothing to process. Exiting.")
        return

    # --- Load model ---
    tokenizer, model = load_model(MODEL_NAME)

    # --- Extraction loop ---
    errors = []
    t_start = datetime.now()

    for i, resource in enumerate(to_process, 1):
        rid = resource["resource_id"]
        rtype = resource.get("resource_type", "?")
        print(f"  [{i:>3}/{len(to_process)}] {rid} ({rtype})", end=" ... ", flush=True)

        content = get_extraction_content(resource)
        user_prompt = build_user_prompt(resource, content)

        try:
            raw_output = generate_response(tokenizer, model, SYSTEM_PROMPT, user_prompt)
            concepts = parse_concept_list(raw_output)
        except Exception as e:
            print(f"ERROR: {e}")
            errors.append({"resource_id": rid, "error": str(e)})
            raw_output = ""
            concepts = []

        results[rid] = format_output_record(resource, concepts, raw_output)

        n = len(concepts)
        preview = concepts[:3]
        suffix = "..." if n > 3 else ""
        print(f"{n} concepts: {preview}{suffix}")

        # Crash-safe: save after every resource
        output_path.write_text(
            json.dumps(list(results.values()), indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        if args.delay > 0:
            time.sleep(args.delay)

    # --- Write log ---
    duration = (datetime.now() - t_start).seconds
    log = {
        "run_at":           datetime.now().isoformat(),
        "input_path":       str(input_path),
        "output_path":      str(output_path),
        "duration_seconds": duration,
        "total_loaded":     len(all_resources),
        "total_processed":  len(to_process),
        "total_in_output":  len(results),
        "total_errors":     len(errors),
        "errors":           errors,
    }
    EXTRACTION_LOG_PATH.write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nDone. {len(to_process)} processed, {len(errors)} errors.")
    print(f"Output  : {output_path}")
    print(f"Log     : {EXTRACTION_LOG_PATH}")


if __name__ == "__main__":
    main()
