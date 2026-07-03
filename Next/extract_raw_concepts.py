"""
extract_raw_concepts.py
========================
Stage 2: LLM concept extraction from resource summaries.

Reads:
    data_resources/**/*.json          resource files with summary fields
    vocab/structural_concepts_filtered.json

Writes:
    vocab/raw_extractions.json        per-resource concept extractions
    vocab/extraction_log.json         run statistics and error log

Resumable: skips resources whose resource_id already appears in
raw_extractions.json. Safe to Ctrl+C and restart.

Usage:
    # Full run
    python extract_raw_concepts.py

    # Test on 20 resources (Colab)
    python extract_raw_concepts.py --max_resources 20

    # Single subject
    python extract_raw_concepts.py --include class_11_phy
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
RESOURCES_DIR         = Path("data_resources")
STRUCTURAL_VOCAB_PATH = Path("vocab/structural_concepts_filtered.json")
RAW_EXTRACTIONS_PATH  = Path("vocab/raw_extractions.json")
EXTRACTION_LOG_PATH   = Path("vocab/extraction_log.json")

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# ------------------------------------------------------------------
# Prompt
# ------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert in NCERT science education for Classes 11 and 12.
Your task is to identify scientific concepts present in a resource summary.
You must return ONLY a JSON array of strings. No explanation. No markdown fences.\
"""

def build_user_prompt(resource: dict, structural_vocab: dict) -> str:
    vocab_display = sorted(
        v["display_name"] for v in structural_vocab.values()
    )
    vocab_str = "\n".join(f"  - {name}" for name in vocab_display[:150])

    section_line = ""
    if resource.get("section"):
        section_line = f"Section: {resource['section']}\n"
    if resource.get("subsection"):
        section_line += f"Subsection: {resource['subsection']}\n"

    return f"""\
Extract the key scientific concepts from the resource summary below.

KNOWN VOCABULARY (prefer these exact terms when applicable):
{vocab_str}

RESOURCE METADATA:
Subject: {resource.get('subject', '')}
Class: {resource.get('class_no', '')}
Chapter: {resource.get('chapter_title', '')}
{section_line}Resource type: {resource.get('resource_type', '')}

RESOURCE SUMMARY:
{resource.get('summary', '').strip()}

INSTRUCTIONS:
- Return 3 to 8 key scientific concepts found in the summary.
- Prefer exact terms from the Known Vocabulary when they match.
- You may propose new terms not in the vocabulary if they are important
  scientific concepts clearly present in the summary.
- Do NOT include: generic words (science, physics, chapter, concept),
  the chapter title itself, vague terms (important, basic, fundamental).
- Return ONLY a JSON array of strings.

Example output: ["Static Friction", "Coefficient of Friction", "Normal Force"]

Your output:"""


# ------------------------------------------------------------------
# Model loader
# ------------------------------------------------------------------

def load_model(model_name: str):
    print(f"Loading model: {model_name}")
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
    max_new_tokens: int = 256,
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
    # Decode only the newly generated tokens
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ------------------------------------------------------------------
# Parsing LLM output
# ------------------------------------------------------------------

def parse_concept_list(raw_output: str) -> list[str]:
    """
    Extract a JSON array from LLM output.
    Handles: clean JSON, markdown fences, leading text before the array.
    Returns empty list on failure.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", raw_output)
    text = re.sub(r"```", "", text)

    # Find the first [ ... ] block
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        return []

    try:
        result = json.loads(match.group())
        if isinstance(result, list):
            return [str(c).strip() for c in result if isinstance(c, str) and c.strip()]
        return []
    except json.JSONDecodeError:
        return []


# ------------------------------------------------------------------
# Resource loading
# ------------------------------------------------------------------

def load_all_resources(
    resources_dir: Path,
    include: list[str] | None = None,
) -> list[dict]:
    resources = []
    for chapter_file in sorted(resources_dir.rglob("*.json")):
        # Filter by subject folder if --include specified
        if include:
            if not any(inc in str(chapter_file) for inc in include):
                continue
        try:
            data = json.loads(chapter_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for r in data:
                    r["_source_file"] = str(chapter_file)
                resources.extend(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [WARN] Could not read {chapter_file}: {e}")
    return resources


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_resources", type=int, default=None,
                        help="Limit for testing (e.g. 20 for Colab)")
    parser.add_argument("--include", nargs="*", default=None,
                        help="Only process folders containing these strings")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Delay between generations (seconds)")
    args = parser.parse_args()

    # Load structural vocab
    structural_vocab = json.loads(STRUCTURAL_VOCAB_PATH.read_text(encoding="utf-8"))
    print(f"Structural vocab size: {len(structural_vocab)}")

    # Load existing extractions (for resumability)
    if RAW_EXTRACTIONS_PATH.exists():
        raw_extractions = json.loads(RAW_EXTRACTIONS_PATH.read_text(encoding="utf-8"))
        print(f"Resuming: {len(raw_extractions)} resources already processed")
    else:
        raw_extractions = {}
    RAW_EXTRACTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load resources
    all_resources = load_all_resources(RESOURCES_DIR, include=args.include)
    print(f"Total resources found: {len(all_resources)}")

    # Filter: skip already processed, skip missing summaries
    to_process = [
        r for r in all_resources
        if r.get("resource_id")
        and r["resource_id"] not in raw_extractions
        and (r.get("summary") or "").strip()
    ]

    if args.max_resources:
        to_process = to_process[:args.max_resources]

    print(f"Resources to process: {len(to_process)}")

    if not to_process:
        print("Nothing to process. Exiting.")
        return

    # Load model
    tokenizer, model = load_model(MODEL_NAME)

    # Extraction loop
    errors = []
    t_start = datetime.now()

    for i, resource in enumerate(to_process, 1):
        rid = resource["resource_id"]
        print(f"  [{i}/{len(to_process)}] {rid}", end=" ... ", flush=True)

        user_prompt = build_user_prompt(resource, structural_vocab)

        try:
            raw_output = generate_response(tokenizer, model, SYSTEM_PROMPT, user_prompt)
            concepts = parse_concept_list(raw_output)
        except Exception as e:
            print(f"ERROR: {e}")
            errors.append({"resource_id": rid, "error": str(e)})
            concepts = []

        raw_extractions[rid] = {
            "concepts":   concepts,
            "raw_output": raw_output if concepts else "",
            "source_file": resource.get("_source_file", ""),
        }

        print(f"{len(concepts)} concepts: {concepts[:3]}{'...' if len(concepts) > 3 else ''}")

        # Save after every resource
        RAW_EXTRACTIONS_PATH.write_text(
            json.dumps(raw_extractions, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        if args.delay > 0:
            time.sleep(args.delay)

    # Write log
    log = {
        "run_at":          datetime.now().isoformat(),
        "duration_seconds": (datetime.now() - t_start).seconds,
        "total_processed": len(to_process),
        "total_errors":    len(errors),
        "errors":          errors,
        "total_in_file":   len(raw_extractions),
    }
    EXTRACTION_LOG_PATH.write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nDone. {len(to_process)} processed, {len(errors)} errors.")
    print(f"Output: {RAW_EXTRACTIONS_PATH}")


if __name__ == "__main__":
    main()
