"""
annotate_resources.py
======================
Stage 4: LLM annotation pass.

For each resource, the LLM sees:
    - resource summary
    - a CANDIDATE SUBSET of the vocabulary (not the full vocab)

and decides which concepts from that subset are actually present.

Candidate subset generation:
    1. Start with all vocabulary concepts whose canonical_id appears
       in the resource's chapter_title / section / subsection fields
       (structural match — zero cost, high precision).
    2. Add vocabulary concepts whose display_name has token overlap
       with the resource summary (BM25-style term matching).
    3. Cap at MAX_CANDIDATES concepts sent to the LLM.

This avoids sending hundreds of vocabulary terms per resource while
ensuring the LLM sees all plausibly relevant candidates.

Reads:
    data_resources/**/*.json          (with summary fields)
    vocab/extended_concepts.json

Writes:
    data_resources/**/*.json          (updated in-place: adds concepts field)
    vocab/annotation_log.json

Each resource gains:
    "concepts":      ["static_friction", "coefficient_of_friction"]   canonical IDs
    "concepts_raw":  ["Static Friction", "Coefficient of Friction"]   display names

Usage:
    python annotate_resources.py
    python annotate_resources.py --max_resources 20   # Colab test
    python annotate_resources.py --include class_11_phy
    python annotate_resources.py --max_candidates 30
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
RESOURCES_DIR      = Path("data_resources")
EXTENDED_VOCAB_PATH = Path("vocab/extended_concepts.json")
ANNOTATION_LOG_PATH = Path("vocab/annotation_log.json")

MODEL_NAME       = "Qwen/Qwen2.5-7B-Instruct"
MAX_CANDIDATES   = 40   # max vocab entries shown to LLM per resource
SENTINEL_FIELD   = "concepts"  # if this field exists, resource is already annotated


# ------------------------------------------------------------------
# Prompt
# ------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are annotating educational resources from NCERT science textbooks.
Your task: given a resource summary and a list of candidate concepts,
identify which concepts are genuinely present in the summary.
Return ONLY a JSON array of strings from the candidate list.
Do not invent new concepts. Do not include concepts not in the list.\
"""


def build_annotation_prompt(resource: dict, candidates: list[str]) -> str:
    candidate_str = "\n".join(f"  - {c}" for c in candidates)
    return f"""\
RESOURCE METADATA:
Subject: {resource.get('subject', '')}
Class: {resource.get('class_no', '')}
Chapter: {resource.get('chapter_title', '')}
Section: {resource.get('section') or ''}
Resource type: {resource.get('resource_type', '')}

RESOURCE SUMMARY:
{resource.get('summary', '').strip()}

CANDIDATE CONCEPTS (choose only from this list):
{candidate_str}

TASK:
Return a JSON array containing ONLY the concepts from the candidate list
that are clearly present or discussed in the summary above.
Include a concept if it is explicitly mentioned or directly implied.
Exclude concepts that are only tangentially related or not relevant.

Return ONLY a JSON array. Example: ["Static Friction", "Normal Force"]

Your output:"""


# ------------------------------------------------------------------
# Candidate generation
# ------------------------------------------------------------------

def tokenize_simple(text: str) -> set[str]:
    """Lowercase word tokens, length >= 3."""
    return {
        w.lower() for w in re.findall(r"[a-zA-Z]+", text)
        if len(w) >= 3
    }


def get_candidate_concepts(
    resource: dict,
    vocab: dict,          # canonical_id -> {display_name, frequency, source}
    max_candidates: int,
) -> list[str]:
    """
    Return up to max_candidates display_names from vocab that are
    plausibly relevant to this resource.

    Priority:
        1. Structural match: vocab term appears in section/subsection/chapter
        2. Token overlap: vocab display_name shares tokens with summary
        3. Fill remaining slots from structural vocab by frequency
    """
    resource_text = " ".join(filter(None, [
        resource.get("chapter_title", ""),
        resource.get("section", ""),
        resource.get("subsection", ""),
        resource.get("summary", ""),
    ]))
    resource_tokens = tokenize_simple(resource_text)

    scores: dict[str, float] = {}

    for cid, entry in vocab.items():
        display = entry["display_name"]
        display_tokens = tokenize_simple(display)
        overlap = len(display_tokens & resource_tokens)

        # Score: overlap count, boosted for structural matches
        score = overlap
        # Boost if the display_name appears verbatim in resource_text
        if display.lower() in resource_text.lower():
            score += 5
        if score > 0:
            scores[cid] = score

    # Sort by score descending, take top max_candidates
    sorted_cids = sorted(scores.keys(), key=lambda c: -scores[c])[:max_candidates]

    # If we have fewer than max_candidates, pad with high-frequency structural concepts
    if len(sorted_cids) < max_candidates:
        existing = set(sorted_cids)
        structural_by_freq = sorted(
            [(cid, e) for cid, e in vocab.items()
             if e.get("source") == "structural" and cid not in existing],
            key=lambda x: -x[1].get("frequency", 0)
        )
        pad_count = max_candidates - len(sorted_cids)
        sorted_cids += [cid for cid, _ in structural_by_freq[:pad_count]]

    return [vocab[cid]["display_name"] for cid in sorted_cids if cid in vocab]


# ------------------------------------------------------------------
# Parse LLM output
# ------------------------------------------------------------------

def parse_annotation_output(
    raw_output: str,
    valid_candidates: set[str],
) -> tuple[list[str], list[str]]:
    """
    Parse LLM output into (display_names, canonical_ids).
    Only accepts terms that were in the candidate list.
    """
    text = re.sub(r"```(?:json)?\s*", "", raw_output)
    text = re.sub(r"```", "", text).strip()

    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        return [], []

    try:
        parsed = json.loads(match.group())
        if not isinstance(parsed, list):
            return [], []
    except json.JSONDecodeError:
        return [], []

    # Only accept terms that were in the candidate list (case-insensitive)
    valid_lower = {c.lower(): c for c in valid_candidates}
    accepted_display = []
    for term in parsed:
        if isinstance(term, str):
            term_lower = term.strip().lower()
            if term_lower in valid_lower:
                accepted_display.append(valid_lower[term_lower])

    return accepted_display


def canonicalize(term: str) -> str:
    t = term.strip().lower()
    t = t.replace("'s", "s").replace("'s", "s").replace("'", "")
    t = t.replace("-", " ")
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", "_", t.strip())
    return t


# ------------------------------------------------------------------
# Resource I/O
# ------------------------------------------------------------------

def load_chapter_file(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_chapter_file(path: Path, resources: list[dict]) -> None:
    path.write_text(
        json.dumps(resources, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def collect_unannotated(
    resources_dir: Path,
    include: list[str] | None,
    max_resources: int | None,
) -> list[tuple[Path, int, dict]]:
    """
    Returns list of (chapter_file_path, index_in_file, resource_dict)
    for resources that do not yet have a `concepts` field.
    """
    unannotated = []
    for chapter_file in sorted(resources_dir.rglob("*.json")):
        if include and not any(inc in str(chapter_file) for inc in include):
            continue
        resources = load_chapter_file(chapter_file)
        for idx, r in enumerate(resources):
            if (r.get("resource_id") and (not r.get("concepts")) and (r.get("summary") or "").strip()):
                unannotated.append((chapter_file, idx, r))
                if max_resources and len(unannotated) >= max_resources:
                    return unannotated
    return unannotated


# ------------------------------------------------------------------
# Model
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


def generate(tokenizer, model, system: str, user: str, max_new_tokens=256) -> str:
    messages = [{"role": "system", "content": system},
                {"role": "user",   "content": user}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_resources",  type=int,   default=None)
    parser.add_argument("--include",        nargs="*",  default=None)
    parser.add_argument("--max_candidates", type=int,   default=MAX_CANDIDATES)
    args = parser.parse_args()

    vocab = json.loads(EXTENDED_VOCAB_PATH.read_text(encoding="utf-8"))
    print(f"Vocabulary size: {len(vocab)}")

    unannotated = collect_unannotated(
        RESOURCES_DIR, args.include, args.max_resources
    )
    print(f"Resources to annotate: {len(unannotated)}")
    if not unannotated:
        print("Nothing to annotate.")
        return

    tokenizer, model = load_model(MODEL_NAME)

    errors = []
    t_start = datetime.now()

    # Group by file to minimize file reads/writes
    by_file: dict[Path, list[tuple[int, dict]]] = defaultdict(list)
    for fpath, idx, resource in unannotated:
        by_file[fpath].append((idx, resource))

    processed = 0
    for fpath, items in by_file.items():
        chapter_resources = load_chapter_file(fpath)
        changed = False

        for idx, resource in items:
            rid = resource["resource_id"]
            print(f"  [{processed+1}/{len(unannotated)}] {rid}", end=" ... ", flush=True)

            candidates = get_candidate_concepts(resource, vocab, args.max_candidates)
            if not candidates:
                chapter_resources[idx]["concepts"]     = []
                chapter_resources[idx]["concepts_raw"] = []
                processed += 1
                changed = True
                print("0 candidates, skipped")
                continue

            prompt = build_annotation_prompt(resource, candidates)
            try:
                raw_output = generate(tokenizer, model, SYSTEM_PROMPT, prompt)
                accepted_display = parse_annotation_output(
                    raw_output, set(candidates)
                )
            except Exception as e:
                print(f"ERROR: {e}")
                errors.append({"resource_id": rid, "error": str(e)})
                accepted_display = []

            concepts_ids = [canonicalize(d) for d in accepted_display]

            chapter_resources[idx]["concepts"]     = concepts_ids
            chapter_resources[idx]["concepts_raw"] = accepted_display
            changed = True
            processed += 1

            print(f"{len(accepted_display)} concepts: {accepted_display[:3]}"
                  f"{'...' if len(accepted_display) > 3 else ''}")

        if changed:
            save_chapter_file(fpath, chapter_resources)

    log = {
        "run_at":           datetime.now().isoformat(),
        "duration_seconds": (datetime.now() - t_start).seconds,
        "total_processed":  processed,
        "total_errors":     len(errors),
        "errors":           errors,
    }
    ANNOTATION_LOG_PATH.write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nDone. {processed} annotated, {len(errors)} errors.")


if __name__ == "__main__":
    main()
