"""
build_extended_vocab.py
========================
Stage 3: Build the extended concept vocabulary.

Reads:
    vocab/structural_concepts_filtered.json
    vocab/raw_extractions.json

Writes:
    vocab/extended_concepts.json
    vocab/merge_map.json           (alias -> canonical, for audit)
    vocab/rejected_concepts.json   (below-threshold terms, for inspection)

Rules:
    - All structural vocabulary concepts are retained unconditionally.
    - New concepts from raw_extractions are added only if they appear
      in >= FREQUENCY_THRESHOLD resources (default 3).
    - Before frequency counting, raw terms are normalized to catch
      near-duplicates ("Centre of Mass" / "Center of Mass").
    - A merge_map records every normalization decision for auditability.

Usage:
    python build_extended_vocab.py
    python build_extended_vocab.py --threshold 5
    python build_extended_vocab.py --threshold 2 --dry_run
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
STRUCTURAL_VOCAB_PATH  = Path("vocab/structural_concepts_filtered.json")
RAW_EXTRACTIONS_PATH   = Path("vocab/raw_extractions.json")
EXTENDED_VOCAB_PATH    = Path("vocab/extended_concepts.json")
MERGE_MAP_PATH         = Path("vocab/merge_map.json")
REJECTED_PATH          = Path("vocab/rejected_concepts.json")

DEFAULT_THRESHOLD = 3


# ------------------------------------------------------------------
# Normalization
# ------------------------------------------------------------------

def canonicalize(term: str) -> str:
    """
    Convert display-form concept to a canonical ID string.
    "Newton's First Law" -> "newtons_first_law"
    Deterministic: same input always produces same output.
    """
    t = term.strip().lower()
    t = t.replace("'s", "s").replace("'s", "s").replace("'", "")
    t = t.replace("-", " ")
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", "_", t.strip())
    return t


def normalize_display(term: str) -> str:
    """Title-case normalization for display names."""
    # Preserve all-caps acronyms (DNA, ATP, NaCl)
    words = term.strip().split()
    result = []
    for w in words:
        if w.isupper() and len(w) > 1:
            result.append(w)
        else:
            result.append(w.capitalize())
    return " ".join(result)


def string_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def are_near_duplicates(a: str, b: str) -> bool:
    """
    Return True if two display-form terms likely refer to the same concept.
    Conservative — only catches clear cases to avoid false merges.
    """
    a_norm = re.sub(r"[^a-z0-9 ]", "", a.lower().replace("'s", "s"))
    b_norm = re.sub(r"[^a-z0-9 ]", "", b.lower().replace("'s", "s"))

    # Exact after normalization
    if a_norm == b_norm:
        return True

    # One is a contained substring of the other (short ones may be abbreviations)
    if len(a_norm) > 5 and a_norm in b_norm:
        return True
    if len(b_norm) > 5 and b_norm in a_norm:
        return True

    # British/American spelling variants
    variants = [("centre", "center"), ("colour", "color"), ("fibre", "fiber")]
    for brit, amer in variants:
        if a_norm.replace(brit, amer) == b_norm or a_norm == b_norm.replace(brit, amer):
            return True

    # High string similarity
    if string_similarity(a_norm, b_norm) > 0.92:
        return True

    return False


# ------------------------------------------------------------------
# Deduplication within a set of terms
# ------------------------------------------------------------------

def deduplicate_terms(
    terms_with_counts: dict[str, int],  # display_name -> count
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Deduplicate a set of display-form terms.

    Returns:
        canonical_map: display_name -> canonical display_name (chosen representative)
        merge_map:     alias display_name -> canonical display_name
    """
    terms = list(terms_with_counts.keys())
    # Sort by count descending so the most frequent form becomes canonical
    terms.sort(key=lambda t: terms_with_counts.get(t, 0), reverse=True)

    canonical_map: dict[str, str] = {}  # term -> its canonical representative
    merge_map:     dict[str, str] = {}  # alias -> canonical (only merged ones)

    for term in terms:
        if term in canonical_map:
            continue  # already assigned

        # This term becomes its own canonical representative
        canonical_map[term] = term

        # Find all other terms that are near-duplicates
        for other in terms:
            if other == term or other in canonical_map:
                continue
            if are_near_duplicates(term, other):
                canonical_map[other] = term
                merge_map[other] = term

    return canonical_map, merge_map


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                        help="Minimum resource count for new concepts (default 3)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print stats without writing files")
    args = parser.parse_args()

    # Load inputs
    structural_vocab = json.loads(STRUCTURAL_VOCAB_PATH.read_text(encoding="utf-8"))
    raw_extractions  = json.loads(RAW_EXTRACTIONS_PATH.read_text(encoding="utf-8"))

    print(f"Structural vocab entries : {len(structural_vocab)}")
    print(f"Resources with extractions: {len(raw_extractions)}")
    print(f"Frequency threshold      : {args.threshold}")

    # Build set of structural display names (lowercased) for fast lookup
    structural_display_lower: set[str] = {
        v["display_name"].lower()
        for v in structural_vocab.values()
    }
    structural_canonical_ids: set[str] = set(structural_vocab.keys())

    # ── Step 1: Collect all raw extracted terms with per-resource counts ──
    # Count: how many distinct resources mention each term
    term_resource_count: Counter = Counter()
    # We need per-resource dedup: term should count once per resource
    for rid, extraction in raw_extractions.items():
        concepts = extraction.get("concepts", [])
        seen_in_resource: set[str] = set()
        for c in concepts:
            c_lower = c.strip().lower()
            if c_lower and c_lower not in seen_in_resource:
                term_resource_count[c.strip()] += 1
                seen_in_resource.add(c_lower)

    print(f"Unique raw terms extracted: {len(term_resource_count)}")

    # ── Step 2: Separate structural matches from new candidates ──────────
    # A term "matches" the structural vocab if it canonicalizes to an
    # existing structural canonical ID, OR if its lowercase display name
    # appears in the structural display names.

    new_candidates: dict[str, int] = {}  # display_name -> resource count

    for term, count in term_resource_count.items():
        term_canonical = canonicalize(term)
        term_lower     = term.lower()

        in_structural = (
            term_canonical in structural_canonical_ids
            or term_lower in structural_display_lower
        )
        if not in_structural:
            new_candidates[term] = count

    print(f"New candidate terms      : {len(new_candidates)}")

    # ── Step 3: Deduplicate new candidates ───────────────────────────────
    canonical_map, merge_map_display = deduplicate_terms(new_candidates)

    # Merge counts: for each canonical representative, sum counts of all aliases
    canonical_counts: Counter = Counter()
    for term, canon in canonical_map.items():
        canonical_counts[canon] += new_candidates.get(term, 0)

    print(f"After dedup              : {len(canonical_counts)} unique candidates")

    # ── Step 4: Apply frequency threshold ────────────────────────────────
    accepted: dict[str, int] = {}
    rejected: dict[str, int] = {}

    for term, count in canonical_counts.items():
        if count >= args.threshold:
            accepted[term] = count
        else:
            rejected[term] = count

    print(f"Accepted (>= {args.threshold})         : {len(accepted)}")
    print(f"Rejected (< {args.threshold})          : {len(rejected)}")

    # ── Step 5: Build extended vocabulary ────────────────────────────────
    extended_vocab: dict[str, dict] = {}

    # Add all structural concepts first (always retained)
    for cid, entry in structural_vocab.items():
        extended_vocab[cid] = {
            "display_name": entry["display_name"],
            "frequency":    entry["frequency"],
            "source":       "structural",
        }

    # Add accepted new concepts
    for term, count in sorted(accepted.items(), key=lambda x: -x[1]):
        cid = canonicalize(term)
        if cid in extended_vocab:
            # Collision with structural: skip (structural takes priority)
            continue
        extended_vocab[cid] = {
            "display_name": normalize_display(term),
            "frequency":    count,
            "source":       "extracted",
        }

    print(f"Extended vocab total     : {len(extended_vocab)}")

    # ── Step 6: Build full merge map (canonical_id -> canonical_id) ──────
    # Maps alias display_name -> canonical display_name for audit
    merge_map_out: dict[str, str] = dict(merge_map_display)

    if not args.dry_run:
        Path("vocab").mkdir(parents=True, exist_ok=True)
        EXTENDED_VOCAB_PATH.write_text(
            json.dumps(extended_vocab, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        MERGE_MAP_PATH.write_text(
            json.dumps(merge_map_out, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        REJECTED_PATH.write_text(
            json.dumps(
                {t: c for t, c in sorted(rejected.items(), key=lambda x: -x[1])},
                indent=2, ensure_ascii=False
            ),
            encoding="utf-8"
        )
        print(f"\nWrote: {EXTENDED_VOCAB_PATH}")
        print(f"Wrote: {MERGE_MAP_PATH}")
        print(f"Wrote: {REJECTED_PATH}")
    else:
        print("\n[DRY RUN] No files written.")

    # ── Step 7: Preview ───────────────────────────────────────────────────
    print("\nTop 10 new concepts by frequency:")
    for term, count in sorted(accepted.items(), key=lambda x: -x[1])[:10]:
        print(f"  {count:4d}  {normalize_display(term)}")

    if merge_map_out:
        print(f"\nSample merges ({min(5, len(merge_map_out))} of {len(merge_map_out)}):")
        for alias, canon in list(merge_map_out.items())[:5]:
            print(f"  '{alias}' -> '{canon}'")


if __name__ == "__main__":
    main()
