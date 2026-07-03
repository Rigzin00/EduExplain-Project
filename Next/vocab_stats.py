"""
vocab_stats.py
==============
Diagnostic script. Run after Stage 4 to validate annotation quality
before proceeding to query generation.

Reads:
    data_resources/**/*.json          (annotated resources)
    vocab/extended_concepts.json
    vocab/structural_concepts_filtered.json
    vocab/raw_extractions.json        (optional)
    vocab/rejected_concepts.json      (optional)

Outputs to stdout. No files written.

Usage:
    python vocab_stats.py
    python vocab_stats.py --include class_11_phy
    python vocab_stats.py --output_json vocab/stats_report.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


RESOURCES_DIR          = Path("data_resources")
EXTENDED_VOCAB_PATH    = Path("vocab/extended_concepts.json")
STRUCTURAL_VOCAB_PATH  = Path("vocab/structural_concepts_filtered.json")
RAW_EXTRACTIONS_PATH   = Path("vocab/raw_extractions.json")
REJECTED_PATH          = Path("vocab/rejected_concepts.json")


def load_all_resources(
    resources_dir: Path,
    include: list[str] | None,
) -> list[dict]:
    resources = []
    for f in sorted(resources_dir.rglob("*.json")):
        if include and not any(inc in str(f) for inc in include):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, list):
                resources.extend(data)
        except (json.JSONDecodeError, OSError):
            pass
    return resources


def section(title: str) -> None:
    print(f"\n── {title} {'─' * max(0, 55 - len(title))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include", nargs="*", default=None)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    # Load data
    vocab      = json.loads(EXTENDED_VOCAB_PATH.read_text(encoding="utf-8"))
    struct_v   = json.loads(STRUCTURAL_VOCAB_PATH.read_text(encoding="utf-8"))
    resources  = load_all_resources(RESOURCES_DIR, args.include)

    raw_ext    = {}
    if RAW_EXTRACTIONS_PATH.exists():
        raw_ext = json.loads(RAW_EXTRACTIONS_PATH.read_text(encoding="utf-8"))

    rejected   = {}
    if REJECTED_PATH.exists():
        rejected = json.loads(REJECTED_PATH.read_text(encoding="utf-8"))

    total = len(resources)
    stats = {}

    # ── 1. Corpus overview ─────────────────────────────────────────────
    section("Corpus Overview")
    annotated   = [r for r in resources if "concepts" in r]
    unannotated = [r for r in resources if "concepts" not in r]
    no_summary  = [r for r in resources if not r.get("summary", "").strip()]

    print(f"  Total resources          : {total}")
    print(f"  Annotated (has concepts) : {len(annotated)} ({100*len(annotated)/max(total,1):.1f}%)")
    print(f"  Unannotated              : {len(unannotated)}")
    print(f"  No summary               : {len(no_summary)}")
    stats["total"] = total
    stats["annotated"] = len(annotated)

    # ── 2. Vocabulary overview ─────────────────────────────────────────
    section("Vocabulary Overview")
    structural_ids = set(struct_v.keys())
    extracted_ids  = {cid for cid, e in vocab.items() if e.get("source") == "extracted"}

    print(f"  Extended vocab size      : {len(vocab)}")
    print(f"  From structural titles   : {len(structural_ids)}")
    print(f"  From LLM extraction      : {len(extracted_ids)}")
    if rejected:
        print(f"  Rejected (below threshold): {len(rejected)}")
    stats["vocab_size"] = len(vocab)

    # ── 3. Concept frequency across annotated resources ────────────────
    section("Concept Frequency Distribution")
    concept_counter: Counter = Counter()
    for r in annotated:
        for cid in r.get("concepts", []):
            concept_counter[cid] += 1

    # Distribution: how many concepts appear in exactly N resources
    freq_dist: Counter = Counter(concept_counter.values())
    print("  (N resources → count of concepts appearing in exactly N resources)")
    thresholds = [1, 2, 3, 5, 10, 20, 50]
    prev = 0
    for t in thresholds:
        count = sum(v for k, v in freq_dist.items() if prev < k <= t)
        print(f"  appears in {prev+1:3d}–{t:3d} resources: {count:5d} concepts")
        prev = t
    remaining = sum(v for k, v in freq_dist.items() if k > thresholds[-1])
    if remaining:
        print(f"  appears in {thresholds[-1]+1:3d}+    resources: {remaining:5d} concepts")

    # ── 4. Singleton concepts (appear in exactly 1 resource) ──────────
    section("Singleton Concepts (appear in 1 resource — potential noise)")
    singletons = [(cid, vocab.get(cid, {}).get("display_name", cid))
                  for cid, c in concept_counter.items() if c == 1]
    print(f"  Count: {len(singletons)}")
    if singletons:
        print("  Sample (first 10):")
        for cid, display in singletons[:10]:
            print(f"    {display}")
    stats["singleton_concepts"] = len(singletons)

    # ── 5. Top 20 most frequent concepts ──────────────────────────────
    section("Top 20 Most Frequent Concepts")
    for cid, count in concept_counter.most_common(20):
        display = vocab.get(cid, {}).get("display_name", cid)
        source  = vocab.get(cid, {}).get("source", "?")
        bar     = "█" * min(count // 2, 30)
        print(f"  {count:5d}  [{source[:6]:6s}]  {display:<40}  {bar}")

    # ── 6. Concepts per resource distribution ─────────────────────────
    section("Concepts Per Resource")
    counts_per_resource = [len(r.get("concepts", [])) for r in annotated]
    if counts_per_resource:
        avg = sum(counts_per_resource) / len(counts_per_resource)
        print(f"  Mean concepts/resource   : {avg:.1f}")
        print(f"  Min                      : {min(counts_per_resource)}")
        print(f"  Max                      : {max(counts_per_resource)}")
        zeros = sum(1 for c in counts_per_resource if c == 0)
        print(f"  Resources with 0 concepts: {zeros} "
              f"({100*zeros/max(len(counts_per_resource),1):.1f}%)")
        stats["avg_concepts_per_resource"] = round(avg, 2)
        stats["resources_with_zero"] = zeros

    # ── 7. Coverage by resource type ──────────────────────────────────
    section("Coverage by Resource Type")
    type_total:  defaultdict = defaultdict(int)
    type_with:   defaultdict = defaultdict(int)
    type_avg:    defaultdict = defaultdict(list)
    for r in resources:
        rt = r.get("resource_type", "unknown")
        type_total[rt] += 1
        if "concepts" in r:
            type_with[rt] += 1
            type_avg[rt].append(len(r.get("concepts", [])))

    print(f"  {'Type':<22} {'Total':>6}  {'Annot':>6}  {'%':>5}  {'Avg C':>6}")
    for rt in sorted(type_total.keys()):
        n   = type_total[rt]
        w   = type_with[rt]
        avg = sum(type_avg[rt]) / max(w, 1)
        print(f"  {rt:<22} {n:>6}  {w:>6}  {100*w/n:>4.0f}%  {avg:>6.1f}")

    # ── 8. Coverage by subject + class ────────────────────────────────
    section("Coverage by Subject + Class")
    subj_total: defaultdict = defaultdict(int)
    subj_annot: defaultdict = defaultdict(int)
    for r in resources:
        key = f"Class {r.get('class_no','?')} {r.get('subject','?')}"
        subj_total[key] += 1
        if "concepts" in r:
            subj_annot[key] += 1

    print(f"  {'Subject + Class':<35} {'Total':>6}  {'Annot':>6}  {'%':>5}")
    for key in sorted(subj_total.keys()):
        n = subj_total[key]
        w = subj_annot[key]
        print(f"  {key:<35} {n:>6}  {w:>6}  {100*w/n:>4.0f}%")

    # ── 9. Unused vocabulary terms ────────────────────────────────────
    section("Unused Vocabulary Terms (never assigned)")
    used_cids   = set(concept_counter.keys())
    unused_cids = set(vocab.keys()) - used_cids
    print(f"  Vocab terms used   : {len(used_cids)}")
    print(f"  Vocab terms unused : {len(unused_cids)}")
    if unused_cids:
        print("  Sample unused (first 10):")
        for cid in list(unused_cids)[:10]:
            display = vocab[cid].get("display_name", cid)
            source  = vocab[cid].get("source", "?")
            print(f"    [{source}] {display}")
    stats["unused_vocab_terms"] = len(unused_cids)

    # ── 10. Stage 2 extraction quality (if raw_extractions available) ─
    if raw_ext:
        section("Stage 2 Extraction Quality")
        empty_extractions = sum(
            1 for v in raw_ext.values()
            if not v.get("concepts")
        )
        all_counts = [len(v.get("concepts", [])) for v in raw_ext.values()]
        print(f"  Resources with extractions  : {len(raw_ext)}")
        print(f"  Empty extractions           : {empty_extractions}")
        if all_counts:
            print(f"  Avg concepts extracted      : {sum(all_counts)/len(all_counts):.1f}")

    # ── 11. Quality gates ─────────────────────────────────────────────
    section("Quality Gate Summary")
    zero_pct = 100 * stats.get("resources_with_zero", 0) / max(len(annotated), 1)
    unused_pct = 100 * stats.get("unused_vocab_terms", 0) / max(len(vocab), 1)

    gates = [
        ("Annotation coverage >= 95%",
         len(annotated) / max(total, 1) >= 0.95),
        ("Resources with 0 concepts < 10%",
         zero_pct < 10),
        ("Unused vocab terms < 30%",
         unused_pct < 30),
        ("Avg concepts/resource >= 2",
         stats.get("avg_concepts_per_resource", 0) >= 2),
    ]
    all_pass = True
    for label, passed in gates:
        symbol = "✓" if passed else "✗"
        print(f"  {symbol}  {label}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n  All quality gates passed. Safe to proceed to query generation.")
    else:
        print("\n  Some gates failed. Review before proceeding.")

    # ── Optional JSON export ───────────────────────────────────────────
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nStats written to {args.output_json}")


if __name__ == "__main__":
    main()
