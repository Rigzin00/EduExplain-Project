import json

chunks = json.load(open("output/Phy_NIOS_chunks.json", encoding="utf-8"))

with open("audit_result.txt", "w", encoding="utf-8") as f:
    f.write(f"{'#':<4} {'type':<22} {'sec':<8} {'subsec':<10} {'pg':<8} chapter_title\n")
    f.write("-" * 90 + "\n")
    for i, c in enumerate(chunks):
        pg = f"{c['page_start']}-{c['page_end']}"
        f.write(f"[{i+1:02d}] {c['chunk_type']:<22} {str(c['section_no']):<8} {str(c['subsection_no']):<10} {pg:<8} {c['chapter_title']}\n")

    f.write("\n=== Header leak check (looking for 'Units, Dimensions' in text) ===\n")
    for i, c in enumerate(chunks):
        if "Units, Dimensions and Vectors" in c["text"] or "MODULE" in c["text"] or "Motion, Force" in c["text"]:
            f.write(f"  [LEAK] chunk {i+1} ({c['chunk_type']}) pg {c['page_start']}-{c['page_end']}\n")
            snippet = [l for l in c["text"].split("\n") if "Units" in l or "MODULE" in l or "Motion" in l]
            for s in snippet[:3]:
                f.write(f"         >> {repr(s[:80])}\n")

    f.write("\n=== Derived Units check (should have its own chunk) ===\n")
    for i, c in enumerate(chunks):
        sub = c.get("text", "") or ""
        sub_title = c.get("subsection", "") or ""
        if "Derived Units" in sub_title or "Derived" in sub[:50]:
            f.write(f"  chunk {i+1}: subsection={c.get('subsection')} type={c['chunk_type']}\n")
    