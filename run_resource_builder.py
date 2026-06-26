import json
import os
from src.resources.resource_builder import ResourceBuilder
from src.resources import resource_types

# 1. Dynamically update the resource type mappings to understand NIOS chunks
resource_types.CHUNK_TYPE_TO_RESOURCE_TYPE.update({
    "content": "explanation",            # NIOS standard text
    "terminal_exercise": "exercise",     # NIOS end-of-chapter questions
    "answers": "explanation",            # NIOS answer key
})

def preprocess_nios_chunks(chunks):
    """
    PREPROCESSING STAGE
    The Resource schemas strictly require fields like `class_no`. Our NIOS parser didn't
    hardcode a class_no, so we inject missing required fields here before building.
    """
    for chunk in chunks:
        # Inject standard required fields if missing
        if "class_no" not in chunk:
            chunk["class_no"] = 12  # Put a default class number here, or infer if possible
        
        # Ensure chapter_no exists and is an integer
        if "chapter_no" not in chunk or chunk["chapter_no"] is None:
             chunk["chapter_no"] = 0
             
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Build LLM resources from chunked Data.")
    parser.add_argument("--input", "-i", type=str, default="output/Phy_NIOS_chunks.json", help="Path to parsed chunks JSON")
    parser.add_argument("--raw", "-r", type=str, default="Phy_NIOS.json", help="Path to raw marker extraction JSON")
    parser.add_argument("--output", "-o", type=str, default="output/Phy_NIOS_resources.json", help="Output path for final resources")
    parser.add_argument("--subject", type=str, default="NIOS_PHY", help="Document Identifier")
    parser.add_argument("--class-no", type=int, default=12, help="Class number to embed")
    args = parser.parse_args()

    print(f"==================================================")
    print(f" RESOURCE BUILDER PIPELINE")
    print(f"==================================================")
    print(f"[*] Input Chunks: {args.input}")
    print(f"[*] Raw Source:   {args.raw}")
    print(f"[*] Output:       {args.output}")
    print(f"[*] Identity:     {args.subject} (Class {args.class_no})")
    print(f"--------------------------------------------------")

    if not os.path.exists(args.input):
        print(f"[!] Error: Chunk file {args.input} not found.")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        parsed_chunks = json.load(f)
        
    raw_extraction = {}
    if os.path.exists(args.raw):
        with open(args.raw, "r", encoding="utf-8") as f:
            raw_extraction = json.load(f)
    else:
        print(f"[-] Warning: Raw extraction {args.raw} not found. Equation mapping will use weak regex fallback.")
            
    # Inject command line class_no
    for chunk in parsed_chunks:
        if "class_no" not in chunk:
            chunk["class_no"] = args.class_no
        if chunk.get("chapter_no") is None:
             chunk["chapter_no"] = 0

    print(f"[*] Read {len(parsed_chunks)} logic chunks.")
    print(f"[*] Preprocessing complete. Initializing Builder...")

    builder = ResourceBuilder(
        document_id=args.subject,
        raw_extraction=raw_extraction
    )
    
    resources = builder.build_from_chunks(parsed_chunks)
    
    print(f"[*] Evaluated Schemas. {len(resources)} Resources fully compiled.")
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in resources], f, indent=2, ensure_ascii=False)
        
    print(f"[+] Success! Compiled objects saved to: {args.output}")
    print(f"==================================================")

if __name__ == "__main__":
    main()
