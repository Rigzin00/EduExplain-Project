import json
import os

def load_json(filepath):
    """Loads a JSON file if it exists, otherwise returns an empty list."""
    if not os.path.exists(filepath):
        print(f"[!] Warning: File {filepath} not found.")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def merge_multimodal(resources_path, images_path, output_path):
    print(f"[*] Loading text resources from: {resources_path}")
    resources = load_json(resources_path)
    
    print(f"[*] Loading images metadata from: {images_path}")
    images = load_json(images_path)
    
    if not resources:
        print("[!] Error: No resources found to merge. Exiting.")
        return
        
    print(f"[*] Starting multimodal merge for {len(resources)} text resources and {len(images)} images...")
    
    # 1. Create a dictionary to quickly map page_no to contextual metadata from text chunks
    # This allows the image to inherit the chapter, section, and class info of the text around it
    context_map = {}
    for res in resources:
        for p in range(res.get("page_start", 0), res.get("page_end", 0) + 1):
            if p not in context_map:
                context_map[p] = res
                
    # 2. Spawn a separate Resource dict for every valid image
    image_resources = []
    for img in images:
        img_page = img.get("page_no")
        
        # Get context from the text chunk on the same page (or fallback to generic)
        ctx = context_map.get(img_page, resources[0] if resources else {})
        
        # Derive a clean unique Image Resource ID (e.g. IMG_12_CH01_001)
        img_raw_id = img.get("image_id", "001").replace("IMG_", "")
        c_no = ctx.get("class_no", 12)
        ch_no = ctx.get("chapter_no", 0)
        res_id = f"IMG_{c_no:02d}_CH{ch_no:02d}_{img_raw_id}"
        
        image_res = {
            "document_id": ctx.get("document_id", "NIOS_PHY"),
            "resource_id": res_id,
            "resource_type": "diagram",
            "class_no": c_no,
            "subject": ctx.get("subject", "Physics"),
            "chapter_no": ch_no,
            "chapter_title": ctx.get("chapter_title", ""),
            "section": ctx.get("section"),
            "subsection": ctx.get("subsection"),
            "page_start": img_page,
            "page_end": img_page,
            "text": None,
            "image_path": img.get("image_path"),
            "image_caption": img.get("caption"),
            "description": None,
            "summary": None,
            "concepts": [],
            "has_equation": False
        }
        image_resources.append(image_res)
        
    print(f"[*] Generated {len(image_resources)} independent 'diagram' resources.")
    
    # 3. Bundle them all together
    final_dataset = resources + image_resources
    
    # Save the output
    print(f"[*] Saving multimodal dataset to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)
        
    print(f"[+] Done! Total Final Resources: {len(final_dataset)}")

if __name__ == "__main__":
    RESOURCES_PATH = "output/Phy_NIOS_resources.json"
    IMAGES_PATH = "Phy_NIOS_images.json"
    OUTPUT_PATH = "output/Phy_NIOS_multimodal.json"
    
    merge_multimodal(RESOURCES_PATH, IMAGES_PATH, OUTPUT_PATH)
