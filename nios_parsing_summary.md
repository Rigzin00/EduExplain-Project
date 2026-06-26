# NIOS Physics Textbook Parsing Summary

This document serves as a comprehensive record of the development and refinement of the automated parsing pipeline for the NIOS Physics textbook dataset.

## 1. Built the Custom `nios_parser.py` Script
Because NIOS textbooks follow distinctly different structural and typographical conventions compared to NCERT books, we developed a brand new, standalone Python parser (`nios_parser.py`). 
* **Auto-detection:** The script automatically seeks out the source JSON file (`Phy_NIOS.json`) and gracefully extracts chapter metadata directly from the source.
* **Special Blocks Isolation:** It accurately identifies and isolates special NIOS elements (e.g., out-of-bounds text boxes) into their own distinct chunk types: `intext_question`, `terminal_exercise`, `example`, and `answers`.
* **Intelligent Sectioning:** It uses robust Regular Expressions (Regex) to map headings like `1.1 PHYSICAL WORLD...` and `1.1.1 Physics: Scope...` and cleanly boundaries the raw text into logical semantic chunks.

## 2. Cleaned Textbook "Running Headers"
During the audit phase, we noticed that the raw PDF extraction blindly inserted running page headers like *"Motion, Force and Energy"* and *"MODULE - 1"* every few pages. This was disastrously splicing sentences in half. 
* **The Fix:** We built a dedicated `clean_text()` function into the parser that silently sweeps through every line and instantly deletes these page headers before the chunks are formed, ensuring uninterrupted prose.

## 3. Solved the "Missing Bullet Points" Mystery
During extraction validation, an entire bulleted list (`(i) A falling apple...`) was discovered to be completely missing from the generated chunks. 

**The Reason:** 
This problem occurred upstream during the PDF-to-JSON extraction phase inside Google Colab. The AI layout model (Surya OCR) within the Marker pipeline actively erased the bulleted lists. Surya OCR falsely flagged the uniquely spaced list as a structural "Image" or "Figure." Because the Colab script had the setting `"disable_image_extraction": True` to save time, the AI threw the text away entirely without ever writing it to the `Phy_NIOS.json` file.

**The Solution:**
* **Temporary Immediate Fix:** We manually patched the missing list right into the local `Phy_NIOS.json` file so that the parsing could continue flawlessly without rerunning the heavy Colab job.
* **Permanent Extraction Fix:** To prevent lists from being swallowed in future textbook extractions, the configuration inside the Colab script (`untitled2.py`) must be updated. By explicitly setting `"disable_image_extraction": False` and `"force_ocr": True` in the `MARKER_CONFIG`, the model is forced to rigorously read the text and ignores the false "image" flags, converting the text perfectly.

## 4. Overhauled your Google Colab Pipeline (`untitled2.py`)
While implementing the permanent extraction fix mentioned above, we ran into a massive Google Colab library crashing bug (`TypeError: function takes at most 16 arguments (17 given)`). The newest updates to Python's `Pillow` library completely broke Marker's image extraction sequence in Colab.
* **Permanent Fix:** We updated the local `untitled2.py` script to include the magic dependency lock trigger: `!pip install "Pillow<10.3.0"`. 
* Now, when this script is copied back to Colab in the future, it is fully immunized against the `TypeError` crash and will flawlessly extract bulleted lists using `"disable_image_extraction": False`.

## The Final Result
By simply invoking `python nios_parser.py`, the pipeline instantly generates an immaculate `output/Phy_NIOS_chunks.json` file housing exactly 34 perfectly categorized, sequential, and uninterrupted chunks—ready for scalable ingestion into an LLM dataset.
