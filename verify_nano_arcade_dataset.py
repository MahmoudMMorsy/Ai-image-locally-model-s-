#!/usr/bin/env python3
"""
verify_nano_arcade_dataset.py

Verification script for `nano_arcade_data/`.
Checks:
1. Directory structure matches taxonomy:
   nano_arcade_data / [platform] / [poster_type] / [language_structure] / [typography_style] / [art_style] / [resolution] /
2. Allowed image extensions only (PNG, JPG, JPEG, GIF).
3. ZERO metadata files (JSON, CSV, TXT, XML) in nano_arcade_data/.
4. Resolution scale strictly 128x128 or 256x256 matching folder name.
5. Total file count > 0.
"""

import os
import sys
from PIL import Image

TARGET_DIR = "nano_arcade_data"

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif"}
FORBIDDEN_EXTENSIONS = {".json", ".csv", ".txt", ".xml"}

ALLOWED_RESOLUTIONS = {"128x128", "256x256"}

def verify():
    if not os.path.exists(TARGET_DIR):
        print(f"Error: Target directory '{TARGET_DIR}' does not exist.")
        sys.exit(1)

    total_files = 0
    errors = []

    for root, dirs, files in os.walk(TARGET_DIR):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            file_path = os.path.join(root, f)

            if ext in FORBIDDEN_EXTENSIONS:
                errors.append(f"Forbidden metadata file found: {file_path}")
                continue

            if ext not in VALID_EXTENSIONS:
                errors.append(f"Invalid file extension: {file_path}")
                continue

            total_files += 1

            # Verify folder depth relative to TARGET_DIR
            rel_path = os.path.relpath(file_path, TARGET_DIR)
            parts = rel_path.split(os.sep)

            # Expected depth: platform / poster_type / lang_struct / typo_style / art_style / resolution / filename
            if len(parts) != 7:
                errors.append(f"Invalid folder hierarchy depth ({len(parts)} parts): {rel_path}")

            res_folder = parts[5] if len(parts) >= 6 else ""
            if res_folder not in ALLOWED_RESOLUTIONS:
                errors.append(f"Invalid resolution folder '{res_folder}': {rel_path}")

            # Verify actual image dimensions
            try:
                with Image.open(file_path) as img:
                    w, h = img.size
                    expected_w, expected_h = map(int, res_folder.split("x"))
                    if w != expected_w or h != expected_h:
                        errors.append(f"Image dimension mismatch in {file_path}: actual ({w}x{h}), expected ({expected_w}x{expected_h})")
            except Exception as e:
                errors.append(f"Failed to open image {file_path}: {e}")

    print(f"--- Verification Summary ---")
    print(f"Total verified image files in '{TARGET_DIR}': {total_files}")

    if errors:
        print(f"FAILED: Found {len(errors)} error(s):")
        for err in errors[:20]:
            print(f" - {err}")
        if len(errors) > 20:
            print(f" ... and {len(errors) - 20} more errors.")
        sys.exit(1)
    else:
        print("SUCCESS: `nano_arcade_data/` dataset passed all verification checks!")

if __name__ == "__main__":
    verify()
