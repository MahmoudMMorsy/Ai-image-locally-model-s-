#!/usr/bin/env python3
"""
verify_nano_pixel_dataset.py

Dataset Integrity and Quality Verification Suite.
Validates:
 1. 1:1 Pairing between PNG images and JSON sidecars.
 2. 7-level folder taxonomy depth:
    <target_dir>/<type>/<gender>/<style>/<race>/<source>/<pose>/<filename>
 3. Valid JSON schema:
    - file, generation_prompt, model_used, platform, type, gender, race,
      style, source, pose, colors, tags, resolution, seed
 4. Image integrity:
    - Verifies PNG format and valid PIL image opening
    - Validates pixel art resolution (32x32, 64x64, 128x128)
 5. Zero-tolerance policy on sacred Islamic figures & symbols:
    - No forbidden religious terminology in prompts, metadata, or names
"""

import os
import sys
import json
import tarfile
from pathlib import Path
from PIL import Image

TARGET_DIR = Path("Nano pixel 30.10.2026")
ARCHIVE_PATH = Path("Nano_pixel_30.10.2026.tar.gz")

PROHIBITION_TERMS = [
    "allah", "god_supreme", "god", "muhammad", "mohammed", "prophet", "messenger",
    "jesus", "moses", "abraham", "noah", "ibrahim", "musa", "nuh", "isa",
    "abu-bakr", "abu_bakr", "umar", "uthman", "ali", "fatima", "aisha",
    "kaaba", "quran", "koran", "mecca", "medina", "mosque", "masjid",
    "sacred_symbol", "holy_prophet", "religious_figure", "sahaba", "prophets"
]

REQUIRED_FIELDS = [
    "file", "generation_prompt", "model_used", "platform", "type",
    "gender", "race", "style", "source", "pose", "colors", "tags",
    "resolution", "seed"
]

def verify_dataset():
    extracted_temp = False
    if not TARGET_DIR.exists():
        if ARCHIVE_PATH.exists():
            print(f"Extracting archive {ARCHIVE_PATH} for verification...")
            with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
                tar.extractall()
            extracted_temp = True
        else:
            print(f"Error: Neither {TARGET_DIR} nor {ARCHIVE_PATH} found!")
            return False

    png_files = list(TARGET_DIR.glob("**/*.png"))
    json_files = list(TARGET_DIR.glob("**/*.json"))

    print(f"Found {len(png_files)} PNG files and {len(json_files)} JSON files.")

    errors = []

    if len(png_files) == 0:
        errors.append("No PNG files found in dataset.")

    if len(png_files) != len(json_files):
        errors.append(f"PNG count ({len(png_files)}) does not match JSON count ({len(json_files)}).")

    # Check 1:1 pairing and schema
    for png_path in png_files:
        json_path = png_path.with_suffix(".json")
        if not json_path.exists():
            errors.append(f"Missing JSON sidecar for: {png_path}")
            continue

        # Check directory depth
        # Relative path: type/gender/style/race/source/pose/filename
        rel_parts = png_path.relative_to(TARGET_DIR).parts
        if len(rel_parts) != 7:
            errors.append(f"Invalid directory depth for {png_path}: {rel_parts} (expected 7 parts)")

        # Validate JSON content
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            for req in REQUIRED_FIELDS:
                if req not in meta:
                    errors.append(f"Field '{req}' missing in {json_path}")

            # Check prohibition terms
            meta_str = json.dumps(meta).lower()
            for term in PROHIBITION_TERMS:
                if term in meta_str:
                    errors.append(f"Prohibition violation! Term '{term}' found in {json_path}")

        except Exception as e:
            errors.append(f"Error parsing JSON {json_path}: {e}")

        # Validate Image Integrity
        try:
            with Image.open(png_path) as im:
                im.verify()
            with Image.open(png_path) as im:
                w, h = im.size
                if (w, h) not in [(32, 32), (64, 64), (128, 128)]:
                    errors.append(f"Unexpected image resolution {w}x{h} for {png_path}")
        except Exception as e:
            errors.append(f"Corrupt image file {png_path}: {e}")

    if extracted_temp:
        # Re-pack if needed or clean up
        import shutil
        shutil.rmtree(TARGET_DIR)

    if errors:
        print(f"Verification FAILED with {len(errors)} errors:")
        for err in errors[:20]:
            print(f"  - {err}")
        return False
    else:
        print(f"Verification PASSED! {len(png_files)} items verified successfully.")
        return True

if __name__ == "__main__":
    success = verify_dataset()
    sys.exit(0 if success else 1)
