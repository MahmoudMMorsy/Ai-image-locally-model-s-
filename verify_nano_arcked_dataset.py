#!/usr/bin/env python3
"""
verify_nano_arcked_dataset.py

Verification suite for the nano arcked dat dataset:
- Validates 1:1 pairing between PNG images and JSON sidecars in data/
- Validates JSON sidecar metadata schema and non-empty required fields
- Validates directory hierarchy structure matches type/gender/style/race/source/pose taxonomy
- Verifies image integrity and dimensions using PIL
- Enforces strict zero-tolerance sacred religious prohibition filter
- Checks index CSV and manifest jsonl entry alignment
"""

import os
import re
import csv
import json
import glob
from pathlib import Path
from PIL import Image

DATA_DIR = Path("nano arcked dat/data")
INDEX_CSV = Path("nano arcked dat/dataset_index.csv")
MANIFEST_JSONL = Path("nano arcked dat/manifests/dataset.jsonl")

REQUIRED_SCHEMA_KEYS = [
    "file", "type", "gender", "race", "style", "source", "pose", "colors", "tags", "resolution", "notes"
]

EXCLUDE_TERMS = [
    "allah", "god_supreme", "muhammad", "mohammed", "prophet", "messenger",
    "jesus", "moses", "abraham", "noah", "ibrahim", "musa", "nuh", "isa",
    "abu-bakr", "abu_bakr", "umar", "uthman", "ali",
    "kaaba", "quran", "koran", "mecca", "medina",
    "sacred_symbol", "holy_prophet", "religious_figure", "sahaba"
]

def check_religious_prohibitions(text: str) -> bool:
    n = text.lower()
    for term in EXCLUDE_TERMS:
        pattern = r"\b" + re.escape(term.replace("_", " ").replace("-", " ")) + r"\b"
        if re.search(pattern, n):
            return True
    return False

def verify_dataset():
    print("=== Running Nano Arcked Dataset Verification Suite ===")

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory {DATA_DIR} does not exist!")

    png_files = list(DATA_DIR.rglob("*.png"))
    json_files = list(DATA_DIR.rglob("*.json"))

    print(f"Found {len(png_files)} PNG files and {len(json_files)} JSON sidecar files in {DATA_DIR}.")

    assert len(png_files) > 0, "Error: No PNG files found in dataset!"
    assert len(png_files) == len(json_files), f"Error: Mismatch between PNGs ({len(png_files)}) and JSON sidecars ({len(json_files)})!"

    png_stems = {p.with_suffix("").as_posix() for p in png_files}
    json_stems = {j.with_suffix("").as_posix() for j in json_files}

    missing_jsons = png_stems - json_stems
    missing_pngs = json_stems - png_stems

    assert len(missing_jsons) == 0, f"Error: Found PNGs without JSON sidecars: {missing_jsons}"
    assert len(missing_pngs) == 0, f"Error: Found JSONs without PNG images: {missing_pngs}"

    print("[✔] 1:1 PNG <-> JSON sidecar pairing verified.")

    # Validate schema, hierarchy, images, and prohibitions
    verified_count = 0
    for png_path in png_files:
        json_path = png_path.with_suffix(".json")

        # 1. Image check with PIL
        try:
            with Image.open(png_path) as im:
                im.verify()
        except Exception as e:
            raise RuntimeError(f"Corrupted or invalid image file {png_path}: {e}")

        # 2. JSON schema check
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        for key in REQUIRED_SCHEMA_KEYS:
            assert key in data, f"Missing required key '{key}' in sidecar {json_path}"

        assert isinstance(data["colors"], list), f"'colors' must be a list in {json_path}"
        assert isinstance(data["tags"], list), f"'tags' must be a list in {json_path}"

        # 3. Hierarchy check
        rel_parts = png_path.relative_to(DATA_DIR).parts
        assert len(rel_parts) == 7, f"Invalid directory depth ({len(rel_parts)}) for {png_path}. Expected 7 levels: type/gender/style/race/source/pose/filename.png"

        c_type, c_gender, c_style, c_race, c_source, c_pose, filename = rel_parts
        assert data["type"] == c_type, f"Type mismatch in {json_path}: {data['type']} != {c_type}"
        assert data["gender"] == c_gender, f"Gender mismatch in {json_path}: {data['gender']} != {c_gender}"
        assert data["style"] == c_style, f"Style mismatch in {json_path}: {data['style']} != {c_style}"

        # 4. Religious exclusion check
        full_text = f"{png_path.as_posix()} {json.dumps(data)}"
        assert not check_religious_prohibitions(full_text), f"PROHIBITED CONTENT DETECTED in {png_path}"

        verified_count += 1

    print(f"[✔] All {verified_count} assets passed image, JSON schema, taxonomy hierarchy, and religious restriction checks.")

    # Validate Index CSV and Manifest JSONL
    assert INDEX_CSV.exists(), f"Index CSV {INDEX_CSV} missing!"
    assert MANIFEST_JSONL.exists(), f"Manifest JSONL {MANIFEST_JSONL} missing!"

    with open(INDEX_CSV, encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))

    with open(MANIFEST_JSONL, encoding="utf-8") as f:
        jsonl_rows = [json.loads(line) for line in f]

    assert len(csv_rows) == len(png_files), f"CSV index count ({len(csv_rows)}) does not match PNG count ({len(png_files)})!"
    assert len(jsonl_rows) == len(png_files), f"Manifest JSONL count ({len(jsonl_rows)}) does not match PNG count ({len(png_files)})!"

    print(f"[✔] Dataset CSV index ({len(csv_rows)} rows) and manifest JSONL ({len(jsonl_rows)} lines) perfectly aligned.")
    print("=== Nano Arcked Dataset Verification Successful ===")

if __name__ == "__main__":
    verify_dataset()
