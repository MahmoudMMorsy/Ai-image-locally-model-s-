#!/usr/bin/env python3
"""
verify_nano_pixel_dataset.py

Verification suite for the synthetic pixel art dataset in `Nano pixel 30.10.2026/`:
- Validates 1:1 pairing between PNG images and JSON metadata sidecars
- Validates JSON sidecar metadata schema and non-empty required fields
- Validates directory hierarchy structure matches type/gender/style/race/source/pose taxonomy (7 levels)
- Verifies image integrity, format (PNG), and valid resolutions (32x32, 64x64, 128x128) using PIL
- Enforces strict zero-tolerance sacred religious prohibition filter
- Checks dataset_index.csv and manifest.jsonl alignment
"""

import os
import re
import csv
import json
from pathlib import Path
from PIL import Image

DATASET_ROOT = Path("Nano pixel 30.10.2026")
INDEX_CSV = DATASET_ROOT / "dataset_index.csv"
MANIFEST_JSONL = DATASET_ROOT / "manifest.jsonl"

REQUIRED_SCHEMA_KEYS = [
    "file", "generation_prompt", "model_used", "platform", "type",
    "gender", "race", "style", "source", "pose", "colors", "tags",
    "resolution", "seed"
]

ALLOWED_RESOLUTIONS = ["32x32", "64x64", "128x128"]

PROHIBITED_TERMS = [
    "allah", "god_supreme", "muhammad", "mohammed", "prophet", "messenger",
    "jesus", "moses", "abraham", "noah", "ibrahim", "musa", "nuh", "isa",
    "abu-bakr", "abu_bakr", "umar", "uthman", "ali",
    "kaaba", "quran", "koran", "mecca", "medina",
    "sacred_symbol", "holy_prophet", "religious_figure", "sahaba",
    "الله", "محمد", "عيسى", "موسى", "إبراهيم", "نوح", "أبو بكر", "عمر", "عثمان", "علي", "الكعبة", "المصحف"
]

def check_religious_prohibitions(text: str) -> bool:
    low = text.lower()
    for term in PROHIBITED_TERMS:
        pattern = r"\b" + re.escape(term.replace("_", " ").replace("-", " ")) + r"\b"
        if re.search(pattern, low):
            return True
    return False

def verify_dataset():
    print("=== Running Nano Pixel 30.10.2026 Dataset Verification Suite ===")

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset directory '{DATASET_ROOT}' does not exist!")

    png_files = list(DATASET_ROOT.rglob("*.png"))
    json_files = list(DATASET_ROOT.rglob("*.json"))

    print(f"Found {len(png_files)} PNG files and {len(json_files)} JSON sidecar files in {DATASET_ROOT}.")

    assert len(png_files) > 0, "Error: No PNG files found in dataset!"
    assert len(png_files) == len(json_files), (
        f"Error: Mismatch between PNGs ({len(png_files)}) and JSON sidecars ({len(json_files)})!"
    )

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
            with Image.open(png_path) as im:
                w, h = im.size
                res_str = f"{w}x{h}"
                assert res_str in ALLOWED_RESOLUTIONS, f"Invalid resolution {res_str} for {png_path}"
        except Exception as e:
            raise RuntimeError(f"Corrupted or invalid image file {png_path}: {e}")

        # 2. JSON schema check
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        for key in REQUIRED_SCHEMA_KEYS:
            assert key in data, f"Missing required key '{key}' in sidecar {json_path}"

        assert isinstance(data["colors"], list), f"'colors' must be a list in {json_path}"
        assert isinstance(data["tags"], list), f"'tags' must be a list in {json_path}"
        assert data["resolution"] == res_str, (
            f"Resolution mismatch in {json_path}: {data['resolution']} != {res_str}"
        )

        # 3. Hierarchy check (7 levels: type/gender/style/race/source/pose/filename.png)
        rel_parts = png_path.relative_to(DATASET_ROOT).parts
        assert len(rel_parts) == 7, (
            f"Invalid directory depth ({len(rel_parts)}) for {png_path}. "
            f"Expected 7 levels: type/gender/style/race/source/pose/filename.png"
        )

        c_type, c_gender, c_style, c_race, c_source, c_pose, filename = rel_parts
        assert data["type"] == c_type, f"Type mismatch in {json_path}: {data['type']} != {c_type}"
        assert data["gender"] == c_gender, f"Gender mismatch in {json_path}: {data['gender']} != {c_gender}"
        assert data["style"] == c_style, f"Style mismatch in {json_path}: {data['style']} != {c_style}"
        assert data["race"] == c_race, f"Race mismatch in {json_path}: {data['race']} != {c_race}"
        assert data["source"] == c_source, f"Source mismatch in {json_path}: {data['source']} != {c_source}"
        assert data["pose"] == c_pose, f"Pose mismatch in {json_path}: {data['pose']} != {c_pose}"
        assert data["file"] == filename, f"Filename mismatch in {json_path}: {data['file']} != {filename}"

        # 4. Religious exclusion check
        full_text = f"{png_path.as_posix()} {json.dumps(data, ensure_ascii=False)}"
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

    assert len(csv_rows) == len(png_files), (
        f"CSV index count ({len(csv_rows)}) does not match PNG count ({len(png_files)})!"
    )
    assert len(jsonl_rows) == len(png_files), (
        f"Manifest JSONL count ({len(jsonl_rows)}) does not match PNG count ({len(png_files)})!"
    )

    print(f"[✔] Dataset CSV index ({len(csv_rows)} rows) and manifest JSONL ({len(jsonl_rows)} lines) perfectly aligned.")
    print("=== Nano Pixel 30.10.2026 Dataset Verification Successful ===")

if __name__ == "__main__":
    verify_dataset()
