"""Unified Repository Image Manifest Builder for 64x64 Arcade Training.

Walks:
  - mm.trine/**
  - nano arcked dat/**
  - nano_arcade_data/**
  - examples/**
  - All-data-Trine/**

Filters:
  - Resizes NEAREST to 64x64.
  - Drops near-empty / transparent frames; logs drops to arcade_t2i/manifests/drops.log.
  - Writes arcade_t2i/manifests/manifest_unified.jsonl
    Fields: path, source_folder, class_name, caption_ar, caption_en, char_hash.
"""
from __future__ import annotations
import glob
import json
import os
import re
import sys
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

CLASSES = [
    {"name": "knight", "ar": "فارس", "en": "knight", "keywords": ["knight", "paladin", "crusader", "فارس"]},
    {"name": "mage", "ar": "ساحر", "en": "mage", "keywords": ["mage", "wizard", "sorcerer", "witch", "warlock", "spellcaster", "ساحر"]},
    {"name": "robot", "ar": "روبوت", "en": "robot", "keywords": ["robot", "mech", "cyber", "android", "cyborg", "mecha", "روبوت"]},
    {"name": "cat", "ar": "قطة", "en": "cat", "keywords": ["cat", "feline", "kitten", "kitty", "قطة"]},
    {"name": "dragon", "ar": "تنين", "en": "dragon", "keywords": ["dragon", "drake", "wyvern", "تنين"]},
    {"name": "warrior", "ar": "محارب", "en": "warrior", "keywords": ["warrior", "fighter", "barbarian", "berserker", "brawler", "محارب"]},
    {"name": "archer", "ar": "رامي", "en": "archer", "keywords": ["archer", "bowman", "ranger", "hunter", "رامي"]},
    {"name": "monster", "ar": "وحش", "en": "monster", "keywords": ["monster", "creature", "beast", "demon", "goblin", "orc", "ogre", "troll", "وحش"]},
    {"name": "ghost", "ar": "شبح", "en": "ghost", "keywords": ["ghost", "undead", "zombie", "skeleton", "wraith", "specter", "شبح"]},
    {"name": "rogue", "ar": "لص", "en": "rogue", "keywords": ["rogue", "assassin", "ninja", "thief", "shadow", "لص"]},
    {"name": "pirate", "ar": "قرصان", "en": "pirate", "keywords": ["pirate", "corsair", "buccaneer", "قرصان"]},
    {"name": "princess", "ar": "أميرة", "en": "princess", "keywords": ["princess", "queen", "lady", "أميرة"]},
    {"name": "animal", "ar": "حيوان", "en": "animal", "keywords": ["animal", "dog", "wolf", "bear", "fox", "حيوان"]},
    {"name": "character", "ar": "شخصية", "en": "character", "keywords": ["character", "hero", "npc", "civilian", "person", "شخصية"]},
]

MM_TRINE_HASH_RE = re.compile(r"^\d+__([0-9a-fA-F]+)__")

def get_class_info(path_str: str) -> dict:
    p_lower = path_str.replace("\\", "/").lower()
    parts = p_lower.split("/")
    for cls in CLASSES:
        if cls["name"] in parts:
            return cls
    for cls in CLASSES[:-1]:
        for kw in cls["keywords"]:
            if kw in p_lower:
                return cls
    import hashlib
    h = int(hashlib.md5(path_str.encode("utf-8")).hexdigest(), 16)
    core = [c for c in CLASSES if c["name"] != "character"]
    return core[h % len(core)]

def extract_char_hash(path_str: str) -> str | None:
    filename = Path(path_str).name
    m = MM_TRINE_HASH_RE.match(filename)
    if m:
        return m.group(1)
    return None

def is_usable_frame(img: Image.Image) -> tuple[bool, str]:
    w, h = img.size
    if w < 8 or h < 8:
        return False, f"too_small_{w}x{h}"

    # Resize to 64x64 NEAREST
    res = img.resize((64, 64), Image.Resampling.NEAREST)

    if res.mode in ("RGBA", "LA") or (res.mode == "P" and "transparency" in res.info):
        rgba = res.convert("RGBA")
        arr = np.array(rgba)
        alpha = arr[:, :, 3]
        opaque_count = np.sum(alpha > 32)
        total_pixels = 64 * 64
        # Drop if less than 0.5% visible pixels (< 20 pixels)
        if opaque_count < 20:
            return False, f"near_empty_alpha_{opaque_count}_pixels"
        # Check standard deviation of RGB values where alpha is visible
        visible_rgb = arr[:, :, :3][alpha > 32]
        if len(visible_rgb) > 0 and np.std(visible_rgb) < 1.0 and np.mean(visible_rgb) in (0.0, 255.0):
            return False, "flat_color_solid"
    else:
        rgb = res.convert("RGB")
        arr = np.array(rgb)
        if np.std(arr) < 1.0:
            return False, "flat_color_solid"

    return True, "ok"

def main():
    target_dirs = [
        "mm.trine",
        "nano arcked dat",
        "nano_arcade_data",
        "examples",
        "All-data-Trine"
    ]

    out_dir = ROOT / "arcade_t2i" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = out_dir / "manifest_unified.jsonl"
    drops_file = out_dir / "drops.log"

    all_files: list[tuple[str, str]] = []
    for d in target_dirs:
        dir_path = ROOT / d
        if not dir_path.exists():
            print(f"Warning: directory {d} not found.")
            continue
        patterns = ["**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.webp", "**/*.gif"]
        for pat in patterns:
            for p in dir_path.glob(pat):
                if p.is_file():
                    rel_p = str(p.relative_to(ROOT))
                    all_files.append((d, rel_p))

    # Sort files deterministically
    all_files.sort(key=lambda x: x[1])
    print(f"Discovered total candidate image files across repo: {len(all_files)}")

    manifest_rows = []
    drop_logs = []
    folder_counts: dict[str, int] = {d: 0 for d in target_dirs}
    class_counts: dict[str, int] = {}

    for idx, (src_folder, rel_path) in enumerate(all_files):
        abs_path = ROOT / rel_path
        try:
            with Image.open(abs_path) as img:
                usable, reason = is_usable_frame(img)
                if not usable:
                    drop_logs.append(f"{rel_path}\t{reason}\n")
                    continue

                cls = get_class_info(rel_path)
                char_hash = extract_char_hash(rel_path) if src_folder == "mm.trine" else None

                row = {
                    "path": rel_path,
                    "source_folder": src_folder,
                    "class_name": cls["name"],
                    "caption_ar": f"شخصية بكسل آرت {cls['ar']} في لعبة أركيد",
                    "caption_en": f"pixel art {cls['en']} arcade character sprite",
                    "char_hash": char_hash
                }
                manifest_rows.append(row)
                folder_counts[src_folder] = folder_counts.get(src_folder, 0) + 1
                class_counts[cls["name"]] = class_counts.get(cls["name"], 0) + 1
        except Exception as e:
            drop_logs.append(f"{rel_path}\terror_{type(e).__name__}_{str(e)}\n")

        if (idx + 1) % 2500 == 0 or idx == len(all_files) - 1:
            print(f"Processed {idx + 1}/{len(all_files)}... kept={len(manifest_rows)}, dropped={len(drop_logs)}")

    # Write manifest
    with manifest_file.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Write drops log
    with drops_file.open("w", encoding="utf-8") as f:
        f.writelines(drop_logs)

    total_n = len(manifest_rows)
    print("\n" + "="*50)
    print(f"Total Kept Usable Frames (N): {total_n}")
    print(f"Total Dropped Frames: {len(drop_logs)}")
    print("="*50)
    print("Per-folder Counts:")
    for fld, cnt in sorted(folder_counts.items()):
        print(f"  - {fld}: {cnt}")
    print("="*50)
    print("Class Distribution:")
    for cls_name, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  - {cls_name}: {cnt}")
    print("="*50)

    if total_n < 1000:
        print(f"HARD STOP ERROR: N ({total_n}) < 1000!", file=sys.stderr)
        sys.exit(1)

    print(f"Unified manifest successfully written to: {manifest_file}")

if __name__ == "__main__":
    main()
