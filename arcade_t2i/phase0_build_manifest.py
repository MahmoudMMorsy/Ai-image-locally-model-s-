#!/usr/bin/env python3
"""Phase 0: index all arcade sprites into manifest_unified.jsonl

Run from repo root or arcade_t2i:
  python arcade_t2i/phase0_build_manifest.py --repo-root .
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

SOURCE_DIRS = [
    "mm.trine",
    "nano arcked dat",
    "nano_arcade_data",
    "examples",
    "All-data-Trine",
    "Nano pixel 30.10.2026",
]

CLASS_MAP = {
    "knight": ("فارس", "knight"),
    "mage": ("ساحر", "mage"),
    "wizard": ("ساحر", "wizard"),
    "robot": ("روبوت", "robot"),
    "warrior": ("محارب", "warrior"),
    "dragon": ("تنين", "dragon"),
    "archer": ("رامي", "archer"),
    "ninja": ("نينجا", "ninja"),
    "pirate": ("قرصان", "pirate"),
    "monster": ("وحش", "monster"),
    "ghost": ("شبح", "ghost"),
    "princess": ("أميرة", "princess"),
    "merchant": ("تاجر", "merchant"),
    "weapon": ("سلاح", "weapon"),
    "rogue": ("لص", "rogue"),
    "animal": ("حيوان", "animal"),
}


def guess_class(path: Path) -> str:
    low = str(path).lower().replace("\\", "/")
    for key in CLASS_MAP:
        if f"/{key}/" in low or f"_{key}_" in low or low.endswith(f"/{key}"):
            return key
    return "character"


def char_hash_from_name(name: str) -> str | None:
    m = re.search(r"__([0-9a-f]{8,16})__", name)
    return m.group(1) if m else None


def captions(class_name: str) -> tuple[str, str]:
    if class_name in CLASS_MAP:
        ar, en = CLASS_MAP[class_name]
        return f"{ar}، شخصية كاملة أركيد 64",
            f"{en} full body arcade character 64x64"
    return "شخصية أركيد بكسل",
        "arcade pixel character full body"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: <repo>/arcade_t2i/manifests/manifest_unified.jsonl",
    )
    args = ap.parse_args()
    root = args.repo_root.resolve()
    out = args.out or (root / "arcade_t2i" / "manifests" / "manifest_unified.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    per_folder: Counter[str] = Counter()
    per_class: Counter[str] = Counter()
    skipped = 0

    for folder in SOURCE_DIRS:
        base = root / folder
        if not base.exists():
            print(f"[skip missing] {folder}")
            continue
        for p in base.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in IMG_EXT:
                continue
            # lightweight filter: skip obvious non-sprites by name
            name_l = p.name.lower()
            if name_l.endswith(".json") or "readme" in name_l:
                skipped += 1
                continue
            rel = str(p.relative_to(root)).replace("\\", "/")
            cls = guess_class(p)
            ar, en = captions(cls)
            row = {
                "path": rel,
                "source_folder": folder,
                "class_name": cls,
                "caption_ar": ar,
                "caption_en": en,
                "char_hash": char_hash_from_name(p.name),
            }
            rows.append(row)
            per_folder[folder] += 1
            per_class[cls] += 1

    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "N": len(rows),
        "skipped": skipped,
        "per_folder": dict(per_folder),
        "per_class": dict(per_class),
        "out": str(out),
    }
    report_path = out.parent / "manifest_unified_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if len(rows) < 1000:
        print("HARD STOP WARNING: N < 1000 — wrong dataset or incomplete checkout")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
