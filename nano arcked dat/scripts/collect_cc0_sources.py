#!/usr/bin/env python3
"""
Broad, repeatable pixel-art image collector.

The collector is image-first and provenance-first:
+ downloads actual image files from configured sources
- keeps original SOURCE.md files when available
- hashes every image
- records dimensions and alpha
- applies conservative exclusion terms
- quarantines ambiguous filenames instead of silently including them
"""

from __future__ import annotations
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

try:
    from PIL import Image
except Exception:
    Image = None

ROOT = Path("nano arcked dat")
WORK = Path(".collector_work")
OUT = ROOT / "images"
META = ROOT / "metadata"
MANIFEST = ROOT / "manifests"
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "250"))
MAX_FILE_MB = float(os.environ.get("MAX_FILE_MB", "8"))


SOURCES = [
    {"name":"tiddybub_2d_assets","url":"https://github.com/Tiddybub/2d-assets.git","license":"CC0","license_url":"https://creativecommons.org/publicdomain/zero/1.0/","ai_generated":False},
    {"name":"papyszoo_cc0_public_domain_sprites","url":"https://github.com/Papyszoo/CC0-Public-Domain-Sprites.git","license":"CC0","license_url":"https://creativecommons.org/publicdomain/zero/1.0/","ai_generated":False},
    {"name":"spritecook_free_game_assets","url":"https://github.com/SpriteCook/spritecook-free-game-assets.git","license":"CC0","license_url":"https://creativecommons.org/publicdomain/zero/1.0/","ai_generated":True},
    {"name":"kenney_cc0_2d","url":"https://github.com/shorepine/kenney.git","license":"CC0","license_url":"https://creativecommons.org/publicdomain/zero/1.0/","ai_generated":False},
]


# Conservative text-level exclusion. Visual verification is still required for uncertain cases.
EXCLUDE_TERMS = [
    "allah", "muhammad", "mohammed", "prophet", "messenger",
    "jesus", "moses", "abraham", "noah", "ibrahim", "musa", "nuh",
    "abu-bakr", "abu_bakr", "umar", "uthman", "ali",
    "kaaba", "quran", "koran", "mecca", "medina",
    "sacred", "holy-prophet", "religious-figure"
]

TYPE_MAP = [
    ("cyber_soldier", ["cyber", "trooper", "space soldier", "space_soldier"]),
    ("knight", ["knight", "swordsman"]),
    ("mage", ["mage", "wizard", "sorcer"]),
    ("archer", ["archer", "bow"]),
    ("rogue", ["rogue"]),
    ("assassin", ["assassin"]),
    ("ninja", ["ninja"]),
    ("monk", ["monk"]),
    ("paladin", ["paladin"]),
    ("pirate", ["pirate"]),
    ("viking", ["viking"]),
    ("samurai", ["samurai"]),
    ("warrior", ["warrior", "fighter", "hero", "soldier"]),
    ("robot", ["robot", "droid", "mecha", "cyborg", "android"]),
    ("dragon", ["dragon"]),
    ("demon", ["demon"]),
    ("undead", ["undead", "zombie", "skeleton", "vampire", "ghost"]),
    ("animal", ["animal", "bird", "cat", "dog", "fish", "frog", "bear", "horse"]),
    ("monster", ["monster", "creature", "beast", "slime", "enemy"]),
    ("weapon", ["sword", "weapon", "axe", "bow", "gun", "dagger"]),
    ("vehicle", ["car", "ship", "spaceship", "plane", "tank", "vehicle"]),
    ("item", ["item", "coin", "potion", "chest", "icon"]),
    ("civilian", ["civilian", "person", "character", "npc", "villager", "people"]),
]

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def blocked(text: str) -> bool:
    n = norm(text)
    return any(re.search(r"\b" + re.escape(term.replace("-", " ")) + r"\b", n) for term in EXCLUDE_TERMS)

def infer_type(text: str) -> str:
    n = norm(text)
    for kind, words in TYPE_MAP:
        if any(w in n for w in words):
            return kind
    return "creature"

def infer_gender(text: str) -> str:
    n = norm(text)
    if any(w in n for w in ["female", "woman", "girl", "lady"]):
        return "female"
    if any(w in n for w in ["male", "man", "boy", "gentleman"]):
        return "male"
    return "androgynous"

def infer_style(text: str) -> str:
    n = norm(text)
    if "8 bit" in n or "nes" in n or "pico 8" in n:
        return "retro_8bit"
    if "16 bit" in n or "gba" in n:
        return "retro_16bit"
    if "arcade" in n:
        return "arcade_pixel"
    return "clean_pixel"

def infer_pose(text: str) -> str:
    n = norm(text)
    if "run" in n or "running" in n:
        return "running"
    if "walk" in n:
        return "walking"
    if "attack" in n or "combat" in n:
        return "attacking"
    if "idle" in n or "stand" in n:
        return "idle"
    return "action"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def image_info(path: Path):
    if Image is None:
        return None, None, None
    try:
        with Image.open(path) as im:
            return f"{im.width}x{im.height}", bool("A" in im.getbands()), im.mode
    except Exception:
        return None, None, None

def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value[:180] or "asset.png"

def main():
    WORK.mkdir(exist_ok=True)
    sources = []
    for spec in SOURCES:
        src = WORK / spec["name"]
        if src.exists():
            shutil.rmtree(src)
        subprocess.run(["git", "clone", "--depth", "1", spec["url"], str(src)], check=True)
        sources.append((spec, src))

    candidates = []
    for spec, src in sources:
        for p in src.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in {".png", ".webp", ".gif", ".jpg", ".jpeg"}:
                continue
            rel = p.relative_to(src).as_posix()
            if any(part.startswith(".") for part in p.parts):
                continue
            size_mb = p.stat().st_size / (1024 * 1024)
            if size_mb > MAX_FILE_MB:
                continue
            if blocked(rel):
                continue
            candidates.append((spec, src, p))

    candidates.sort(key=lambda item: (0 if "character" in item[2].as_posix().lower() else 1, item[2].as_posix().lower()))
    candidates = candidates[:MAX_IMAGES]

    MANIFEST.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    records = []
    quarantine = []

    for spec, src_root, src in candidates:
        rel = src.relative_to(src_root).as_posix()
        # SOURCE.md is read only for provenance/classification. The collected asset itself is always a real image file.
        source_md = ""
        for parent in [src.parent, *src.parents]:
            if parent == src_root:
                break
            candidate = parent / "SOURCE.md"
            if candidate.exists():
                source_md = candidate.read_text(encoding="utf-8", errors="replace")
                break

        combined = rel + "\n" + source_md
        if blocked(combined):
            quarantine.append({"file": rel, "reason": "excluded keyword in path/source metadata"})
            continue

        digest = sha256(src)
        asset_id = digest[:16]
        canvas, alpha, mode = image_info(src)
        kind = infer_type(combined)
        gender = infer_gender(combined)
        style = infer_style(combined)
        pose = infer_pose(combined)
        source_pack = src.relative_to(src_root).parts[0] if src.relative_to(src_root).parts else spec["name"]

        target_dir = OUT / kind / gender / style / "unknown" / spec["name"] / pose
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{asset_id}_{safe_name(src.name)}"
        dst = target_dir / filename
        if not dst.exists():
            shutil.copy2(src, dst)

        meta = {
            "id": asset_id,
            "file": dst.relative_to(ROOT).as_posix(),
            "type": kind,
            "gender": gender,
            "race": "unknown",
            "style": style,
            "source": spec["name"],
            "source_url": spec["url"].removesuffix(".git"),
            "author": "see source repository metadata",
            "license": spec["license"],
            "license_url": spec["license_url"],
            "attribution_required": False,
            "pose": pose,
            "view": "unknown",
            "canvas_size": canvas or "unknown",
            "sprite_sheet": bool(canvas and ("sheet" in norm(rel) or "spritesheet" in norm(rel))),
            "alpha": alpha,
            "tags": [x for x in re.split(r"[/_ -]+", norm(rel)) if x][:40],
            "sha256": digest,
            "original_path": rel,
            "collection": source_pack,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "ai_generated_source": bool(spec.get("ai_generated", False)),
            "notes": "Candidate for human review. Visual suitability is not inferred from license metadata."
        }
        (META / f"{asset_id}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(meta)

    with (MANIFEST / "dataset.jsonl").open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with (MANIFEST / "quarantine.jsonl").open("a", encoding="utf-8") as f:
        for q in quarantine:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    csv_path = MANIFEST / "dataset.csv"
    fields = ["id","file","type","gender","race","style","source","source_url","license","pose","canvas_size","sprite_sheet","sha256","original_path","collection"]
    existing = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not existing:
            w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in fields})

    print(f"COLLECTED={len(records)}")
    print(f"QUARANTINED={len(quarantine)}")
    print(f"CANDIDATES={len(candidates)}")
    print(f"SOURCES={len(sources)}")

if __name__ == "__main__":
    main()
