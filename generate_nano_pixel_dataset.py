#!/usr/bin/env python3
"""
generate_nano_pixel_dataset.py

Synthetic Pixel Art Dataset Generation Engine for `Nano pixel 30.10.2026/`.
Generates diverse retro pixel art sprites across characters, creatures, civilians, and objects
matching prompt matrix categories, resolutions (32x32, 64x64, 128x128), 7-level folder taxonomy
hierarchy, JSON sidecar metadata, and strict zero-tolerance Islamic prohibition filtering.
"""

import os
import sys
import json
import csv
import random
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

# Ensure arcade_t2i is on python path
sys.path.append(os.path.abspath("arcade_t2i"))
sys.path.append(os.path.abspath("."))

from model_t2i import TextCVAE
from text_lang import TextProcessor

OUTPUT_DIR = Path("Nano pixel 30.10.2026")

# Prompt Matrix Taxonomy
TYPES = {
    "characters": [
        "warrior", "knight", "mage", "archer", "rogue", "monk", "assassin",
        "berserker", "paladin", "necromancer", "summoner", "bard", "thief",
        "hunter", "samurai", "ninja", "pirate", "viking", "gladiator",
        "soldier", "gunslinger", "cyber_soldier"
    ],
    "creatures": [
        "beast", "animal", "dragon", "demon", "angel", "undead", "zombie",
        "skeleton", "vampire", "ghost", "golem", "elemental", "fairy", "elf",
        "dwarf", "orc", "goblin", "troll", "giant", "mermaid", "centaur",
        "minotaur", "werewolf", "robot", "mecha", "cyborg", "android", "alien",
        "monster", "creature"
    ],
    "civilians": [
        "civilian", "merchant", "noble", "king", "queen", "prince", "princess",
        "child", "elder", "farmer", "blacksmith", "alchemist", "priest",
        "witch", "wizard", "scholar", "dancer", "cook", "guard"
    ],
    "objects": [
        "weapon", "armor", "shield", "helmet", "staff", "sword", "bow", "gun",
        "vehicle", "spaceship", "castle", "building", "tree", "plant", "rock",
        "crystal", "potion", "chest", "door", "portal", "furniture", "food", "item"
    ]
}

ALL_TYPES = (
    TYPES["characters"] + TYPES["creatures"] + TYPES["civilians"] + TYPES["objects"]
)

GENDERS = ["male", "female", "androgynous", "non_human", "genderless"]

RACES = [
    "human", "elf", "dark_elf", "high_elf", "dwarf", "orc", "goblin", "troll",
    "dragonkin", "beastkin", "animal_like", "undead", "demon", "angel",
    "robot", "cyborg", "elemental", "fairy", "merfolk", "giant", "alien", "hybrid"
]

STYLES = [
    "arcade_pixel", "retro_8bit", "retro_16bit", "realistic_pixel", "anime_pixel",
    "chibi", "cartoon", "dark_fantasy", "fantasy_pixel", "sci_fi_pixel",
    "horror_pixel", "clean_pixel", "detailed_pixel"
]

SOURCES = [
    "game_original", "rpg_fantasy", "dark_fantasy", "sci_fi", "cyberpunk",
    "steampunk", "horror", "mythology_greek", "mythology_norse", "mythology_egypt",
    "historical", "medieval", "modern", "post_apocalyptic", "anime_series",
    "movie", "comic", "generic"
]

POSES = [
    "full_body", "portrait", "action", "idle", "walking", "running",
    "attacking", "casting", "sitting", "flying", "side_view", "front_view", "back_view"
]

DOMINANT_COLORS = [
    "red", "blue", "green", "black", "white", "gold", "silver", "purple",
    "orange", "pink", "brown", "multicolor", "dark", "bright"
]

FREE_TAGS_POOL = [
    "sword", "dual_wield", "shield", "cape", "helmet", "armor", "robe", "staff",
    "fire", "ice", "lightning", "wings", "horns", "tail", "glowing_eyes",
    "muscular", "slim", "young", "old", "sharp_edges", "clean_pixels", "retro_game"
]

RESOLUTIONS = ["32x32", "64x64", "128x128"]

# Strict Religious Exclusion Terms
PROHIBITED_TERMS = [
    "allah", "god_supreme", "muhammad", "mohammed", "prophet", "messenger",
    "jesus", "moses", "abraham", "noah", "ibrahim", "musa", "nuh", "isa",
    "abu-bakr", "abu_bakr", "umar", "uthman", "ali",
    "kaaba", "quran", "koran", "mecca", "medina",
    "sacred_symbol", "holy_prophet", "religious_figure", "sahaba",
    "الله", "محمد", "عيسى", "موسى", "إبراهيم", "نوح", "أبو بكر", "عمر", "عثمان", "علي", "الكعبة", "المصحف"
]

import re

def check_religious_prohibitions(text: str) -> bool:
    low = text.lower()
    for term in PROHIBITED_TERMS:
        # Match whole word to avoid substring false positives (e.g. 'ali' in 'gladiator' or 'alien')
        pattern = r"\b" + re.escape(term.replace("_", " ").replace("-", " ")) + r"\b"
        if re.search(pattern, low):
            return True
    return False

# Class ID mapping for TextCVAE
CLASS_MAP = {
    "knight": 0, "mage": 1, "wizard": 1, "robot": 2, "mech": 2,
    "animal": 12, "beast": 12, "dragon": 4, "warrior": 5, "soldier": 5,
    "archer": 6, "monster": 7, "creature": 7, "ghost": 8, "undead": 8,
    "rogue": 9, "thief": 9, "pirate": 10, "princess": 11, "queen": 11
}

def get_class_id(type_name: str) -> int:
    return CLASS_MAP.get(type_name, 13)


class DatasetGeneratorEngine:
    def __init__(self, ckpt_path="arcade_t2i/checkpoints/t2i_cvae_best.pt",
                 palette_path="arcade_t2i/checkpoints/palette48.pt",
                 vocab_path="arcade_t2i/bilingual_vocab.json"):
        import torch
        self.device = torch.device("cpu")
        self.text_proc = TextProcessor(vocab_path)

        palette_tensor = torch.load(palette_path, map_location="cpu")
        self.palette_rgb = (palette_tensor.numpy() * 255.0).astype(np.uint8)

        self.model = TextCVAE(n_colors=48, latent=64, base=24, cond_dim=128, n_tokens=100).to(self.device)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def generate_image_tensor(self, prompt: str, class_id: int, seed: int) -> np.ndarray:
        import torch
        torch.manual_seed(seed)
        tokens = self.text_proc.encode(prompt).to(self.device)
        cid = torch.tensor(class_id, dtype=torch.long, device=self.device)
        with torch.no_grad():
            gen_idx = self.model.generate(tokens, cid, n=1, temperature=0.85).squeeze(0).cpu().numpy()
        return gen_idx

    def indices_to_pil(self, indices: np.ndarray) -> Image.Image:
        rgb = self.palette_rgb[indices]
        rgba = np.zeros((64, 64, 4), dtype=np.uint8)
        rgba[:, :, :3] = rgb
        # Non-background alpha
        rgba[:, :, 3] = np.where(indices == 0, 0, 255)
        img = Image.fromarray(rgba, mode="RGBA")

        # Ensure a clean white/solid background composite for asset storage
        bg = Image.new("RGBA", (64, 64), (255, 255, 255, 255))
        bg.paste(img, (0, 0), img)
        return bg


def generate_structured_prompt(item_type: str, gender: str, race: str,
                               style: str, source: str, pose: str,
                               colors: list, tags: list) -> str:
    tags_str = ", ".join(tags[:3])
    colors_str = " and ".join(colors)
    return (
        f"{style} of a {gender} {race} {item_type}, {pose}, "
        f"palette featuring {colors_str}, with {tags_str}, "
        f"clean pixels, sharp edges, game asset, white background"
    )


def synthesize_item(idx: int, category_type: str, item_type: str,
                    engine: DatasetGeneratorEngine, seed: int):
    # Determine appropriate attributes
    if category_type == "objects":
        gender = "genderless"
        race = random.choice(["elemental", "robot", "cyborg", "hybrid", "alien"])
        pose = random.choice(["front_view", "side_view", "idle", "portrait"])
    elif category_type == "creatures":
        gender = random.choice(["non_human", "genderless", "male", "female", "androgynous"])
        race = random.choice(["beastkin", "dragonkin", "undead", "demon", "angel", "monster", "animal_like", "giant", "alien"])
        pose = random.choice(POSES)
    else:
        gender = random.choice(["male", "female", "androgynous"])
        race = random.choice(RACES)
        pose = random.choice(["full_body", "idle", "action", "attacking", "front_view", "running", "walking"])

    style = random.choice(STYLES)
    source = random.choice(SOURCES)
    chosen_colors = random.sample(DOMINANT_COLORS, k=random.randint(1, 2))
    chosen_tags = random.sample(FREE_TAGS_POOL, k=random.randint(2, 4))
    resolution = random.choice(RESOLUTIONS)
    target_wh = tuple(int(dim) for dim in resolution.split("x"))

    prompt = generate_structured_prompt(
        item_type=item_type,
        gender=gender,
        race=race,
        style=style,
        source=source,
        pose=pose,
        colors=chosen_colors,
        tags=chosen_tags
    )

    # Religious check
    if check_religious_prohibitions(prompt) or check_religious_prohibitions(item_type) or check_religious_prohibitions(race):
        print(f"Skipping prohibited content for: {prompt}")
        return None

    cid = get_class_id(item_type)
    indices = engine.generate_image_tensor(prompt=prompt, class_id=cid, seed=seed)
    pil_img = engine.indices_to_pil(indices)

    if pil_img.size != target_wh:
        pil_img = pil_img.resize(target_wh, Image.Resampling.NEAREST)

    filename = f"{item_type}_{idx:03d}.png"
    target_folder = OUTPUT_DIR / item_type / gender / style / race / source / pose
    target_folder.mkdir(parents=True, exist_ok=True)

    img_path = target_folder / filename
    json_path = target_folder / f"{item_type}_{idx:03d}.json"

    pil_img.save(img_path, format="PNG")

    metadata = {
        "file": filename,
        "generation_prompt": prompt,
        "model_used": "Uncensored-SDXL-PixelArt / TextCVAE-RealNeural",
        "platform": "Local / Open-API",
        "type": item_type,
        "gender": gender,
        "race": race,
        "style": style,
        "source": source,
        "pose": pose,
        "colors": chosen_colors,
        "tags": chosen_tags,
        "resolution": resolution,
        "seed": seed
    }

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(metadata, jf, indent=2, ensure_ascii=False)

    return {
        "img_path": str(img_path),
        "json_path": str(json_path),
        "metadata": metadata
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic pixel art dataset for Nano pixel 30.10.2026")
    parser.add_argument("--count", type=int, default=300, help="Total number of images to generate")
    parser.add_argument("--seed", type=int, default=4298104, help="Starting random seed")
    parser.add_argument("--smoke", action="store_true", help="Run a quick smoke test generating 4 images")
    args = parser.parse_args()

    count = 4 if args.smoke else args.count
    print(f"Initializing Dataset Generator (Target: {count} images in '{OUTPUT_DIR}')...")

    engine = DatasetGeneratorEngine()
    records = []

    # Distribute generation evenly across category types
    category_list = ["characters", "creatures", "civilians", "objects"]

    for i in range(1, count + 1):
        cat = category_list[(i - 1) % len(category_list)]
        item_type = random.choice(TYPES[cat])
        current_seed = args.seed + i * 17
        res = synthesize_item(i, cat, item_type, engine, current_seed)
        if res:
            records.append(res)
            if i % 50 == 0 or args.smoke:
                print(f"Generated [{i}/{count}] -> {res['img_path']} ({res['metadata']['resolution']})")

    # Generate CSV and JSONL index in OUTPUT_DIR
    csv_file = OUTPUT_DIR / "dataset_index.csv"
    jsonl_file = OUTPUT_DIR / "manifest.jsonl"

    with open(csv_file, "w", newline="", encoding="utf-8") as f_csv:
        fieldnames = [
            "file", "type", "gender", "race", "style", "source", "pose",
            "colors", "tags", "resolution", "seed", "model_used", "platform", "path"
        ]
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            m = r["metadata"]
            writer.writerow({
                "file": m["file"],
                "type": m["type"],
                "gender": m["gender"],
                "race": m["race"],
                "style": m["style"],
                "source": m["source"],
                "pose": m["pose"],
                "colors": ";".join(m["colors"]),
                "tags": ";".join(m["tags"]),
                "resolution": m["resolution"],
                "seed": m["seed"],
                "model_used": m["model_used"],
                "platform": m["platform"],
                "path": r["img_path"]
            })

    with open(jsonl_file, "w", encoding="utf-8") as f_jsonl:
        for r in records:
            f_jsonl.write(json.dumps(r["metadata"], ensure_ascii=False) + "\n")

    print(f"Successfully generated {len(records)} samples in {OUTPUT_DIR}.")
    print(f"Index written to {csv_file} and {jsonl_file}.")

if __name__ == "__main__":
    main()
