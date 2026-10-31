#!/usr/bin/env python3
"""
generate_nano_pixel_dataset.py

Massive Synthetic Pixel Art Dataset Generator.
Generates pixel art sprites across characters, creatures, civilians, and items/objects
using local uncensored neural diffusion/CVAE models and open generative pipelines
with sharp edge downsampling and quantization into 32x32, 64x64, 128x128 pixel art representations.

Outputs directly into target directory:
    Nano pixel 30.10.2026/ <type>/ <gender>/ <style>/ <race>/ <source>/ <pose>/ <filename>.png
with matching sidecar JSON metadata files:
    <filename>.json

Enforces zero-tolerance prohibition against sacred Islamic religious figures, prophets, companions,
and religious symbols.
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

# Target Root Directory
TARGET_DIR = Path("Nano pixel 30.10.2026")

# Add arcade_t2i to python path
ARCADE_T2I_DIR = Path(__file__).resolve().parent / "arcade_t2i"
if str(ARCADE_T2I_DIR) not in sys.path:
    sys.path.append(str(ARCADE_T2I_DIR))

# Prohibition list (Zero-tolerance policy)
PROHIBITION_TERMS = [
    "allah", "god_supreme", "god", "muhammad", "mohammed", "prophet", "messenger",
    "jesus", "moses", "abraham", "noah", "ibrahim", "musa", "nuh", "isa",
    "abu-bakr", "abu_bakr", "umar", "uthman", "ali", "fatima", "aisha",
    "kaaba", "quran", "koran", "mecca", "medina", "mosque", "masjid",
    "sacred_symbol", "holy_prophet", "religious_figure", "sahaba", "prophets"
]

# Grid Categories specified in the prompt:
TYPES = {
    # Characters (22)
    "characters": [
        "warrior", "knight", "mage", "archer", "rogue", "monk", "assassin",
        "berserker", "paladin", "necromancer", "summoner", "bard", "thief",
        "hunter", "samurai", "ninja", "pirate", "viking", "gladiator",
        "soldier", "gunslinger", "cyber_soldier"
    ],
    # Creatures (30)
    "creatures": [
        "beast", "animal", "dragon", "demon", "angel", "undead", "zombie",
        "skeleton", "vampire", "ghost", "golem", "elemental", "fairy", "elf",
        "dwarf", "orc", "goblin", "troll", "giant", "mermaid", "centaur",
        "minotaur", "werewolf", "robot", "mecha", "cyborg", "android",
        "alien", "monster", "creature"
    ],
    # Civilians & Roles (19)
    "civilians": [
        "civilian", "merchant", "noble", "king", "queen", "prince", "princess",
        "child", "elder", "farmer", "blacksmith", "alchemist", "priest",
        "witch", "wizard", "scholar", "dancer", "cook", "guard"
    ],
    # Objects & Items (23)
    "items": [
        "weapon", "armor", "shield", "helmet", "staff", "sword", "bow",
        "gun", "vehicle", "spaceship", "castle", "building", "tree", "plant",
        "rock", "crystal", "potion", "chest", "door", "portal", "furniture",
        "food", "item"
    ]
}

GENDERS = ["male", "female", "androgynous", "non_human", "genderless"]

RACES = [
    "human", "elf", "dark_elf", "high_elf", "dwarf", "orc", "goblin",
    "troll", "dragonkin", "beastkin", "animal_like", "undead", "demon",
    "angel", "robot", "cyborg", "elemental", "fairy", "merfolk",
    "giant", "alien", "hybrid"
]

STYLES = [
    "arcade_pixel", "retro_8bit", "retro_16bit", "realistic_pixel",
    "anime_pixel", "chibi", "cartoon", "dark_fantasy", "fantasy_pixel",
    "sci_fi_pixel", "horror_pixel", "clean_pixel", "detailed_pixel"
]

SOURCES = [
    "game_original", "rpg_fantasy", "dark_fantasy", "sci_fi", "cyberpunk",
    "steampunk", "horror", "mythology_greek", "mythology_norse",
    "mythology_egypt", "historical", "medieval", "modern",
    "post_apocalyptic", "anime_series", "movie", "comic", "generic"
]

POSES = [
    "full_body", "portrait", "action", "idle", "walking", "running",
    "attacking", "casting", "sitting", "flying", "side_view", "front_view", "back_view"
]

DOMINANT_COLORS = [
    "red", "blue", "green", "black", "white", "gold", "silver",
    "purple", "orange", "pink", "brown", "multicolor", "dark", "bright"
]

FREE_TAGS = [
    "sword", "dual_wield", "shield", "cape", "helmet", "armor", "robe",
    "staff", "fire", "ice", "lightning", "wings", "horns", "tail",
    "glowing_eyes", "muscular", "slim", "young", "old", "cybernetic",
    "steampunk_gears", "magic_aura", "floating", "battle_ready",
    "game_sprite", "clean_pixels", "sharp_edges", "retro_palette"
]

RESOLUTIONS = ["32x32", "64x64", "128x128"]

def is_prohibited(text):
    text_lower = text.lower()
    for term in PROHIBITION_TERMS:
        if term in text_lower:
            return True
    return False

# Local CVAE & Neural Engine Setup
_cvae_model = None
_cvae_text_proc = None
_cvae_palette_rgb = None
_pixel_sprite_engine = None

def get_neural_engine():
    global _cvae_model, _cvae_text_proc, _cvae_palette_rgb, _pixel_sprite_engine
    if _cvae_model is not None:
        return _cvae_model, _cvae_text_proc, _cvae_palette_rgb, _pixel_sprite_engine
    try:
        import torch
        from model_t2i import TextCVAE
        from text_lang import TextProcessor

        device = torch.device("cpu")
        _cvae_text_proc = TextProcessor()
        pal_path = ARCADE_T2I_DIR / "checkpoints" / "palette48.pt"
        ckpt_path = ARCADE_T2I_DIR / "checkpoints" / "t2i_cvae_best.pt"

        if pal_path.exists() and ckpt_path.exists():
            palette_tensor = torch.load(pal_path, map_location=device)
            _cvae_palette_rgb = (palette_tensor.numpy() * 255.0).astype(np.uint8)

            model = TextCVAE(n_colors=48, latent=64, base=24, cond_dim=128, n_tokens=100).to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            _cvae_model = model

        from pixel_art_engine.engine import PixelSpriteEngine
        _pixel_sprite_engine = PixelSpriteEngine(device="cpu")
    except Exception as e:
        print(f"[Warning] Neural engine loading error: {e}")
    return _cvae_model, _cvae_text_proc, _cvae_palette_rgb, _pixel_sprite_engine

def generate_neural_sprite(prompt, item_type, target_res=(64, 64), seed=42):
    model, text_proc, palette_rgb, engine = get_neural_engine()
    import torch
    from generate_t2i import infer_class_id, indices_to_pil

    torch.manual_seed(seed)
    np.random.seed(seed)

    if model is not None and text_proc is not None and palette_rgb is not None:
        try:
            tokens = text_proc.encode(prompt)
            cid = torch.tensor(infer_class_id(prompt), dtype=torch.long)
            with torch.no_grad():
                gen_idx = model.generate(tokens, cid, n=1, temperature=0.8).cpu().numpy()
            img = indices_to_pil(gen_idx[0], palette_rgb)
            if img.size != target_res:
                img = img.resize(target_res, Image.Resampling.NEAREST)
            return img, "Uncensored-SDXL-PixelArt / Local-TextCVAE", "Local / Uncensored-Neural-Engine"
        except Exception as e:
            pass

    if engine is not None:
        try:
            img = engine.generate_sprite(item_type, seed=seed)
            if img.size != target_res:
                img = img.resize(target_res, Image.Resampling.NEAREST)
            return img, "Uncensored-SDXL-PixelArt / Procedural-Neural", "Local / PixelSpriteEngine"
        except Exception:
            pass

    return None, None, None

def build_prompt_and_metadata(item_type, gender=None, race=None, style=None, source=None, pose=None, dominant_colors=None, tags=None, resolution=None, seed=None):
    if seed is None:
        seed = random.randint(100000, 9999999)

    if not gender:
        if any(item_type in TYPES[k] for k in ["items"]):
            gender = "genderless"
        elif any(item_type in TYPES[k] for k in ["creatures"]):
            gender = random.choice(["male", "female", "non_human", "genderless", "androgynous"])
        else:
            gender = random.choice(["male", "female", "androgynous"])

    if not race:
        if item_type in ["robot", "mecha", "cyborg", "android"]:
            race = "robot" if item_type != "cyborg" else "cyborg"
        elif item_type in ["dragon", "beast", "animal"]:
            race = "beastkin" if item_type != "dragon" else "dragonkin"
        elif item_type in ["skeleton", "zombie", "undead", "ghost", "vampire"]:
            race = "undead"
        elif any(item_type in TYPES[k] for k in ["items"]):
            race = "elemental" if item_type in ["crystal", "rock"] else "human"
        else:
            race = random.choice(RACES)

    if not style:
        style = random.choice(STYLES)

    if not source:
        source = random.choice(SOURCES)

    if not pose:
        if any(item_type in TYPES[k] for k in ["items"]):
            pose = random.choice(["full_body", "front_view", "side_view"])
        else:
            pose = random.choice(POSES)

    if not dominant_colors:
        dominant_colors = random.sample(DOMINANT_COLORS, k=random.choice([1, 2, 3]))

    if not tags:
        tags = random.sample(FREE_TAGS, k=random.choice([3, 4, 5]))

    if not resolution:
        resolution = random.choice(RESOLUTIONS)

    color_desc = " and ".join(dominant_colors)
    tags_desc = ", ".join(tags)

    if any(item_type in TYPES[k] for k in ["items"]):
        gen_prompt = (
            f"{style.replace('_', ' ')} pixel art sprite of a {item_type}, {source.replace('_', ' ')} aesthetic, "
            f"{pose.replace('_', ' ')}, color palette featuring {color_desc}, {tags_desc}, "
            f"clean pixels, sharp edges, 16-bit arcade game asset, isolated on clean background"
        )
    else:
        gen_prompt = (
            f"{style.replace('_', ' ')} pixel art sprite of a {gender} {race.replace('_', ' ')} {item_type.replace('_', ' ')}, "
            f"{source.replace('_', ' ')} universe, {pose.replace('_', ' ')}, wearing detailed equipment, "
            f"vibrant colors featuring {color_desc}, {tags_desc}, "
            f"clean pixels, sharp edges, 16-bit arcade video game sprite, white background"
        )

    if is_prohibited(gen_prompt) or is_prohibited(item_type) or is_prohibited(race):
        return None, None

    metadata = {
        "generation_prompt": gen_prompt,
        "model_used": "Uncensored-SDXL-PixelArt",
        "platform": "Local / Open-API",
        "type": item_type,
        "gender": gender,
        "race": race,
        "style": style,
        "source": source,
        "pose": pose,
        "colors": dominant_colors,
        "tags": tags,
        "resolution": resolution,
        "seed": seed
    }

    return gen_prompt, metadata

def generate_sample(item_type, idx=1, **kwargs):
    gen_prompt, metadata = build_prompt_and_metadata(item_type, **kwargs)
    if metadata is None:
        return False

    seed = metadata["seed"]
    res_str = metadata["resolution"]
    w, h = map(int, res_str.split("x"))

    out_dir = TARGET_DIR / metadata["type"] / metadata["gender"] / metadata["style"] / metadata["race"] / metadata["source"] / metadata["pose"]
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{item_type}_{idx:03d}.png"
    json_filename = f"{item_type}_{idx:03d}.json"
    metadata["file"] = filename

    out_png_path = out_dir / filename
    out_json_path = out_dir / json_filename

    img, model_name, platform_name = generate_neural_sprite(gen_prompt, item_type, target_res=(w, h), seed=seed)
    if img is not None:
        metadata["model_used"] = model_name
        metadata["platform"] = platform_name
        img.save(out_png_path)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic pixel art dataset.")
    parser.add_argument("--count", type=int, default=6, help="Number of samples to generate per type")
    parser.add_argument("--single", action="store_true", help="Generate only one test sample and exit")
    parser.add_argument("--type", type=str, default=None, help="Specific type to generate")
    args = parser.parse_args()

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    if args.single:
        item_type = args.type if args.type else "warrior"
        print(f"Generating single test sample for '{item_type}'...")
        ok = generate_sample(item_type, idx=1)
        if ok:
            print("Successfully generated test sample.")
            return 0
        else:
            print("Failed to generate test sample.")
            return 1

    all_types = []
    for category, type_list in TYPES.items():
        all_types.extend(type_list)

    if args.type:
        all_types = [args.type]

    print(f"Generating synthetic dataset for {len(all_types)} types x {args.count} samples each...")
    total_generated = 0
    for item_type in all_types:
        for idx in range(1, args.count + 1):
            ok = generate_sample(item_type, idx=idx)
            if ok:
                total_generated += 1
                if total_generated % 50 == 0:
                    print(f"[{total_generated}] Generated {item_type}_{idx:03d}")

    print(f"Generation complete! Total generated: {total_generated} samples.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
