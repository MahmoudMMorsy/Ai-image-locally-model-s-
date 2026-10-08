#!/usr/bin/env python3
"""
organize_nano_arcked_dataset.py

Aggregates, classifies, enriches, and structures pixel art image datasets
into nano arcked dat/data/<type>/<gender>/<style>/<race>/<source>/<pose>/
with matching sidecar .json files, master dataset_index.csv, and manifests.
Enforces zero-tolerance sacred religious prohibition filtering.
"""

import os
import re
import csv
import json
import glob
import shutil
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image

ROOT_DIR = Path("nano arcked dat")
DATA_DIR = ROOT_DIR / "data"
MANIFEST_DIR = ROOT_DIR / "manifests"
INDEX_CSV = ROOT_DIR / "dataset_index.csv"
JSONL_PATH = MANIFEST_DIR / "dataset.jsonl"
QUARANTINE_PATH = MANIFEST_DIR / "quarantine.jsonl"

# Sacred religious prohibition terms (zero-tolerance policy)
EXCLUDE_TERMS = [
    "allah", "god_supreme", "muhammad", "mohammed", "prophet", "messenger",
    "jesus", "moses", "abraham", "noah", "ibrahim", "musa", "nuh", "isa",
    "abu-bakr", "abu_bakr", "umar", "uthman", "ali",
    "kaaba", "quran", "koran", "mecca", "medina",
    "sacred_symbol", "holy_prophet", "religious_figure", "sahaba"
]

COLOR_MAP = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 255, 0),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gold": (255, 215, 0),
    "silver": (192, 192, 192),
    "purple": (128, 0, 128),
    "orange": (255, 165, 0),
    "pink": (255, 192, 203),
    "brown": (165, 42, 42),
    "dark": (30, 30, 30),
    "bright": (240, 240, 240)
}

TYPE_RULES = [
    # Specific Characters & Roles
    ("cyber_soldier", ["cyber_soldier", "cybersoldier", "sci_fi_soldier", "cyber_infantry", "cyber_warrior"]),
    ("gunslinger", ["gunslinger", "gunner", "shooter", "pistol", "sniper", "rifleman", "desperado"]),
    ("berserker", ["berserker", "barbarian_fury", "rager"]),
    ("paladin", ["paladin", "holy_knight", "crusader"]),
    ("necromancer", ["necromancer", "death_mage", "lich_lord"]),
    ("summoner", ["summoner", "conjurer", "evoker"]),
    ("bard", ["bard", "minstrel", "musician", "troubadour"]),
    ("thief", ["thief", "pickpocket", "burglar"]),
    ("hunter", ["hunter", "trapper", "tracker"]),
    ("samurai", ["samurai", "ronin", "bushido", "kenshi"]),
    ("ninja", ["ninja", "shinobi", "kunoichi"]),
    ("pirate", ["pirate", "buccaneer", "corsair", "swashbuckler"]),
    ("viking", ["viking", "norse_warrior", "berserk"]),
    ("gladiator", ["gladiator", "arena_champion"]),
    ("soldier", ["soldier", "commando", "infantry", "trooper", "marine", "guard", "sentry"]),
    ("knight", ["knight", "swordsman", "shield_knight", "templar"]),
    ("warrior", ["warrior", "fighter", "brawler", "barbarian", "champion"]),
    ("mage", ["mage", "wizard", "sorcerer", "witch", "alchemist", "priest", "spellcaster", "enchanter", "scholar"]),
    ("archer", ["archer", "bowman", "ranger", "crossbow", "marksman"]),
    ("rogue", ["rogue", "assassin", "scout", "shadow"]),
    ("monk", ["monk", "martial_artist", "shaolin"]),

    # Creatures & Species
    ("werewolf", ["werewolf", "lycanthrope", "wolfman"]),
    ("vampire", ["vampire", "dracula", "bloodsucker"]),
    ("zombie", ["zombie", "undead_runner", "walker"]),
    ("skeleton", ["skeleton", "skeletal", "bone_warrior"]),
    ("ghost", ["ghost", "phantom", "specter", "wraith", "poltergeist"]),
    ("dragon", ["dragon", "drake", "wyvern", "dragonkin"]),
    ("demon", ["demon", "devil", "fiend", "imp", "succubus"]),
    ("angel", ["angel", "seraph", "cherub", "archangel"]),
    ("undead", ["undead", "lich", "ghoul", "mummy"]),
    ("golem", ["golem", "automaton", "construct"]),
    ("elemental", ["elemental", "fire_elemental", "ice_elemental", "earth_elemental", "water_elemental"]),
    ("fairy", ["fairy", "pixie", "nymph", "sprite_fairy"]),
    ("mermaid", ["mermaid", "merfolk", "siren", "triton"]),
    ("centaur", ["centaur"]),
    ("minotaur", ["minotaur"]),
    ("goblin", ["goblin", "hobgoblin"]),
    ("orc", ["orc", "uruk"]),
    ("troll", ["troll", "cave_troll"]),
    ("giant", ["giant", "titan", "colossus"]),
    ("elf", ["elf", "high_elf", "wood_elf"]),
    ("dwarf", ["dwarf", "dwarven"]),
    ("beast", ["beast", "monster", "creature", "chimera", "hydra", "basilisk", "behemoth"]),
    ("animal", ["animal", "cat", "dog", "shiba", "husky", "dalmatian", "bear", "wolf", "bird", "horse", "frog", "slime", "dragonfly", "owl", "tiger", "lion", "snake"]),
    ("robot", ["robot", "mecha", "cyborg", "android", "droid", "mech"]),

    # Civilians & Roles
    ("merchant", ["merchant", "trader", "peddler", "shopkeeper"]),
    ("noble", ["noble", "lord", "lady", "baron"]),
    ("king", ["king", "emperor"]),
    ("queen", ["queen", "empress"]),
    ("prince", ["prince"]),
    ("princess", ["princess"]),
    ("child", ["child", "kid", "boy", "girl"]),
    ("elder", ["elder", "old_man", "old_woman", "grandpa", "grandma"]),
    ("farmer", ["farmer", "peasant"]),
    ("blacksmith", ["blacksmith", "smith"]),
    ("alchemist", ["alchemist", "potion_maker"]),
    ("priest", ["priest", "cleric", "monk_priest"]),
    ("witch", ["witch", "hag"]),
    ("wizard", ["wizard", "archmage"]),
    ("scholar", ["scholar", "scribe"]),
    ("dancer", ["dancer"]),
    ("cook", ["cook", "chef"]),
    ("guard", ["guard", "sentry", "warden"]),
    ("civilian", ["civilian", "villager", "npc", "innkeeper", "citizen"]),

    # Objects & Non-living & Posters
    ("poster", ["poster", "arcade_poster", "game_poster", "typography_poster"]),
    ("typography", ["typography", "arabic_typography", "retro_text", "pixel_font"]),
    ("weapon", ["weapon", "sword", "bow", "gun", "axe", "staff", "shield", "helmet", "armor", "dagger", "blade", "spear", "mace", "wand"]),
    ("vehicle", ["vehicle", "spaceship", "car", "tank", "ship", "aircraft", "mech_unit"]),
    ("structure", ["castle", "building", "house", "tower", "dungeon", "structure", "door", "portal"]),
    ("item", ["item", "potion", "chest", "crystal", "rock", "tree", "plant", "furniture", "food", "coin", "relic", "scroll", "ring", "amulet"])
]

RACE_RULES = [
    ("robot", ["robot", "mecha", "cyborg", "android", "droid", "automaton"]),
    ("cyborg", ["cyborg", "augmented"]),
    ("dragonkin", ["dragon", "drake", "wyvern", "dragonkin"]),
    ("undead", ["undead", "zombie", "skeleton", "vampire", "ghost", "lich", "ghoul", "specter", "mummy"]),
    ("demon", ["demon", "devil", "fiend", "imp"]),
    ("angel", ["angel", "seraph"]),
    ("dark_elf", ["dark_elf", "drow"]),
    ("high_elf", ["high_elf"]),
    ("elf", ["elf", "wood_elf"]),
    ("dwarf", ["dwarf", "dwarven"]),
    ("orc", ["orc", "uruk"]),
    ("goblin", ["goblin"]),
    ("troll", ["troll"]),
    ("beastkin", ["beastkin", "werewolf", "minotaur", "centaur", "cat_person", "dog_person"]),
    ("animal_like", ["cat", "dog", "animal", "shiba", "husky", "dalmatian", "bear", "bird", "horse", "frog", "slime"]),
    ("elemental", ["elemental", "golem", "flame_entity", "ice_entity"]),
    ("fairy", ["fairy", "pixie"]),
    ("merfolk", ["mermaid", "merfolk", "siren"]),
    ("giant", ["giant", "titan"]),
    ("alien", ["alien", "xenomorph"]),
    ("hybrid", ["hybrid", "chimera", "half_elf", "half_orc"]),
    ("non_human", ["poster", "typography", "weapon", "structure", "item", "vehicle"]),
    ("human", ["human", "person", "man", "woman", "guy", "girl", "warrior", "knight", "mage", "archer", "soldier"])
]

STYLE_RULES = [
    ("retro_8bit", ["8bit", "8-bit", "nes", "pico8", "pico-8", "gameboy", "gb", "gameboy_classic"]),
    ("retro_16bit", ["16bit", "16-bit", "snes", "genesis", "gba", "gameboy_advance", "megadrive"]),
    ("arcade_pixel", ["arcade", "brawler", "capcom", "neogeo", "cps2", "cps1", "arcade_pixel"]),
    ("realistic_pixel", ["realistic", "detailed_pixel", "hd_pixel", "realistic_pixel"]),
    ("anime_pixel", ["anime", "manga", "jrpg", "anime_pixel"]),
    ("chibi", ["chibi", "mini", "cute"]),
    ("cartoon", ["cartoon", "toon"]),
    ("dark_fantasy", ["dark_fantasy", "dark_souls", "gothic", "horror_pixel"]),
    ("fantasy_pixel", ["fantasy", "rpg_pixel", "magic_pixel"]),
    ("sci_fi_pixel", ["sci_fi", "cyberpunk", "mecha_pixel", "futuristic"]),
    ("clean_pixel", ["clean", "simple", "flat", "pixel"]),
    ("detailed_pixel", ["detailed", "highres_pixel", "poster"])
]

SOURCE_RULES = [
    ("cyberpunk", ["cyberpunk", "cyber", "neon", "futuristic"]),
    ("dark_fantasy", ["dark_fantasy", "dark", "gothic", "demon", "undead"]),
    ("rpg_fantasy", ["fantasy", "rpg", "magic", "medieval", "kingdom"]),
    ("sci_fi", ["sci_fi", "space", "alien", "spaceship"]),
    ("steampunk", ["steampunk", "steam", "victorian"]),
    ("horror", ["horror", "creepy", "haunted"]),
    ("mythology_greek", ["greek", "olympus", "zeus", "minotaur", "centaur"]),
    ("mythology_norse", ["norse", "valhalla", "thor", "viking"]),
    ("mythology_egypt", ["egypt", "pharaoh", "anubis", "mummy"]),
    ("historical", ["historical", "medieval", "samurai", "ninja"]),
    ("post_apocalyptic", ["post_apocalyptic", "wasteland", "fallout"]),
    ("game_original", ["game", "sprite", "arcade", "retro", "cc0", "trine"]),
    ("generic", [])
]

POSE_RULES = [
    ("running", ["running", "run"]),
    ("walking", ["walking", "walk"]),
    ("attacking", ["attacking", "attack", "slash", "shoot", "cast", "strike", "swing"]),
    ("casting", ["casting", "cast_spell", "spellcast"]),
    ("action", ["action", "combating", "fight", "jump", "dash", "dodge"]),
    ("idle", ["idle", "standing", "hero_pose"]),
    ("sitting", ["sitting", "crouching", "kneeling"]),
    ("flying", ["flying", "hovering", "soaring"]),
    ("lying", ["lying", "down", "defeated"]),
    ("side_view", ["side_view", "side", "profile"]),
    ("front_view", ["front_view", "front"]),
    ("back_view", ["back_view", "back"]),
    ("portrait", ["portrait", "bust", "face", "headshot"]),
    ("full_body", ["full_body", "fullbody", "character", "sprite"])
]

def norm_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

def is_blocked(text: str) -> bool:
    n = norm_text(text)
    for term in EXCLUDE_TERMS:
        pattern = r"\b" + re.escape(term.replace("_", " ").replace("-", " ")) + r"\b"
        if re.search(pattern, n):
            return True
    return False

def extract_dominant_colors(image_path: Path, top_k: int = 3) -> list[str]:
    try:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            im.thumbnail((32, 32))
            colors = im.getcolors(maxcolors=1024)
            if not colors:
                return ["multicolor"]
            colors.sort(key=lambda x: x[0], reverse=True)
            detected = []
            for count, (r, g, b) in colors[:10]:
                if count < 5:
                    continue
                best_color = "dark"
                min_dist = float("inf")
                for c_name, (cr, cg, cb) in COLOR_MAP.items():
                    dist = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        best_color = c_name
                if best_color not in detected:
                    detected.append(best_color)
                if len(detected) >= top_k:
                    break
            return detected or ["multicolor"]
    except Exception:
        return ["multicolor"]

def infer_category(text: str, rules: list, default: str) -> str:
    n = norm_text(text)
    for label, keywords in rules:
        if any(kw in n for kw in keywords):
            return label
    return default

def infer_gender(text: str) -> str:
    n = norm_text(text)
    if any(w in n for w in ["female", "woman", "girl", "lady", "queen", "princess", "witch", "heroine", "sorceress", "amazon"]):
        return "female"
    if any(w in n for w in ["male", "man", "boy", "guy", "king", "prince", "wizard", "hero", "lord", "knight"]):
        return "male"
    if any(w in n for w in ["robot", "mecha", "weapon", "structure", "item", "vehicle", "dragon", "chest", "potion", "poster", "typography"]):
        return "genderless"
    return "androgynous"

def calculate_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def extract_tags(text: str) -> list[str]:
    words = re.split(r"[/_ -]+", norm_text(text))
    ignore = {"png", "jpg", "jpeg", "webp", "gif", "image", "file", "data", "dataset", "asset", "tile", "frame"}
    clean_tags = [w for w in words if len(w) > 2 and w not in ignore]
    seen = set()
    tags = []
    for t in clean_tags:
        if t not in seen:
            seen.add(t)
            tags.append(t)
    return tags[:25]

def gather_all_image_candidates() -> list[tuple[str, Path]]:
    candidates = []

    sources = [
        ("dataset_gdrive", Path(".staging_gdrive")),
        ("mm_trine", Path("mm.trine")),
        ("dataset_clean", Path(".staging_clean")),
        ("dataset_raw", Path(".staging_raw")),
        ("all_data_trine", Path("All-data-Trine")),
        ("nano_arcade_data", Path("nano_arcade_data")),
        ("nano_pixel_mob", Path("nano_pixel_mob")),
        ("examples_showcase", Path("examples")),
        ("cc0_collected", Path("nano arcked dat/images"))
    ]

    for source_name, source_path in sources:
        if source_path.exists():
            for p in source_path.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
                    candidates.append((source_name, p))

    return candidates

def main():
    print("Starting dataset aggregation and taxonomy classification...")
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    candidates = gather_all_image_candidates()
    print(f"Discovered {len(candidates)} candidate image files across sources.")

    processed_hashes = set()
    records = []
    quarantine_records = []

    for source_name, img_path in candidates:
        rel_path_str = str(img_path)
        if is_blocked(rel_path_str):
            quarantine_records.append({
                "file": rel_path_str,
                "reason": "Exclusion keyword match (sacred religious prohibition rule)"
            })
            continue

        digest = calculate_sha256(img_path)
        if digest in processed_hashes:
            continue
        processed_hashes.add(digest)

        asset_id = digest[:16]

        res_str = "64x64"
        try:
            with Image.open(img_path) as im:
                res_str = f"{im.width}x{im.height}"
        except Exception:
            continue

        combined_text = f"{source_name} {rel_path_str}"

        c_type = infer_category(combined_text, TYPE_RULES, "warrior")
        c_gender = infer_gender(combined_text)
        c_race = infer_category(combined_text, RACE_RULES, "human" if c_gender in ["male", "female", "androgynous"] else "non_human")
        c_style = infer_category(combined_text, STYLE_RULES, "arcade_pixel")
        c_source = infer_category(combined_text, SOURCE_RULES, "rpg_fantasy")
        c_pose = infer_category(combined_text, POSE_RULES, "full_body")

        dominant_colors = extract_dominant_colors(img_path)
        tags = extract_tags(combined_text)

        target_dir = DATA_DIR / c_type / c_gender / c_style / c_race / c_source / c_pose
        target_dir.mkdir(parents=True, exist_ok=True)

        filename_png = f"{asset_id}_{c_type}.png"
        filename_json = f"{asset_id}_{c_type}.json"

        dst_png_path = target_dir / filename_png
        dst_json_path = target_dir / filename_json

        shutil.copy2(img_path, dst_png_path)

        sidecar_meta = {
            "file": filename_png,
            "type": c_type,
            "gender": c_gender,
            "race": c_race,
            "style": c_style,
            "source": c_source,
            "pose": c_pose,
            "colors": dominant_colors,
            "tags": tags,
            "resolution": res_str,
            "notes": f"Aggregated asset from {source_name} ({res_str} {c_style} {c_type})"
        }

        with open(dst_json_path, "w", encoding="utf-8") as f:
            json.dump(sidecar_meta, f, ensure_ascii=False, indent=2)

        record = {
            "id": asset_id,
            "file": dst_png_path.relative_to(ROOT_DIR).as_posix(),
            "type": c_type,
            "gender": c_gender,
            "race": c_race,
            "style": c_style,
            "source": c_source,
            "pose": c_pose,
            "colors": ",".join(dominant_colors),
            "resolution": res_str,
            "tags": ",".join(tags),
            "sha256": digest,
            "original_source": source_name,
            "original_path": rel_path_str,
            "processed_at": datetime.now(timezone.utc).isoformat()
        }
        records.append(record)

    print(f"Successfully processed {len(records)} unique pixel art images.")
    print(f"Quarantined {len(quarantine_records)} excluded items.")

    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(QUARANTINE_PATH, "w", encoding="utf-8") as f:
        for q in quarantine_records:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    csv_fields = ["id", "file", "type", "gender", "race", "style", "source", "pose", "colors", "resolution", "tags", "sha256", "original_source", "original_path", "processed_at"]
    with open(INDEX_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(records)

    print(f"Updated {INDEX_CSV} and {JSONL_PATH} successfully.")

if __name__ == "__main__":
    main()
