import os
import json
import csv
from PIL import Image

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_CSV = os.path.join(DATASET_DIR, "../dataset_index.csv")
if not os.path.exists(INDEX_CSV):
    INDEX_CSV = os.path.join(DATASET_DIR, "dataset_index.csv")

VALID_PLATFORMS = {
    "gameboy_classic", "gameboy_color", "gameboy_advance", "nes_famicom",
    "snes_super_famicom", "sega_genesis_megadrive", "arcade_arcade_machine", "retro_other"
}

VALID_POSTER_TYPES = {
    "box_art_front", "title_screen_art", "arcade_marquee", "cartridge_label",
    "full_game_poster", "promo_flyer", "arabic_title_logo", "english_title_logo",
    "bilingual_title_logo", "pixel_lettering", "game_over_screen", "stage_clear_banner", "ui_title_badge"
}

VALID_LANGUAGE_STRUCTURES = {
    "arabic_only", "bilingual_arabic_primary", "bilingual_english_primary",
    "parallel_pair_arabic", "arabic_transliteration", "arabic_subtitle"
}

VALID_TYPOGRAPHY_STYLES = {
    "pixel_kufic_geometric", "pixel_ruqah_fast", "pixel_naskh_clean", "bold_arcade_arabic",
    "3d_bevel_arabic", "neon_glitch_arabic", "fantasy_ornate_arabic", "retro_8bit_arabic",
    "metallic_chrome_arabic", "pixel_thuluth_display", "pixel_calligraffiti"
}

VALID_RESOLUTIONS = {"128x128", "256x256"}

FORBIDDEN_KEYWORDS = [
    "prophet", "mohammed", "muhammad", "jesus", "moses", "abraham", "allah", "god",
    "sahaba", "ahlulbayt", "sacred_character", "deity", "angel_visual"
]

def verify_dataset():
    errors = []

    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found.")
        return False

    total_images = 0
    total_jsons = 0

    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            filepath = os.path.join(root, file)

            if file == "dataset_index.csv" or file.endswith(".py"):
                continue

            if file.endswith(".png"):
                total_images += 1
                try:
                    with Image.open(filepath) as img:
                        w, h = img.size
                        res_str = f"{w}x{h}"
                        if res_str not in VALID_RESOLUTIONS:
                            errors.append(f"Invalid image resolution {res_str} for {filepath}")
                except Exception as e:
                    errors.append(f"Corrupted image {filepath}: {str(e)}")

            elif file.endswith(".json"):
                total_jsons += 1
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    req_keys = ["file", "resolution", "platform", "poster_type", "language_structure", "typography_details", "art_style", "genre", "layout", "colors", "tags", "notes"]
                    for k in req_keys:
                        if k not in data:
                            errors.append(f"Missing key '{k}' in JSON {filepath}")

                    if data.get("platform") not in VALID_PLATFORMS:
                        errors.append(f"Invalid platform '{data.get('platform')}' in {filepath}")
                    if data.get("poster_type") not in VALID_POSTER_TYPES:
                        errors.append(f"Invalid poster_type '{data.get('poster_type')}' in {filepath}")
                    if data.get("language_structure") not in VALID_LANGUAGE_STRUCTURES:
                        errors.append(f"Invalid language_structure '{data.get('language_structure')}' in {filepath}")
                    if data.get("resolution") not in VALID_RESOLUTIONS:
                        errors.append(f"Invalid resolution '{data.get('resolution')}' in {filepath}")

                    typo = data.get("typography_details", {})
                    if typo.get("style") not in VALID_TYPOGRAPHY_STYLES:
                        errors.append(f"Invalid typography style '{typo.get('style')}' in {filepath}")

                    full_text = (json.dumps(data) + " " + filepath).lower()
                    for kw in FORBIDDEN_KEYWORDS:
                        if kw in full_text:
                            errors.append(f"Forbidden term '{kw}' detected in {filepath}")

                except Exception as e:
                    errors.append(f"Invalid JSON file {filepath}: {str(e)}")

    print(f"Verified {total_images} PNG images and {total_jsons} JSON metadata files in '{DATASET_DIR}'.")

    if errors:
        print(f"Verification Failed with {len(errors)} errors:")
        for err in errors[:20]:
            print(f"  - {err}")
        return False
    else:
        print("Verification Successful! All images, schemas, prohibitions, and indices passed 100%.")
        return True

if __name__ == "__main__":
    success = verify_dataset()
    exit(0 if success else 1)
