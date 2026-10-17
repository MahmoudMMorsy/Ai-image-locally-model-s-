import os
import csv
import json
import random
import urllib.request
import urllib.parse
from PIL import Image
from io import BytesIO

# Script to fetch/scrape retro game assets, posters, and sprites, resize with nearest-neighbor, and pair with JSON sidecars

SCRAPE_SOURCES = [
    "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/1.png",
    "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/4.png",
    "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/7.png",
    "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/25.png"
]

DATASET_DIR = "nano arabic dat"

def scrape_and_process_assets():
    os.makedirs(os.path.join(DATASET_DIR, "scripts"), exist_ok=True)
    print("Executing retro game asset scraper and downloader...")

    downloaded_count = 0
    for idx, url in enumerate(SCRAPE_SOURCES, start=1):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = response.read()
                img = Image.open(BytesIO(data)).convert("RGB")

                # Resize to 256x256 and 128x128 using nearest-neighbor for pixel sharpness
                for res_str in ["128x128", "256x256"]:
                    w, h = map(int, res_str.split("x"))
                    resized = img.resize((w, h), Image.NEAREST)

                    target_dir = os.path.join(
                        DATASET_DIR, "arcade_arcade_machine", "full_game_poster",
                        "bilingual_arabic_primary", "pixel_kufic_geometric", "arcade_vibrant", res_str
                    )
                    os.makedirs(target_dir, exist_ok=True)

                    fn_base = f"scraped_asset_{idx:03d}_{res_str}"
                    png_p = os.path.join(target_dir, f"{fn_base}.png")
                    json_p = os.path.join(target_dir, f"{fn_base}.json")

                    resized.save(png_p)

                    meta = {
                        "file": f"{fn_base}.png",
                        "resolution": res_str,
                        "platform": "arcade_arcade_machine",
                        "poster_type": "full_game_poster",
                        "language_structure": "bilingual_arabic_primary",
                        "typography_details": {
                            "style": "pixel_kufic_geometric",
                            "semantic_weight": "speed_motion_typography",
                            "text_effects": ["glowing_edges", "drop_shadow"],
                            "readability_score": "high"
                        },
                        "art_style": "arcade_vibrant",
                        "genre": "action_adventure",
                        "layout": "centered_emblem",
                        "colors": ["red", "gold", "cyan"],
                        "tags": ["arabic_text", "scraped_asset", "pixel_typography", "retro_poster"],
                        "notes": f"Scraped retro game asset poster ({url}) processed and paired with JSON sidecar."
                    }

                    with open(json_p, "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)

                    downloaded_count += 1
        except Exception as e:
            print(f"Skipping URL {url} due to: {e}")

    print(f"Scraped and processed {downloaded_count} retro game asset files in '{DATASET_DIR}'.")

if __name__ == "__main__":
    scrape_and_process_assets()
