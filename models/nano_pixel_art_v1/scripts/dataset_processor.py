"""
Automated Dataset Preprocessing Pipeline for NanoPixel-v1
Slices sprite sheets into individual character frames and resizes to 64x64 and 128x128 using Nearest-Neighbor.
"""
import os
from PIL import Image
import numpy as np

RAW_DIR = "models/nano_pixel_art_v1/dataset/raw_sprites"
PROC_64_DIR = "models/nano_pixel_art_v1/dataset/processed_64x64"
PROC_128_DIR = "models/nano_pixel_art_v1/dataset/processed_128x128"

CHAR_CLASSES = ["knight", "mage", "wizard", "orc", "goblin", "cyber_rogue", "rogue", "robot", "mecha", "monster", "cat", "dog", "char", "hero", "warrior"]
ACTIONS = ["idle", "walk", "run", "jump", "attack", "dash", "hit"]

def parse_filename_tags(filename: str):
    fname_lower = filename.lower()
    char_type = "character"
    for c in CHAR_CLASSES:
        if c in fname_lower:
            char_type = c
            break
    action_state = "idle"
    for a in ACTIONS:
        if a in fname_lower:
            action_state = a
            break
    return char_type, action_state

def slice_and_process_dataset(raw_dir=RAW_DIR, out_64=PROC_64_DIR, out_128=PROC_128_DIR):
    os.makedirs(out_64, exist_ok=True)
    os.makedirs(out_128, exist_ok=True)

    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir, exist_ok=True)

    # Clean existing
    for d in [out_64, out_128]:
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))

    files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    print(f"Processing {len(files)} raw dataset files from {raw_dir}...")

    total_exported = 0
    for file in files:
        file_path = os.path.join(raw_dir, file)
        char_type, action_state = parse_filename_tags(file)

        try:
            im = Image.open(file_path).convert("RGBA")
            w, h = im.size

            crops = []
            if w > 128 or h > 128:
                tile_w = 64 if w <= 512 else 128
                tile_h = 64 if h <= 512 else 128

                for y in range(0, h - tile_h + 1, tile_h):
                    for x in range(0, w - tile_w + 1, tile_w):
                        box = (x, y, x + tile_w, y + tile_h)
                        crop = im.crop(box)
                        arr = np.array(crop)
                        if np.std(arr[:, :, :3]) > 5:
                            crops.append(crop)
            else:
                crops.append(im)

            for idx, crop in enumerate(crops):
                base_name = f"{char_type}_{action_state}_{total_exported}_{idx}"
                img_64_path = os.path.join(out_64, f"{base_name}.png")
                img_128_path = os.path.join(out_128, f"{base_name}.png")

                crop_64 = crop.resize((64, 64), Image.Resampling.NEAREST)
                crop_128 = crop.resize((128, 128), Image.Resampling.NEAREST)

                crop_64.save(img_64_path)
                crop_128.save(img_128_path)

                total_exported += 1

        except Exception as e:
            print(f"Failed to process {file}: {e}")

    print(f"Dataset Pipeline complete: Processed and exported {total_exported} character items.")

if __name__ == "__main__":
    slice_and_process_dataset()
