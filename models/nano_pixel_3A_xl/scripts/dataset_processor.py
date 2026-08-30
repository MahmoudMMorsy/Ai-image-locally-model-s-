"""
nano_pixel_XL0_2 Universal Dataset Preprocessing Pipeline
Slices raw sprite sheets into individual character frames, applies Nearest-Neighbor multi-scale scaling (64x64 & 128x128),
dynamically classifies ANY entity/action without hardcoded restrictions, and generates matching .txt caption files.
"""
import os
import re
from PIL import Image
import numpy as np

RAW_DIR = "models/nano_pixel_XL0_2/dataset/raw_sprites"
PROC_64_DIR = "models/nano_pixel_XL0_2/dataset/processed_64x64"
PROC_128_DIR = "models/nano_pixel_XL0_2/dataset/processed_128x128"

def dynamic_classify_file(filename: str):
    """
    Dynamically extracts entity type and action state from filename without hardcoded lists.
    Examples:
    - 'cat_run_0.png' -> ('cat', 'run')
    - 'dragon_fire_attack_2.png' -> ('dragon', 'fire_attack')
    """
    base = os.path.splitext(os.path.basename(filename))[0].lower()
    clean_name = re.sub(r'[^a-z0-9_]', '_', base)
    parts = [p for p in clean_name.split('_') if p and not p.isdigit()]

    if len(parts) >= 2:
        entity_type = parts[0]
        action_state = "_".join(parts[1:])
    elif len(parts) == 1:
        entity_type = parts[0]
        action_state = "idle"
    else:
        entity_type = "entity"
        action_state = "idle"

    return entity_type, action_state

def slice_and_process_dataset(raw_dir=RAW_DIR, out_64=PROC_64_DIR, out_128=PROC_128_DIR):
    os.makedirs(out_64, exist_ok=True)
    os.makedirs(out_128, exist_ok=True)

    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir, exist_ok=True)

    files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    print(f"[nano_pixel_XL0_2] Processing {len(files)} raw dataset files from {raw_dir}...")

    total_exported = 0
    for file in files:
        file_path = os.path.join(raw_dir, file)
        entity_type, action_state = dynamic_classify_file(file)

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
                        if np.std(arr[:, :, :3]) > 8: # keep valid sprite tiles
                            crops.append(crop)
            else:
                crops.append(im)

            for idx, crop in enumerate(crops):
                base_name = f"{entity_type}_{action_state}_{idx}"
                img_64_path = os.path.join(out_64, f"{base_name}.png")
                txt_64_path = os.path.join(out_64, f"{base_name}.txt")

                img_128_path = os.path.join(out_128, f"{base_name}.png")
                txt_128_path = os.path.join(out_128, f"{base_name}.txt")

                crop_64 = crop.resize((64, 64), Image.Resampling.NEAREST)
                crop_128 = crop.resize((128, 128), Image.Resampling.NEAREST)

                crop_64.save(img_64_path)
                crop_128.save(img_128_path)

                caption = f"pixel art, 16-bit, gba style, {entity_type}, {action_state} movement, animation frame {idx}, full body, isolated background"
                with open(txt_64_path, "w") as f:
                    f.write(caption)
                with open(txt_128_path, "w") as f:
                    f.write(caption)

                total_exported += 1

        except Exception as e:
            print(f"Error processing {file}: {e}")

    print(f"[nano_pixel_XL0_2] Dataset Pipeline finished: Exported {total_exported} character items with captions.")

if __name__ == "__main__":
    slice_and_process_dataset()
