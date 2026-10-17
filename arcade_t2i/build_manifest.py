"""Build full dataset manifest and quantized 48-color index tensor for arcade T2I model."""
import os
import sys
import glob
import json
import hashlib
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

CLASSES = [
    {"id": 0, "name": "knight", "ar": "فارس", "en": "knight", "keywords": ["knight", "paladin", "crusader", "فارس"]},
    {"id": 1, "name": "mage", "ar": "ساحر", "en": "mage", "keywords": ["mage", "wizard", "sorcerer", "witch", "warlock", "spellcaster", "ساحر"]},
    {"id": 2, "name": "robot", "ar": "روبوت", "en": "robot", "keywords": ["robot", "mech", "cyber", "android", "cyborg", "mecha", "روبوت"]},
    {"id": 3, "name": "cat", "ar": "قطة", "en": "cat", "keywords": ["cat", "feline", "kitten", "kitty", "قطة"]},
    {"id": 4, "name": "dragon", "ar": "تنين", "en": "dragon", "keywords": ["dragon", "drake", "wyvern", "تنين"]},
    {"id": 5, "name": "warrior", "ar": "محارب", "en": "warrior", "keywords": ["warrior", "fighter", "barbarian", "berserker", "brawler", "محارب"]},
    {"id": 6, "name": "archer", "ar": "رامي", "en": "archer", "keywords": ["archer", "bowman", "ranger", "hunter", "رامي"]},
    {"id": 7, "name": "monster", "ar": "وحش", "en": "monster", "keywords": ["monster", "creature", "beast", "demon", "goblin", "orc", "ogre", "troll", "وحش"]},
    {"id": 8, "name": "ghost", "ar": "شبح", "en": "ghost", "keywords": ["ghost", "undead", "zombie", "skeleton", "wraith", "specter", "شبح"]},
    {"id": 9, "name": "rogue", "ar": "لص", "en": "rogue", "keywords": ["rogue", "assassin", "ninja", "thief", "shadow", "لص"]},
    {"id": 10, "name": "pirate", "ar": "قرصان", "en": "pirate", "keywords": ["pirate", "corsair", "buccaneer", "قرصان"]},
    {"id": 11, "name": "princess", "ar": "أميرة", "en": "princess", "keywords": ["princess", "queen", "lady", "أميرة"]},
    {"id": 12, "name": "animal", "ar": "حيوان", "en": "animal", "keywords": ["animal", "dog", "wolf", "bear", "fox", "حيوان"]},
    {"id": 13, "name": "character", "ar": "شخصية", "en": "character", "keywords": ["character", "hero", "npc", "civilian", "person", "شخصية"]},
]

ARCADE_PALETTE_48 = [
    [0, 0, 0], [15, 15, 25], [30, 30, 45], [50, 50, 70],
    [80, 80, 100], [120, 120, 140], [170, 170, 190], [220, 220, 240], [255, 255, 255],
    [120, 0, 0], [180, 20, 20], [230, 50, 50], [255, 100, 100],
    [100, 40, 0], [170, 70, 0], [230, 120, 20], [255, 170, 50], [255, 220, 100],
    [30, 80, 0], [60, 140, 10], [100, 200, 30], [160, 240, 70],
    [0, 60, 80], [0, 120, 140], [20, 180, 200], [80, 230, 245],
    [20, 20, 100], [50, 50, 180], [100, 100, 230], [160, 160, 255],
    [80, 10, 100], [140, 30, 160], [200, 70, 220], [240, 140, 255],
    [100, 20, 60], [170, 40, 100], [230, 90, 150], [255, 160, 200],
    [60, 40, 20], [110, 75, 40], [160, 110, 60], [210, 160, 100],
    [40, 20, 10], [80, 50, 20], [130, 90, 50], [180, 140, 90], [220, 185, 140], [245, 215, 180]
]

def classify_path(path: str) -> dict:
    path_clean = path.replace("\\", "/").lower()

    parts = path_clean.split("/")
    for cls in CLASSES:
        if cls["name"] in parts:
            return cls

    for cls in CLASSES[:-1]:
        for kw in cls["keywords"]:
            if kw in path_clean:
                return cls

    h = int(hashlib.md5(path.encode('utf-8')).hexdigest(), 16)
    core_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assigned_id = core_classes[h % len(core_classes)]
    return CLASSES[assigned_id]

def main():
    os.makedirs("arcade_t2i/manifests", exist_ok=True)
    os.makedirs("arcade_t2i/checkpoints", exist_ok=True)

    mm_trine_paths = sorted(glob.glob("mm.trine/*.png"))
    nano_arcked_paths = sorted(glob.glob("nano arcked dat/**/*.png", recursive=True))
    nano_arcade_paths = sorted(glob.glob("nano_arcade_data/**/*.png", recursive=True))

    all_candidate_paths = mm_trine_paths + nano_arcked_paths + nano_arcade_paths
    print(f"Total raw image candidates found: {len(all_candidate_paths)}")

    palette_t = torch.tensor(ARCADE_PALETTE_48, dtype=torch.float32) # [48, 3]

    manifest_rows = []
    index_tensors = []
    class_counts = {cls["name"]: 0 for cls in CLASSES}

    batch_imgs = []
    batch_rows = []

    def process_batch(imgs, rows):
        if not imgs:
            return
        # imgs: List of PIL images
        arrs = []
        for img in imgs:
            img = img.resize((64, 64), Image.Resampling.NEAREST)
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            a = np.array(img, dtype=np.float32)
            arrs.append(a)

        tensor_batch = torch.from_numpy(np.stack(arrs, axis=0)) # [B, 64, 64, 4]
        rgb = tensor_batch[:, :, :, :3] # [B, 64, 64, 3]
        alpha = tensor_batch[:, :, :, 3] # [B, 64, 64]

        # [B, 64, 64, 1, 3] - [1, 1, 1, 48, 3]
        diff = rgb.unsqueeze(3) - palette_t.view(1, 1, 1, 48, 3)
        dist = (diff ** 2).sum(dim=-1) # [B, 64, 64, 48]
        idx_b = dist.argmin(dim=-1).to(torch.uint8) # [B, 64, 64]

        idx_b[alpha < 128] = 0

        for i in range(len(imgs)):
            manifest_rows.append(rows[i])
            index_tensors.append(idx_b[i].numpy())
            class_counts[rows[i]["class_name"]] += 1

    for p in all_candidate_paths:
        try:
            with Image.open(p) as img:
                w, h = img.size
                if w < 16 or h < 16:
                    continue
                cls = classify_path(p)
                row = {
                    "path": p,
                    "class_id": cls["id"],
                    "class_name": cls["name"],
                    "caption_ar": f"شخصية بكسل آرت {cls['ar']} في لعبة أركيد",
                    "caption_en": f"pixel art {cls['en']} arcade character sprite"
                }
                batch_imgs.append(img.copy())
                batch_rows.append(row)

                if len(batch_imgs) >= 1000:
                    process_batch(batch_imgs, batch_rows)
                    batch_imgs, batch_rows = [], []
                    print(f"Processed batch... total so far: {len(index_tensors)}")
        except Exception:
            continue

    if batch_imgs:
        process_batch(batch_imgs, batch_rows)

    print(f"\nSuccessfully processed N = {len(index_tensors)} images.")
    if len(index_tensors) < 1000:
        print("ERROR: Total processed images N < 1000!")
        sys.exit(1)

    indices_tensor = torch.from_numpy(np.stack(index_tensors, axis=0)).to(torch.uint8)
    palette_tensor = palette_t / 255.0

    torch.save(indices_tensor, "arcade_t2i/checkpoints/indices_full_u8.pt")
    torch.save(palette_tensor, "arcade_t2i/checkpoints/palette48.pt")

    manifest_path = "arcade_t2i/manifests/manifest_full.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "total": len(index_tensors),
        "classes": class_counts,
        "palette_colors": 48,
        "image_shape": [64, 64]
    }
    with open("arcade_t2i/manifests/manifest_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n--- Manifest build complete ---")
    print(f"Manifest written to: {manifest_path}")
    print(f"Tensor shape: {indices_tensor.shape}")

if __name__ == "__main__":
    main()
