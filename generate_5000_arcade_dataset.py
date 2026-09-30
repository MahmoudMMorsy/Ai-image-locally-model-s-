import os
import time
import torch
import numpy as np
from PIL import Image
from pixel_art_engine.procedural import generate_arcade_sprite
from pixel_art_engine.palette import quantize_to_pixel_art, GAMEBOY_PALETTE, NES_PALETTE, SIGNATURE_PALETTE

ARCHETYPES = ["knight", "wizard", "monster", "robot", "elf", "ninja", "cyber_warrior", "orc"]
THEMES = ["red", "blue", "green", "gold", "purple", "dark"]
POSES = ["idle", "walk", "run", "attack", "jump"]
PALETTES = [GAMEBOY_PALETTE, NES_PALETTE, SIGNATURE_PALETTE]

def generate_5000_arcade_dataset():
    out_dir = "dataset_5k"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating 5,000+ side-view arcade game character sprite images into {out_dir}...")
    start_time = time.time()

    count = 0
    target_count = 5000

    while count < target_count:
        archetype = ARCHETYPES[count % len(ARCHETYPES)]
        theme = THEMES[(count // len(ARCHETYPES)) % len(THEMES)]
        pose = POSES[(count // (len(ARCHETYPES) * len(THEMES))) % len(POSES)]
        frame = count % 4
        palette = PALETTES[count % len(PALETTES)]

        # Procedural side-view arcade character generation
        sprite = generate_arcade_sprite(archetype=archetype, color_theme=theme, pose=pose, frame=frame)

        # Add side-view noise / variation for dataset richness
        arr = np.array(sprite, dtype=np.int16)
        if count % 3 == 0:
            # Slight hue/intensity shift for pixel variation
            shift = (np.random.randint(-15, 15, size=arr.shape, dtype=np.int16))
            arr = np.clip(arr + shift, 0, 255).astype(np.uint8)
            sprite = Image.fromarray(arr, mode="RGBA")

        # Quantize to target retro palette (Game Boy, NES, or Signature)
        quantized_sprite = quantize_to_pixel_art(sprite, size=(64, 64), palette=palette)

        filename = f"sideview_arcade_{count+1:05d}.png"
        filepath = os.path.join(out_dir, filename)
        quantized_sprite.save(filepath)

        count += 1
        if count % 1000 == 0 or count == target_count:
            print(f" Progress: [{count}/{target_count}] images generated...")

    elapsed = time.time() - start_time
    print(f"Successfully generated {count} side-view arcade character images in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    generate_5000_arcade_dataset()
