import os
import imageio
import numpy as np
from pixel_art_engine.engine import PixelSpriteEngine

def main():
    out_dir = "models/nano_pixel_3A_xl/showcase_high_res"
    os.makedirs(out_dir, exist_ok=True)
    engine = PixelSpriteEngine()

    characters = [
        ("char_01_paladin_knight", "Paladin Knight"),
        ("char_02_elemental_mage", "Elemental Mage"),
        ("char_03_cyber_ninja", "Cyber Ninja"),
        ("char_04_ancient_dragon", "Ancient Dragon"),
        ("char_05_heavy_mech", "Heavy Mech")
    ]

    for prefix, name in characters:
        print(f"Generating high-res assets for: {name}...")

        # 1. Single Image PNG (64x64 or 128x128)
        single = engine.generate_sprite(prompt=name, seed=hash(name) % 10000)
        single_path = os.path.join(out_dir, f"{prefix}_single.png")
        single.save(single_path)

        # 2. 4-frame Sprite Sheet PNG
        frames = [engine.generate_sprite(prompt=f"{name} frame {i}", seed=(hash(name) + i) % 10000) for i in range(4)]
        sheet = engine.create_sprite_sheet(frames)
        sheet_path = os.path.join(out_dir, f"{prefix}_spritesheet.png")
        sheet.save(sheet_path)

        # 3. Loopable Animated GIF
        gif_path = os.path.join(out_dir, f"{prefix}_animation.gif")
        frames_np = [np.array(f) for f in frames]
        imageio.mimsave(gif_path, frames_np, fps=4, loop=0)

        print(f" Saved: {single_path}, {sheet_path}, {gif_path}")

    print("\nAll 15 high-res character assets (5 characters x 3 formats) created successfully!")

if __name__ == "__main__":
    main()
