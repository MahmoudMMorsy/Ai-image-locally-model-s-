import os
import imageio
import numpy as np
from pixel_art_engine.engine import PixelSpriteEngine

def main():
    out_dir = "examples/2026-08-29_gdrive_trained_style"
    os.makedirs(out_dir, exist_ok=True)
    engine = PixelSpriteEngine()

    characters = [
        ("char_01_paladin_knight", "Paladin Knight"),
        ("char_02_elemental_mage", "Elemental Mage"),
        ("char_03_cyber_ninja", "Cyber Ninja"),
        ("char_04_ancient_dragon", "Ancient Dragon"),
        ("char_05_heavy_mech", "Heavy Mech")
    ]

    print("Generating trained pixel art style character showcase...")
    for prefix, name in characters:
        # Single Image
        single = engine.generate_sprite(prompt=name, seed=hash(name) % 10000)
        single.save(os.path.join(out_dir, f"{prefix}_single.png"))

        # 4-frame Sprite Sheet
        frames = [engine.generate_sprite(prompt=f"{name} frame {i}", seed=(hash(name) + i) % 10000) for i in range(4)]
        sheet = engine.create_sprite_sheet(frames)
        sheet.save(os.path.join(out_dir, f"{prefix}_spritesheet.png"))

        # Animated GIF
        frames_np = [np.array(f) for f in frames]
        imageio.mimsave(os.path.join(out_dir, f"{prefix}_animation.gif"), frames_np, fps=4, loop=0)

        print(f" Generated assets for: {name}")

    print("All trained style character showcase assets saved successfully!")

if __name__ == "__main__":
    main()
