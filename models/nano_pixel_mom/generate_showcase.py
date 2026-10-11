import os
import datetime
import imageio
import numpy as np
from models.nano_pixel_mom.inference_engine import NanoPixelMomEngine

def main():
    date_str = datetime.date.today().strftime("%Y-%m-%d")
    out_dirs = [
        f"models/nano_pixel_mom/showcase_{date_str}",
        f"examples/{date_str}_nano_pixel_mom"
    ]

    for d in out_dirs:
        os.makedirs(d, exist_ok=True)

    engine = NanoPixelMomEngine()

    characters = [
        ("char_01_cyber_hero", "Cyber Hero"),
        ("char_02_mystic_wizard", "Mystic Wizard"),
        ("char_03_shadow_assassin", "Shadow Assassin"),
        ("char_04_retro_robot", "Retro Robot"),
        ("char_05_celestial_dragon", "Celestial Dragon")
    ]

    print(f"Generating showcase assets for date {date_str}...")

    for prefix, name in characters:
        # Single PNG
        single = engine.generate_sprite(prompt=name, seed=abs(hash(name)) % 10000)

        # 4-frame Sprite Sheet
        frames = [engine.generate_sprite(prompt=f"{name} frame {i}", seed=(abs(hash(name)) + i * 13) % 10000) for i in range(4)]
        sheet = engine.create_sprite_sheet(frames)

        # Animated GIF
        frames_np = [np.array(f) for f in frames]

        for d in out_dirs:
            single_path = os.path.join(d, f"{prefix}_single.png")
            sheet_path = os.path.join(d, f"{prefix}_spritesheet.png")
            gif_path = os.path.join(d, f"{prefix}_animation.gif")

            single.save(single_path)
            sheet.save(sheet_path)
            imageio.mimsave(gif_path, frames_np, fps=4, loop=0)

        print(f" Generated assets for: {name}")

    print("All showcase assets generated successfully in both target directories!")

if __name__ == "__main__":
    main()
