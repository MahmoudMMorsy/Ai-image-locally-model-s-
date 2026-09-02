import os
import io
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio

from pixel_art_engine.engine import PixelSpriteEngine
from pixel_art_engine.animation import SpriteAnimationGenerator
from pixel_art_engine.palette import quantize_to_pixel_art, GAMEBOY_PALETTE, NES_PALETTE, SIGNATURE_PALETTE
from poster_generator_256.poster_engine import BilingualPosterEngine

def main():
    out_dir = "examples/2026-09-02_pixel_art_showcase"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Generating daily showcase into: {out_dir}")

    engine = PixelSpriteEngine(device="cpu")
    animator = SpriteAnimationGenerator(engine, device="cpu")
    poster_engine = BilingualPosterEngine()

    characters = [
        ("char_01_gameboy_knight", "pixel knight warrior", "run", GAMEBOY_PALETTE, "GAMEBOY"),
        ("char_02_nes_wizard", "pixel wizard mage", "idle", NES_PALETTE, "NES"),
        ("char_03_nes_orc", "retro orc monster", "attack", NES_PALETTE, "NES"),
        ("char_04_gameboy_robot", "cyberpunk robot mech", "run", GAMEBOY_PALETTE, "GAMEBOY"),
        ("char_05_nes_paladin", "holy paladin hero", "walk", NES_PALETTE, "NES"),
    ]

    for filename, prompt, action, palette, style_name in characters:
        base = engine.generate_sprite(prompt=prompt, seed=42)

        base_quant = quantize_to_pixel_art(base, size=(64, 64), palette=palette)
        single_path = os.path.join(out_dir, f"{filename}_single.png")
        base_quant.save(single_path)

        # Generate Animation Frames & Sheet
        frames, sheet, gif_bytes = animator.generate_animation(base_quant, action=action, num_frames=4)

        # Apply strict palette quantization to frames & sheet
        quant_frames = [quantize_to_pixel_art(f, size=(64, 64), palette=palette) for f in frames]
        quant_sheet = quantize_to_pixel_art(sheet, size=(256, 64), palette=palette)

        sheet_path = os.path.join(out_dir, f"{filename}_spritesheet.png")
        quant_sheet.save(sheet_path)

        gif_path = os.path.join(out_dir, f"{filename}_animation.gif")
        imageio.mimsave(gif_path, [np.array(f) for f in quant_frames], format="GIF", duration=0.15, loop=0)

        print(f"  Generated: {filename} (PNG, Sheet, GIF) [{style_name} style]")

    # Generate Bilingual Posters
    posters = [
        ("poster_01_gameboy_legend", "Game Boy Legend", "أسطورة الجيم بوي", GAMEBOY_PALETTE),
        ("poster_02_nes_cyberpunk", "NES Pixel Hero", "بطل البيكسل NES", NES_PALETTE),
    ]

    for filename, title_en, title_ar, palette in posters:
        poster_img = poster_engine.generate_poster(title_en=title_en, title_ar=title_ar, seed=100)
        poster_quant = quantize_to_pixel_art(poster_img, size=(256, 256), palette=palette)
        poster_path = os.path.join(out_dir, f"{filename}.png")
        poster_quant.save(poster_path)
        print(f"  Generated Poster: {filename}.png")

    print(f"\nDaily Showcase Generation Complete in `{out_dir}`!")

if __name__ == "__main__":
    main()
