import os
import glob
import numpy as np
from PIL import Image
import imageio

from models.nano_pixel_mom.inference import NanoPixelMomGenerator
from pixel_art_engine.animation import SpriteAnimationGenerator
from pixel_art_engine.palette import quantize_to_pixel_art, GAMEBOY_PALETTE, NES_PALETTE
from poster_generator_256.poster_engine import BilingualPosterEngine

def main():
    target_dirs = [
        "models/nano_pixel_mom/showcase_2026-09-02",
        "examples/2026-09-02_nano_pixel_mom"
    ]
    for d in target_dirs:
        os.makedirs(d, exist_ok=True)

    print("Generating Nano Pixel MOM Showcase Assets via NanoPixelMomGenerator...")

    mom_gen = NanoPixelMomGenerator(device="cpu")
    animator = SpriteAnimationGenerator(device="cpu")
    poster_engine = BilingualPosterEngine()

    character_configs = [
        ("mom_char_01_gameboy_knight", "pixel knight warrior", "run", GAMEBOY_PALETTE, "GAMEBOY"),
        ("mom_char_02_nes_wizard", "pixel wizard mage", "idle", NES_PALETTE, "NES"),
        ("mom_char_03_nes_orc", "retro orc monster", "attack", NES_PALETTE, "NES"),
        ("mom_char_04_gameboy_robot", "cyberpunk robot mech", "run", GAMEBOY_PALETTE, "GAMEBOY"),
        ("mom_char_05_nes_paladin", "holy paladin hero", "walk", NES_PALETTE, "NES"),
    ]

    for filename, prompt, action, palette, style_name in character_configs:
        print(f"Generating neural sprite using NanoPixelMomGenerator for prompt '{prompt}' -> {filename}")

        # Directly generate character using NanoPixelMomGenerator neural network
        mom_neural_sprite = mom_gen.generate(prompt=prompt, steps=5, seed=200)
        base_quant = quantize_to_pixel_art(mom_neural_sprite, size=(64, 64), palette=palette)

        # Generate Animation Frames & Sheet
        frames, sheet, gif_bytes = animator.generate_animation(base_quant, action=action, num_frames=4)
        quant_frames = [quantize_to_pixel_art(f, size=(64, 64), palette=palette) for f in frames]
        quant_sheet = quantize_to_pixel_art(sheet, size=(256, 64), palette=palette)

        for d in target_dirs:
            base_quant.save(os.path.join(d, f"{filename}_single.png"))
            quant_sheet.save(os.path.join(d, f"{filename}_spritesheet.png"))
            imageio.mimsave(os.path.join(d, f"{filename}_animation.gif"), [np.array(f) for f in quant_frames], format="GIF", duration=0.15, loop=0)

        print(f"  Generated {filename} [{style_name}] via NanoPixelMomGenerator.")

    posters = [
        ("mom_poster_01_gameboy", "Nano Pixel MOM Game Boy", "نانو بيكسل مـام - جيم بوي", GAMEBOY_PALETTE),
        ("mom_poster_02_nes", "Nano Pixel MOM NES", "نانو بيكسل مـام - أن إي إس", NES_PALETTE),
    ]

    for filename, title_en, title_ar, palette in posters:
        poster_img = poster_engine.generate_poster(title_en=title_en, title_ar=title_ar, seed=300)
        poster_quant = quantize_to_pixel_art(poster_img, size=(256, 256), palette=palette)
        for d in target_dirs:
            poster_quant.save(os.path.join(d, f"{filename}.png"))
        print(f"  Generated Poster {filename}.png in both showcase folders.")

    print("\nNano Pixel MOM Showcase Generation Complete!")

if __name__ == "__main__":
    main()
