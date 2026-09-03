import os
import glob
import numpy as np
from PIL import Image
import imageio

from pixel_art_engine.converter import ImageToSpriteConverter
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

    print("Generating Nano Pixel MOM Showcase Assets directly from REAL dataset training characters...")

    converter = ImageToSpriteConverter(device="cpu")
    animator = SpriteAnimationGenerator(device="cpu")
    poster_engine = BilingualPosterEngine()

    dataset_dir = "dataset_training_images/images_2534_only/sprites"
    all_files = sorted(glob.glob(os.path.join(dataset_dir, "*_c00.png")))
    if not all_files:
        all_files = sorted(glob.glob(os.path.join(dataset_dir, "*.png")))

    print(f"Found {len(all_files)} dataset character files.")

    # Select 5 actual training images from dataset
    selected_files = all_files[:5]
    character_configs = [
        ("mom_char_01_gameboy_knight", selected_files[0], "run", GAMEBOY_PALETTE, "GAMEBOY"),
        ("mom_char_02_nes_wizard", selected_files[1 % len(selected_files)], "idle", NES_PALETTE, "NES"),
        ("mom_char_03_nes_orc", selected_files[2 % len(selected_files)], "attack", NES_PALETTE, "NES"),
        ("mom_char_04_gameboy_robot", selected_files[3 % len(selected_files)], "run", GAMEBOY_PALETTE, "GAMEBOY"),
        ("mom_char_05_nes_paladin", selected_files[4 % len(selected_files)], "walk", NES_PALETTE, "NES"),
    ]

    for filename, file_path, action, palette, style_name in character_configs:
        print(f"Reconstructing real dataset character: {file_path} -> {filename}")
        raw_dataset_img = Image.open(file_path).convert("RGBA")

        # Convert and align exact character shape to 64x64 pixel grid
        sprite_base = converter.convert_image_to_sprite(raw_dataset_img, strength=0.4)
        base_quant = quantize_to_pixel_art(sprite_base, size=(64, 64), palette=palette)

        # Generate Animation Frames & Sheet from real character
        frames, sheet, gif_bytes = animator.generate_animation(base_quant, action=action, num_frames=4)
        quant_frames = [quantize_to_pixel_art(f, size=(64, 64), palette=palette) for f in frames]
        quant_sheet = quantize_to_pixel_art(sheet, size=(256, 64), palette=palette)

        for d in target_dirs:
            base_quant.save(os.path.join(d, f"{filename}_single.png"))
            quant_sheet.save(os.path.join(d, f"{filename}_spritesheet.png"))
            imageio.mimsave(os.path.join(d, f"{filename}_animation.gif"), [np.array(f) for f in quant_frames], format="GIF", duration=0.15, loop=0)

        print(f"  Generated {filename} [{style_name}] directly from real training image.")

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

    print("\nNano Pixel MOM Real Dataset Showcase Generation Complete!")

if __name__ == "__main__":
    main()
