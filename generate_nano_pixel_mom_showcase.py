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
    target_dir = "examples/2026-09-02_unique_dataset_showcase"
    os.makedirs(target_dir, exist_ok=True)

    print(f"Generating Unique Nano Pixel MOM Showcase Assets into `{target_dir}`...")

    converter = ImageToSpriteConverter(device="cpu")
    animator = SpriteAnimationGenerator(device="cpu")
    poster_engine = BilingualPosterEngine()

    dataset_dir = "dataset_training_images/images_2534_only/sprites"
    all_files = sorted(glob.glob(os.path.join(dataset_dir, "*_c00.png")))
    if not all_files:
        all_files = sorted(glob.glob(os.path.join(dataset_dir, "*.png")))

    print(f"Found {len(all_files)} dataset character files.")

    # Select 5 unique dataset files spread across the dataset
    step_size = max(1, len(all_files) // 5)
    selected_files = [all_files[i * step_size] for i in range(min(5, len(all_files)))]

    character_configs = [
        ("char_01_arabic_warrior", selected_files[0], "run", NES_PALETTE, "NES"),
        ("char_02_desert_nomad", selected_files[1 % len(selected_files)], "walk", GAMEBOY_PALETTE, "GAMEBOY"),
        ("char_03_cyber_bedouin", selected_files[2 % len(selected_files)], "idle", NES_PALETTE, "NES"),
        ("char_04_ancient_pharaoh", selected_files[3 % len(selected_files)], "attack", GAMEBOY_PALETTE, "GAMEBOY"),
        ("char_05_mystic_vizier", selected_files[4 % len(selected_files)], "idle", NES_PALETTE, "NES"),
    ]

    for filename, file_path, action, palette, style_name in character_configs:
        print(f"Reconstructing dataset character: {file_path} -> {filename}")
        raw_dataset_img = Image.open(file_path).convert("RGBA")

        # Convert and align character shape to 64x64 pixel grid
        sprite_base = converter.convert_image_to_sprite(raw_dataset_img, strength=0.4)
        base_quant = quantize_to_pixel_art(sprite_base, size=(64, 64), palette=palette)

        # Generate Animation Frames & Sheet from dataset character
        frames, sheet, gif_bytes = animator.generate_animation(base_quant, action=action, num_frames=4)
        quant_frames = [quantize_to_pixel_art(f, size=(64, 64), palette=palette) for f in frames]
        quant_sheet = quantize_to_pixel_art(sheet, size=(256, 64), palette=palette)

        base_quant.save(os.path.join(target_dir, f"{filename}_single.png"))
        quant_sheet.save(os.path.join(target_dir, f"{filename}_spritesheet.png"))
        imageio.mimsave(os.path.join(target_dir, f"{filename}_animation.gif"), [np.array(f) for f in quant_frames], format="GIF", duration=0.15, loop=0)

        print(f"  Generated {filename} [{style_name}] directly from dataset character.")

    posters = [
        ("poster_01_arabic_hero", "Arabic Pixel Legend", "بطل البيكسل العربي", NES_PALETTE),
        ("poster_02_desert_quest", "Desert Pixel Quest", "مغامرة الصحراء البيكسل", GAMEBOY_PALETTE),
    ]

    for filename, title_en, title_ar, palette in posters:
        poster_img = poster_engine.generate_poster(title_en=title_en, title_ar=title_ar, seed=500)
        poster_quant = quantize_to_pixel_art(poster_img, size=(256, 256), palette=palette)
        poster_quant.save(os.path.join(target_dir, f"{filename}.png"))
        print(f"  Generated Poster {filename}.png in `{target_dir}`")

    print(f"\nUnique Nano Pixel MOM Showcase Generation Complete in `{target_dir}`!")

if __name__ == "__main__":
    main()
