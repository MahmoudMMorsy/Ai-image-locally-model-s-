import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

from pixel_art_engine.engine import PixelSpriteEngine
from pixel_art_engine.animation import SpriteAnimationGenerator
from pixel_art_engine.palette import SIGNATURE_PALETTE, GAMEBOY_PALETTE, NES_PALETTE, quantize_to_pixel_art

# Import model training entry points
from models.real_diffusion_onnx.train_and_export_real_onnx import main as train_real_diffusion
from models.nano_pixel_3A_xl.scripts.train_nano_pixel_3A_xl import main as train_nano_3a_xl
from models.nano_pixel_XL0_2.scripts.train_nano_pixel import train as train_nano_xl02
from models.nano_pixel_art_v1.scripts.train_nano_pixel import train_pixel_space_diffusion as train_nano_v1
from poster_generator_256.arabic_dataset_trainer import train_arabic_poster_model

def run_daily_training():
    print("=" * 70)
    print("Running Daily Training Pipeline for All Repository Models")
    print(f"Execution Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 1. Dataset Verification
    dataset_dir = "dataset_training_images/clean"
    if not os.path.exists(dataset_dir) or len(os.listdir(dataset_dir)) == 0:
        print("[Dataset] Extracting clean dataset zip...")
        os.system("mkdir -p dataset_training_images/clean && unzip -q -o dataset_clean.zip -d dataset_training_images/tmp && mv dataset_training_images/tmp/dataset_clean/* dataset_training_images/clean/ && rm -rf dataset_training_images/tmp")

    files = [f for f in os.listdir(dataset_dir) if f.endswith(".png")]
    print(f"[Dataset] Verified {len(files)} clean sprite training images.")

    # 2. Train All Models on Real Dataset Images
    print("\n--- [Step 1/5] Training Real Latent Diffusion UNet & VAE Decoder ---")
    train_real_diffusion()

    print("\n--- [Step 2/5] Training NanoPixel 3A XL Model ---")
    train_nano_3a_xl()

    print("\n--- [Step 3/5] Training NanoPixel XL0.2 Model ---")
    train_nano_xl02()

    print("\n--- [Step 4/5] Training NanoPixel-v1 Model ---")
    train_nano_v1(max_epochs=10)

    print("\n--- [Step 5/5] Training Arabic Poster Text & Vocabulary Model ---")
    train_arabic_poster_model()

    # 3. Generate Daily Showcase Outputs (Game Boy & NES Palettes + Bilingual Posters)
    print("\n--- Generating Daily Retro Pixel Art & Poster Showcase ---")
    showcase_dir = f"examples/daily_pixel_art_showcase_{time.strftime('%Y-%m-%d')}"
    os.makedirs(showcase_dir, exist_ok=True)

    engine = PixelSpriteEngine(device="cpu")
    animator = SpriteAnimationGenerator(engine, device="cpu")

    # Character Generation with Retro Game Boy & NES Quantization
    char_types = [
        ("paladin_knight", "pixel knight warrior in heavy plate armor"),
        ("cyber_wizard", "cyberpunk sorcerer casting glowing spell"),
        ("shadow_ninja", "shadow assassin ninja with katana blade"),
        ("mech_soldier", "futuristic retro robot mecha unit")
    ]

    for idx, (name, prompt) in enumerate(char_types, start=1):
        sprite = engine.generate_sprite(prompt=prompt, seed=100 + idx)

        # Standard Signature Palette PNG
        sprite.save(os.path.join(showcase_dir, f"char_0{idx}_{name}_signature.png"))

        # Game Boy 4-Green Palette Quantized PNG
        gb_sprite = quantize_to_pixel_art(sprite, size=(64, 64), palette=GAMEBOY_PALETTE)
        gb_sprite.save(os.path.join(showcase_dir, f"char_0{idx}_{name}_gameboy.png"))

        # NES 16-Color Palette Quantized PNG
        nes_sprite = quantize_to_pixel_art(sprite, size=(64, 64), palette=NES_PALETTE)
        nes_sprite.save(os.path.join(showcase_dir, f"char_0{idx}_{name}_nes.png"))

        # 4-frame action sprite sheet & animated GIF
        frames, sheet, gif_bytes = animator.generate_animation(sprite, action="run", num_frames=4)
        sheet.save(os.path.join(showcase_dir, f"char_0{idx}_{name}_action_sheet.png"))
        with open(os.path.join(showcase_dir, f"char_0{idx}_{name}_anim.gif"), "wb") as f:
            f.write(gif_bytes)

    # Bilingual Poster Generation
    font_path = None
    possible_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
    ]
    for f in possible_fonts:
        if os.path.exists(f):
            font_path = f
            break

    poster = Image.new("RGB", (256, 256), (15, 25, 45))
    draw = ImageDraw.Draw(poster)
    for y in range(256):
        r = int(15 + (45 - 15) * (y / 256.0))
        g = int(25 + (120 - 25) * (y / 256.0))
        b = int(45 + (180 - 45) * (y / 256.0))
        draw.line([(0, y), (256, y)], fill=(r, g, b))

    draw.rectangle([10, 10, 245, 245], outline=(0, 240, 255), width=2)
    draw.ellipse([88, 70, 168, 150], outline=(255, 215, 0), width=3)

    ar_title = get_display(arabic_reshaper.reshape("أسطورة البكسل الرقمية"))
    ar_sub = get_display(arabic_reshaper.reshape("توليد سريع علي CPU بدون GPU"))

    if font_path:
        f_title = ImageFont.truetype(font_path, 16)
        f_sub = ImageFont.truetype(font_path, 11)
        draw.text((128, 35), ar_title, fill=(255, 255, 255), font=f_title, anchor="mm")
        draw.text((128, 205), ar_sub, fill=(0, 240, 255), font=f_sub, anchor="mm")
        draw.text((128, 225), f"Digital Pixel Legend — {time.strftime('%Y-%m-%d')}", fill=(200, 200, 200), font=f_sub, anchor="mm")
    else:
        draw.text((128, 35), ar_title, fill=(255, 255, 255), anchor="mm")
        draw.text((128, 205), ar_sub, fill=(0, 240, 255), anchor="mm")

    poster_path = os.path.join(showcase_dir, "bilingual_showcase_poster.png")
    poster.save(poster_path)

    # 4. Log Progress
    log_file = "TRAINING_LOG.md"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n## Daily Training Execution — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Clean Training Images Processed: {len(files)}\n")
        f.write("- Models Fine-tuned: `real_diffusion_onnx`, `nano_pixel_3A_xl`, `nano_pixel_XL0_2`, `nano_pixel_art_v1`, `poster_generator_256`\n")
        f.write("- Exported CPU ONNX Models (< 50MB): `real_latent_unet_256.onnx`, `real_vae_decoder_256.onnx`, `nanopixel_3A_xl.onnx`, `nanopixel_XL0_2.onnx`, `nanopixel_v1.onnx`, `poster_generator_256.onnx`\n")
        f.write(f"- Generated Retro Palette Showcases (Game Boy & NES) & Bilingual Poster in: `{showcase_dir}`\n")

    print(f"\nDaily Training & ONNX Pipeline Execution Completed Successfully!")
    print(f"Showcase outputs saved in: {showcase_dir}")

if __name__ == "__main__":
    run_daily_training()
