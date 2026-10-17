import os
import sys
import torch
from PIL import Image

sys.path.append(".")

from models.real_diffusion_onnx.real_onnx_generator import RealNeuralDiffusionGenerator
from pixel_art_engine.engine import PixelSpriteEngine
from pixel_art_engine.palette import GAMEBOY_PALETTE, NES_PALETTE, SIGNATURE_PALETTE, quantize_to_pixel_art
from pixel_art_engine.animation import SpriteAnimationGenerator
from poster_generator_256.poster_engine import BilingualPosterEngine

def main():
    showcase_dir = "examples/2026-10-05_10_neural_characters_showcase"
    os.makedirs(showcase_dir, exist_ok=True)
    print(f"Generating 10 Neural Character Samples using trained neural models into '{showcase_dir}'...")

    # Initialize neural inference engines
    real_gen = RealNeuralDiffusionGenerator()
    px_engine = PixelSpriteEngine(device="cpu")
    animator = SpriteAnimationGenerator(px_engine, device="cpu")
    poster_engine = BilingualPosterEngine()

    characters = [
        ("01_egyptian_farmer", "فلاح مصري", "EGYPTIAN FARMER", "An Egyptian traditional farmer in countryside retro pixel art"),
        ("02_ancient_wizard", "ساحر الأسرار", "ANCIENT WIZARD", "Mystic ancient wizard casting magical spells"),
        ("03_pharaoh_guard", "حارس الفراعنة", "PHARAOH GUARD", "Royal Egyptian Pharaoh temple guard warrior"),
        ("04_cyber_ninja", "نينجا النيون", "CYBER NINJA", "Futuristic neon cyber ninja warrior with energy blade"),
        ("05_desert_bedouin", "بدوي الصحراء", "DESERT BEDOUIN", "Nomadic desert bedouin explorer in golden sands"),
        ("06_mech_warrior", "محارب الميكا", "MECH WARRIOR", "Heavy armored cyborg mech hero with plasma rifle"),
        ("07_space_pirate", "قرصان الفضاء", "SPACE PIRATE", "Intergalactic space pirate captain with laser pistol"),
        ("08_shadow_assassin", "قاطعة الظلال", "SHADOW ASSASSIN", "Stealth shadow assassin draped in dark cloak"),
        ("09_galactic_knight", "فارس المجرات", "GALACTIC KNIGHT", "Noble galactic knight with shining energy shield"),
        ("10_elemental_golem", "جولم العناصر", "ELEMENTAL GOLEM", "Ancient stone elemental golem glowing with magic power")
    ]

    generated_files = []

    for idx, (id_name, ar_title, en_title, prompt) in enumerate(characters, start=1):
        print(f"[{idx}/10] Generating Neural Character: {en_title} ({ar_title})...")

        # 1. Real Latent Diffusion 256x256 Neural Image Generation
        neural_img_256 = real_gen.generate_image(prompt=prompt, steps=10, seed=100 + idx)
        diffusion_path = os.path.join(showcase_dir, f"{id_name}_latent_diffusion_256.png")
        neural_img_256.save(diffusion_path)
        generated_files.append(diffusion_path)

        # 2. Neural Pixel Sprite Synthesis (64x64)
        base_sprite = px_engine.generate_sprite(prompt=prompt, seed=100 + idx)

        # Game Boy Quantization
        gb_img = quantize_to_pixel_art(base_sprite, size=(64, 64), palette=GAMEBOY_PALETTE)
        gb_path = os.path.join(showcase_dir, f"{id_name}_gameboy.png")
        gb_img.save(gb_path)
        generated_files.append(gb_path)

        # NES Quantization
        nes_img = quantize_to_pixel_art(base_sprite, size=(64, 64), palette=NES_PALETTE)
        nes_path = os.path.join(showcase_dir, f"{id_name}_nes.png")
        nes_img.save(nes_path)
        generated_files.append(nes_path)

        # Signature Palette
        sig_img = quantize_to_pixel_art(base_sprite, size=(64, 64), palette=SIGNATURE_PALETTE)
        sig_path = os.path.join(showcase_dir, f"{id_name}_signature.png")
        sig_img.save(sig_path)
        generated_files.append(sig_path)

        # 4-Frame Action Animation Sheet & Animated GIF
        frames, sheet, gif_bytes = animator.generate_animation(base_sprite, action="run", num_frames=4)
        sheet_path = os.path.join(showcase_dir, f"{id_name}_action_sheet.png")
        gif_path = os.path.join(showcase_dir, f"{id_name}_action_anim.gif")
        sheet.save(sheet_path)
        with open(gif_path, "wb") as f_gif:
            f_gif.write(gif_bytes)
        generated_files.extend([sheet_path, gif_path])

        # Bilingual Poster (256x256)
        poster_img = poster_engine.generate_poster(title_ar=ar_title, title_en=en_title, category="Showcase")
        poster_path = os.path.join(showcase_dir, f"{id_name}_poster.png")
        poster_img.save(poster_path)
        generated_files.append(poster_path)

    # Write Showcase README.md
    readme_path = os.path.join(showcase_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f_readme:
        f_readme.write("# 10 Genuine Neural Character Showcase (2026-10-05)\n\n")
        f_readme.write("This directory contains 10 distinct character archetypes generated using the trained neural generative models (`RealLatentUNet`, `RealLatentDecoder`, and `PixelSpriteEngine`):\n\n")
        f_readme.write("1. **فلاح مصري** (Egyptian Farmer)\n")
        f_readme.write("2. **ساحر الأسرار** (Ancient Wizard)\n")
        f_readme.write("3. **حارس الفراعنة** (Pharaoh Guard)\n")
        f_readme.write("4. **نينجا النيون** (Cyber Ninja)\n")
        f_readme.write("5. **بدوي الصحراء** (Desert Bedouin)\n")
        f_readme.write("6. **محارب الميكا** (Mech Warrior)\n")
        f_readme.write("7. **قرصان الفضاء** (Space Pirate)\n")
        f_readme.write("8. **قاطعة الظلال** (Shadow Assassin)\n")
        f_readme.write("9. **فارس المجرات** (Galactic Knight)\n")
        f_readme.write("10. **جولم العناصر** (Elemental Golem)\n\n")
        f_readme.write("## Each Archetype Includes:\n")
        f_readme.write("- `*_latent_diffusion_256.png`: Real Neural Latent Diffusion 256x256 output.\n")
        f_readme.write("- `*_gameboy.png`: Game Boy retro green-scale pixel art sprite.\n")
        f_readme.write("- `*_nes.png`: Authentic 16-color NES palette pixel art sprite.\n")
        f_readme.write("- `*_signature.png`: High-contrast 32-color arcade palette pixel art sprite.\n")
        f_readme.write("- `*_action_sheet.png`: 4-frame action animation sprite sheet.\n")
        f_readme.write("- `*_action_anim.gif`: 4-frame animated running GIF.\n")
        f_readme.write("- `*_poster.png`: 256x256 bilingual Arabic/English game poster.\n")

    print("\nSuccessfully generated all 10 neural character showcase samples!")

if __name__ == "__main__":
    main()
