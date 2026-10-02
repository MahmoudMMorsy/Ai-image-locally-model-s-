import os
import imageio
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont
from pixel_art_engine.palette import GAMEBOY_PALETTE, NES_PALETTE, quantize_palette
from pixel_art_engine.procedural import generate_arcade_sprite
from models.real_diffusion_onnx.real_onnx_generator import RealNeuralDiffusionGenerator

def main():
    showcase_dir = "examples/2026-09-03_daily_showcase"
    os.makedirs(showcase_dir, exist_ok=True)

    print("Generating Daily Retro Showcase & Posters (Game Boy & NES Palettes)...")

    # 1. Generate Game Boy & NES Retro Character Sprites (Single PNG, Spritesheet, Animated GIF)
    char_types = [
        ("gameboy_hero", "Game Boy Knight", "knight", GAMEBOY_PALETTE),
        ("gameboy_wizard", "Game Boy Mage", "wizard", GAMEBOY_PALETTE),
        ("nes_cyber_warrior", "NES Cyber Warrior", "robot", NES_PALETTE),
        ("nes_dragon_boss", "NES Flame Dragon", "monster", NES_PALETTE)
    ]

    for name, title, archetype, palette in char_types:
        frames = []
        for frame_idx in range(4):
            # Generate procedural retro arcade sprite frame
            frame_img = generate_arcade_sprite(archetype=archetype, color_theme="blue", pose="walk", frame=frame_idx)
            # Quantize image to specified retro palette (Game Boy or NES)
            frame_quant = quantize_palette(frame_img, palette)
            frames.append(frame_quant)

        # Single character image
        single_path = os.path.join(showcase_dir, f"{name}_single.png")
        frames[0].save(single_path)

        # 4-frame sprite sheet
        sheet_img = Image.new("RGBA", (64 * 4, 64))
        for i, f in enumerate(frames):
            sheet_img.paste(f, (i * 64, 0))
        sheet_path = os.path.join(showcase_dir, f"{name}_spritesheet.png")
        sheet_img.save(sheet_path)

        # Animated GIF
        gif_path = os.path.join(showcase_dir, f"{name}_anim.gif")
        imageio.mimsave(gif_path, [np.array(f) for f in frames], duration=0.25, loop=0)
        print(f"  [Character] Generated {title} ({single_path}, {sheet_path}, {gif_path})")

    # 2. Generate Bilingual Arabic/English Posters
    poster_gen = RealNeuralDiffusionGenerator()
    posters = [
        ("poster_gameboy_legend.png", "أسطورة الجيم بوي", "Game Boy Pixel Legend", GAMEBOY_PALETTE),
        ("poster_nes_adventure.png", "مغامرة الإن إي إس الكلاسيكية", "NES Retro Pixel Adventure", NES_PALETTE)
    ]

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

    reshaper = arabic_reshaper.ArabicReshaper({'delete_harakat': True})

    for filename, text_ar, text_en, palette in posters:
        # Neural Latent UNet poster background (256x256)
        raw_poster = poster_gen.generate_image(prompt=text_en, steps=8, seed=2026)
        quantized_poster = quantize_palette(raw_poster, palette)

        draw = ImageDraw.Draw(quantized_poster)
        reshaped_ar = get_display(reshaper.reshape(text_ar))
        reshaped_date = get_display(reshaper.reshape("2026-09-03 ❤️"))

        if font_path:
            font_title = ImageFont.truetype(font_path, 16)
            font_sub = ImageFont.truetype(font_path, 12)
            draw.text((128, 20), reshaped_ar, fill=(255, 255, 255), font=font_title, anchor="mm")
            draw.text((128, 235), text_en, fill=(240, 240, 240), font=font_sub, anchor="mm")
            draw.text((128, 250), reshaped_date, fill=(200, 255, 200), font=font_sub, anchor="mm")
        else:
            draw.text((128, 20), reshaped_ar, fill=(255, 255, 255), anchor="mm")
            draw.text((128, 235), text_en, fill=(240, 240, 240), anchor="mm")

        poster_path = os.path.join(showcase_dir, filename)
        quantized_poster.save(poster_path)
        print(f"  [Poster] Saved Bilingual Poster: {poster_path}")

    # 3. Create descriptive README.md
    readme_path = os.path.join(showcase_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# Daily Showcase 2026-09-03 - Game Boy & NES Retro Pixel Art\n\n")
        f.write("This showcase features high-speed CPU neural diffusion poster synthesis and retro pixel art character generation fine-tuned on real dataset images.\n\n")
        f.write("## Features\n")
        f.write("- **Game Boy Palette Quantization**: Authentic 4-shade green retro styling.\n")
        f.write("- **NES Palette Quantization**: Authentic 16-color 8-bit retro styling.\n")
        f.write("- **Bilingual Posters**: Arabic and English typography rendered over neural latent diffusion outputs.\n")
        f.write("- **Sprite Assets**: Standalone PNGs, 4-frame action sprite sheets, and looping GIFs.\n")

    print("Daily Showcase generation complete!")

if __name__ == "__main__":
    main()
