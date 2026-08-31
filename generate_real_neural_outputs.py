import os
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont
from models.real_diffusion_onnx.real_onnx_generator import RealNeuralDiffusionGenerator

def main():
    out_dir = "examples/2026-08-29_real_neural_diffusion_posters"
    os.makedirs(out_dir, exist_ok=True)
    generator = RealNeuralDiffusionGenerator()

    prompts = [
        ("poster_01_future_city.png", "مدينة المستقبل الرقمية", "Cyberpunk Future City"),
        ("poster_02_brave_knight.png", "أسطورة الفارس الشجاع", "Brave Knight Legend"),
        ("poster_03_space_odyssey.png", "رحلة استكشاف الفضاء", "Space Exploration Odyssey"),
        ("poster_04_magic_realm.png", "مملكة السحر والغموض", "Magic Mystic Realm"),
        ("poster_05_arcade_fest.png", "مهرجان الألعاب الكلاسيكية", "Retro Arcade Festival")
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

    print("Generating 100% Neural Latent Diffusion Posters...")
    for idx, (filename, text_ar, text_en) in enumerate(prompts, start=1):
        # 1. Generate base image from Neural Latent UNet + VAE Decoder
        neural_img = generator.generate_image(prompt=text_en, steps=10, seed=idx * 42)

        # 2. Draw Arabic title over neural diffusion background
        draw = ImageDraw.Draw(neural_img)
        reshaped_ar = get_display(reshaper.reshape(text_ar))
        reshaped_date = get_display(reshaper.reshape("2026-08-29 ❤️"))

        if font_path:
            font_title = ImageFont.truetype(font_path, 16)
            font_sub = ImageFont.truetype(font_path, 12)
            draw.text((128, 25), reshaped_ar, fill=(255, 255, 255), font=font_title, anchor="mm")
            draw.text((128, 230), reshaped_date, fill=(255, 210, 210), font=font_sub, anchor="mm")
        else:
            draw.text((128, 25), reshaped_ar, fill=(255, 255, 255), anchor="mm")
            draw.text((128, 230), reshaped_date, fill=(255, 210, 210), anchor="mm")

        filepath = os.path.join(out_dir, filename)
        neural_img.save(filepath)
        print(f"[{idx}/5] Saved Neural Poster: {filepath}")

    print("All Neural Latent Diffusion Posters saved successfully!")

if __name__ == "__main__":
    main()
