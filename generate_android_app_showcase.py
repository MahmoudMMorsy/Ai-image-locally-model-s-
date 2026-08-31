import os
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont
from models.real_diffusion_onnx.real_onnx_generator import RealNeuralDiffusionGenerator

def main():
    out_dir = "examples/2026-08-29_android_app_outputs"
    os.makedirs(out_dir, exist_ok=True)
    generator = RealNeuralDiffusionGenerator()

    prompts = [
        ("app_output_01_hero_knight.png", "فارس أساطير النور", "Paladin Hero Knight"),
        ("app_output_02_elemental_mage.png", "ساحر العناصر العظيم", "Elemental Archmage"),
        ("app_output_03_cyber_city.png", "بوستر مدينة المستقبل", "Cyberpunk Future City Poster"),
        ("app_output_04_space_marine.png", "جندي الفضاء المستقبلي", "Space Marine Sentinel"),
        ("app_output_05_desert_caravan.png", "بوستر قافلة الصحراء", "Golden Desert Caravan Poster")
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

    print("Generating 5 Android App ONNX-driven outputs...")
    for idx, (filename, text_ar, text_en) in enumerate(prompts, start=1):
        # Neural diffusion image generation
        img = generator.generate_image(prompt=text_en, steps=10, seed=idx * 77)

        draw = ImageDraw.Draw(img)
        reshaped_ar = get_display(reshaper.reshape(text_ar))
        reshaped_date = get_display(reshaper.reshape("Android App ONNX • 2026-08-29 ❤️"))

        if font_path:
            font_title = ImageFont.truetype(font_path, 15)
            font_sub = ImageFont.truetype(font_path, 11)
            draw.text((128, 25), reshaped_ar, fill=(255, 255, 255), font=font_title, anchor="mm")
            draw.text((128, 230), reshaped_date, fill=(0, 229, 255), font=font_sub, anchor="mm")
        else:
            draw.text((128, 25), reshaped_ar, fill=(255, 255, 255), anchor="mm")
            draw.text((128, 230), reshaped_date, fill=(0, 229, 255), anchor="mm")

        filepath = os.path.join(out_dir, filename)
        img.save(filepath)
        print(f"[{idx}/5] Saved Android App Output: {filepath}")

    print("All Android App showcase outputs generated successfully!")

if __name__ == "__main__":
    main()
