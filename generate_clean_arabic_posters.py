import os
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

def render_arabic_text(text):
    reshaper = arabic_reshaper.ArabicReshaper({'delete_harakat': True})
    return get_display(reshaper.reshape(text))

def main():
    out_dir = "examples/2026-08-29_arabic_posters_clean"
    os.makedirs(out_dir, exist_ok=True)

    # 10 clear, meaningful Arabic phrases
    posters = [
        ("poster_01_future.png", "المستقبل الرقمي", "عالم الذكاء الاصطناعي", (20, 15, 45), (220, 30, 140), (0, 230, 255)),
        ("poster_02_knight.png", "الفارس الشجاع", "أسطورة مملكة النور", (35, 20, 10), (210, 150, 30), (255, 215, 0)),
        ("poster_03_space.png", "استكشاف الفضاء", "رحلة إلى المجرات البعيدة", (10, 25, 50), (20, 160, 220), (100, 240, 255)),
        ("poster_04_magic.png", "مملكة السحر", "أسرار الغابة المسحورة", (25, 10, 35), (170, 30, 190), (220, 110, 255)),
        ("poster_05_arcade.png", "مهرجان الألعاب", "بطولة التحدي الأكبر", (40, 15, 15), (220, 60, 25), (255, 190, 40)),
        ("poster_06_ocean.png", "أعماق المحيط", "مغامرة المرجان الأزرق", (10, 25, 40), (20, 140, 170), (80, 230, 190)),
        ("poster_07_desert.png", "قافلة الصحراء", "طريق الحرير والأساطير", (45, 30, 10), (200, 130, 25), (255, 210, 90)),
        ("poster_08_samurai.png", "محارب الساموراي", "سيف العدالة والكرامة", (15, 15, 30), (210, 25, 80), (0, 245, 190)),
        ("poster_09_solar.png", "الطاقة النظيفة", "مستقبل البيئة الخضراء", (10, 40, 30), (35, 180, 100), (140, 250, 140)),
        ("poster_10_dragon.png", "قمة التنين", "عرش الأسطورة والبطولة", (30, 10, 20), (180, 20, 55), (255, 170, 170))
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

    print("Generating 10 clean Arabic poster samples into examples/2026-08-29_arabic_posters_clean...")
    for idx, (filename, title, subtitle, c1, c2, accent) in enumerate(posters, start=1):
        img = Image.new('RGB', (256, 256), c1)
        draw = ImageDraw.Draw(img)

        # Gradient background
        for y in range(256):
            r = int(c1[0] + (c2[0] - c1[0]) * (y / 256.0))
            g = int(c1[1] + (c2[1] - c1[1]) * (y / 256.0))
            b = int(c1[2] + (c2[2] - c1[2]) * (y / 256.0))
            draw.line([(0, y), (256, y)], fill=(r, g, b))

        # Geometric art frames
        draw.rectangle([12, 12, 243, 243], outline=accent, width=2)
        draw.rectangle([16, 16, 239, 239], outline=(255, 255, 255, 70), width=1)
        draw.ellipse([80, 70, 176, 166], outline=accent, width=3)
        draw.polygon([(128, 80), (160, 145), (96, 145)], outline=(255, 255, 255), fill=c1)

        title_render = render_arabic_text(title)
        sub_render = render_arabic_text(subtitle)
        date_render = render_arabic_text("2026-08-29 ❤️")

        if font_path:
            font_title = ImageFont.truetype(font_path, 17)
            font_sub = ImageFont.truetype(font_path, 12)
            draw.text((128, 35), title_render, fill=(255, 255, 255), font=font_title, anchor="mm")
            draw.text((128, 200), sub_render, fill=accent, font=font_sub, anchor="mm")
            draw.text((128, 222), date_render, fill=(255, 220, 220), font=font_sub, anchor="mm")
        else:
            draw.text((128, 35), title_render, fill=(255, 255, 255), anchor="mm")
            draw.text((128, 200), sub_render, fill=accent, anchor="mm")

        filepath = os.path.join(out_dir, filename)
        img.save(filepath)
        print(f"[{idx}/10] Saved clean poster: {filepath}")

    print("\nAll 10 clean Arabic posters generated successfully!")

if __name__ == "__main__":
    main()
