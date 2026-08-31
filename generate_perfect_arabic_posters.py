import os
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

def reshape_arabic_correct(text):
    # Standard reshaper without duplicate BiDi
    reshaper = arabic_reshaper.ArabicReshaper({'delete_harakat': True})
    reshaped = reshaper.reshape(text)
    return get_display(reshaped)

def generate_posters():
    out_dir = "examples/2026-08-29_arabic_posters_256_heart"
    os.makedirs(out_dir, exist_ok=True)

    posters = [
        ("poster_01_future_city.png", "مدينة المستقبل الرقمية", "عالم الذكاء الاصطناعي ❤️", (20, 10, 45), (240, 20, 140), (0, 240, 255)),
        ("poster_02_brave_knight.png", "أسطورة الفارس الشجاع", "معركة مملكة النور ❤️", (30, 20, 10), (220, 160, 40), (255, 215, 0)),
        ("poster_03_space_voyage.png", "رحلة استكشاف الفضاء", "مكوك المجرات البعيدة ❤️", (10, 30, 50), (30, 180, 220), (100, 255, 255)),
        ("poster_04_magic_world.png", "مملكة السحر والغموض", "أسرار الغابة المسحورة ❤️", (25, 10, 35), (180, 40, 200), (230, 120, 255)),
        ("poster_05_arcade_fest.png", "مهرجان الألعاب الكلاسيكية", "بطولة التحدي الأكبر ❤️", (40, 15, 15), (230, 70, 30), (255, 200, 50)),
        ("poster_06_ocean_depths.png", "أعماق المحيط الغامض", "مغامرة المرجان الأزرق ❤️", (10, 25, 40), (20, 150, 180), (80, 240, 200)),
        ("poster_07_golden_caravan.png", "قافلة الصحراء الذهبية", "طريق الحرير والأساطير ❤️", (45, 30, 10), (210, 140, 30), (255, 220, 100)),
        ("poster_08_neon_samurai.png", "محارب الساموراي النيون", "سيف العدالة والتنين ❤️", (15, 15, 30), (220, 30, 90), (0, 255, 200)),
        ("poster_09_solar_energy.png", "طاقة الشمس المتجددة", "مستقبل الطاقة النظيفة ❤️", (10, 40, 30), (40, 190, 110), (150, 255, 150)),
        ("poster_10_black_dragon.png", "قمة التنين الأسود", "عرش الأسطورة والخير ❤️", (30, 10, 20), (190, 20, 60), (255, 180, 180))
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

    print("Generating 10 correctly ordered Arabic posters with date 2026-08-29 and ❤️...")
    for idx, (filename, title_ar, sub_ar, c1, c2, accent) in enumerate(posters, start=1):
        img = Image.new('RGB', (256, 256), c1)
        draw = ImageDraw.Draw(img)

        # Gradient
        for y in range(256):
            r = int(c1[0] + (c2[0] - c1[0]) * (y / 256.0))
            g = int(c1[1] + (c2[1] - c1[1]) * (y / 256.0))
            b = int(c1[2] + (c2[2] - c1[2]) * (y / 256.0))
            draw.line([(0, y), (256, y)], fill=(r, g, b))

        # Frames
        draw.rectangle([10, 10, 245, 245], outline=accent, width=2)
        draw.rectangle([14, 14, 241, 241], outline=(255, 255, 255, 80), width=1)
        draw.ellipse([78, 68, 178, 168], outline=accent, width=3)
        draw.polygon([(128, 78), (163, 148), (93, 148)], outline=(255, 255, 255), fill=c1)

        # Reshaped Arabic
        title_formatted = reshape_arabic_correct(title_ar)
        sub_formatted = reshape_arabic_correct(sub_ar)
        date_formatted = reshape_arabic_correct("— 2026-08-29 ❤️ —")

        if font_path:
            font_title = ImageFont.truetype(font_path, 16)
            font_sub = ImageFont.truetype(font_path, 12)
            draw.text((128, 32), title_formatted, fill=(255, 255, 255), font=font_title, anchor="mm")
            draw.text((128, 205), sub_formatted, fill=accent, font=font_sub, anchor="mm")
            draw.text((128, 228), date_formatted, fill=(255, 220, 220), font=font_sub, anchor="mm")
        else:
            draw.text((128, 32), title_formatted, fill=(255, 255, 255), anchor="mm")
            draw.text((128, 205), sub_formatted, fill=accent, anchor="mm")

        filepath = os.path.join(out_dir, filename)
        img.save(filepath)
        print(f"[{idx}/10] Saved Correct Arabic Poster: {filepath}")

    print("All 10 Arabic posters updated and saved successfully!")

if __name__ == "__main__":
    generate_posters()
