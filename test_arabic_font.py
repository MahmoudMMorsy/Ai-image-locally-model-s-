import os
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

def render_arabic_poster(title_ar, subtitle_ar, filename):
    out_dir = "examples/2026-08-29_arabic_posters_256"
    os.makedirs(out_dir, exist_ok=True)

    img = Image.new('RGB', (256, 256), (15, 20, 35))
    draw = ImageDraw.Draw(img)

    # Draw background aesthetics & border
    draw.rectangle([10, 10, 245, 245], outline=(0, 220, 255), width=2)
    draw.ellipse([78, 60, 178, 160], outline=(255, 180, 0), width=3)

    # Try finding an Arabic TrueType font on system, or create clean pixel font rendering
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

    reshaped_title = get_display(arabic_reshaper.reshape(title_ar))
    reshaped_sub = get_display(arabic_reshaper.reshape(subtitle_ar))

    if font_path:
        font_large = ImageFont.truetype(font_path, 18)
        font_small = ImageFont.truetype(font_path, 13)
        draw.text((128, 32), reshaped_title, fill=(255, 255, 255), font=font_large, anchor="mm")
        draw.text((128, 200), reshaped_sub, fill=(0, 220, 255), font=font_small, anchor="mm")
    else:
        draw.text((128, 32), reshaped_title, fill=(255, 255, 255), anchor="mm")
        draw.text((128, 200), reshaped_sub, fill=(0, 220, 255), anchor="mm")

    filepath = os.path.join(out_dir, filename)
    img.save(filepath)
    print("Saved:", filepath)

if __name__ == "__main__":
    render_arabic_poster("مدينة المستقبل الرقمية", "عالم الذكاء الاصطناعي", "poster_01_future_city.png")
