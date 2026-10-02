import os
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

def reshape_arabic(text):
    reshaper = arabic_reshaper.ArabicReshaper({'delete_harakat': True})
    reshaped = reshaper.reshape(text)
    return get_display(reshaped)

def generate_ai_learning_poster():
    out_dir = "examples/learn_ai_showcase"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "learn_ai_poster.png")

    img = Image.new('RGB', (256, 256), (15, 20, 45))
    draw = ImageDraw.Draw(img)

    # Background gradient
    c1, c2 = (15, 20, 45), (10, 140, 210)
    for y in range(256):
        r = int(c1[0] + (c2[0] - c1[0]) * (y / 256.0))
        g = int(c1[1] + (c2[1] - c1[1]) * (y / 256.0))
        b = int(c1[2] + (c2[2] - c1[2]) * (y / 256.0))
        draw.line([(0, y), (256, y)], fill=(r, g, b))

    # Frames and accents
    accent = (0, 240, 255)
    draw.rectangle([10, 10, 245, 245], outline=accent, width=2)
    draw.rectangle([14, 14, 241, 241], outline=(255, 255, 255, 80), width=1)

    # Center Brain/AI Symbol
    cx, cy = 128, 110
    draw.ellipse([cx - 40, cy - 40, cx + 40, cy + 40], outline=accent, width=3)
    draw.polygon([(cx, cy - 25), (cx + 25, cy + 20), (cx - 25, cy + 20)], outline=(255, 255, 255), fill=(20, 40, 80))

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

    title = reshape_arabic("تعلم الذكاء الاصطناعي")
    subtitle = reshape_arabic("طور النماذج وابنِ المستقبل")
    tagline = reshape_arabic("لا تضيع وقتك - ابدأ اليوم")

    if font_path:
        font_title = ImageFont.truetype(font_path, 16)
        font_sub = ImageFont.truetype(font_path, 11)
        draw.text((128, 35), title, fill=(255, 255, 255), font=font_title, anchor="mm")
        draw.text((128, 185), subtitle, fill=accent, font=font_sub, anchor="mm")
        draw.text((128, 210), tagline, fill=(220, 250, 220), font=font_sub, anchor="mm")
    else:
        draw.text((128, 35), title, fill=(255, 255, 255), anchor="mm")
        draw.text((128, 185), subtitle, fill=accent, anchor="mm")

    img.save(out_path)
    print(f"Generated AI learning poster: {out_path}")

def print_arabic_ai_learning_guide():
    guide = """
===================================================================
🚀 دليل عملي: كيف تبدأ تعلم الذكاء الاصطناعي وتطوير النماذج بنفسك
===================================================================

1️⃣ **تعلم الأساسيات المفاهيمية:**
   - البرمجة بـ Python وهياكل البيانات.
   - الرياضيات الأساسية (الجبر الخطي، التفاضل والتكامل، والاحتمالات).

2️⃣ **فهم أطر عمل التعلم العميق (Deep Learning Frameworks):**
   - PyTorch أو TensorFlow لبناء الشبكات العصبية (Neural Networks).
   - تصميم الشبكات مثل U-Net للصور أو Transformers للنصوص.

3️⃣ **تطوير ونشر النماذج (Model Fine-Tuning & Export):**
   - تدريب النماذج على بياناتك الخاصة (مثل خوارزميات البكسل آرت Diffusion أو توليد الملصقات).
   - تصدير النماذج بصيغة ONNX للتشغيل على الهواتف والأجهزة الذكية بكفاءة وInference سريعة.

4️⃣ **التطبيق العملي والمشاريع:**
   - بناء محركات توليد الصور والـ Sprite Sheets.
   - دمج النماذج مع تطبيقات الويب والهواتف (مثل Android NDK/Kotlin ONNX Runtime).

💡 **النصيحة:** استغل وقتك في بناء مشاريع حقيقية وتطوير نماذجك الخاصة بدلاً من الاكتفاء بالتصفح العابر!
===================================================================
"""
    print(guide)

if __name__ == "__main__":
    generate_ai_learning_poster()
    print_arabic_ai_learning_guide()
