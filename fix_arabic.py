import os
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

def render_correct_arabic(text, font_path=None, font_size=18):
    # Properly reshape Arabic text and apply bidirectional algorithm so letters connect correctly from right to left
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    return bidi_text

def test_rendering():
    sample = "مدينة المستقبل الرقمية ❤️"
    rendered = render_correct_arabic("مدينة المستقبل الرقمية")
    print("Original Text:", sample)
    print("Reshaped & BiDi Text for PIL:", rendered)

if __name__ == "__main__":
    test_rendering()
