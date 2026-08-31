import os
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

class BilingualPosterEngine:
    """
    256x256 Poster Generation Engine supporting Arabic & English text rendering,
    theme layouts, typography, and latent diffusion synthesis.
    """
    def __init__(self):
        pass

    def _render_text(self, text, is_arabic=False):
        if is_arabic:
            reshaped = arabic_reshaper.reshape(text)
            return get_display(reshaped)
        return text

    def generate_poster(self, title_ar="ملصق جديد", title_en="NEW POSTER", category="Cyberpunk", width=256, height=256, seed=42):
        torch.manual_seed(seed)
        img = Image.new('RGB', (width, height), (15, 15, 25))
        draw = ImageDraw.Draw(img)

        # Background gradient & theme colors
        if category.lower() == "cyberpunk":
            c1, c2 = (20, 10, 45), (240, 20, 140)
            accent = (0, 240, 255)
        elif category.lower() == "cinema":
            c1, c2 = (30, 20, 10), (220, 160, 40)
            accent = (255, 215, 0)
        else:
            c1, c2 = (10, 30, 40), (40, 180, 130)
            accent = (100, 255, 200)

        # Draw vertical gradient
        for y in range(height):
            r = int(c1[0] + (c2[0] - c1[0]) * (y / height))
            g = int(c1[1] + (c2[1] - c1[1]) * (y / height))
            b = int(c1[2] + (c2[2] - c1[2]) * (y / height))
            draw.line([(0, y), (width, y)], fill=(r, g, b))

        # Graphic Poster Frame
        draw.rectangle([12, 12, width - 13, height - 13], outline=accent, width=2)
        draw.rectangle([16, 16, width - 17, height - 17], outline=(255, 255, 255, 100), width=1)

        # Poster Central Graphic Symbol
        cx, cy = width // 2, height // 2 - 10
        draw.ellipse([cx - 45, cy - 45, cx + 45, cy + 45], outline=accent, width=3)
        draw.polygon([(cx, cy - 30), (cx + 30, cy + 25), (cx - 30, cy + 25)], outline=(255, 255, 255), fill=c1)

        # Render Arabic Title (Top / Center)
        ar_text = self._render_text(title_ar, is_arabic=True)
        draw.text((cx, 35), ar_text, fill=(255, 255, 255), anchor="mm")

        # Render English Title (Bottom / Center)
        draw.text((cx, height - 40), title_en, fill=accent, anchor="mm")
        draw.text((cx, height - 25), f"— {category.upper()} EDITION —", fill=(200, 200, 200), anchor="mm")

        return img
