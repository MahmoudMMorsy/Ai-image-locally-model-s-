import torch
import numpy as np
from PIL import Image
from pixel_art_engine.clip_text import SimpleCLIPTextEncoder
from pixel_art_engine.procedural import generate_arcade_sprite
from pixel_art_engine.palette import quantize_to_signature_palette

class PixelSpriteEngine:
    def __init__(self):
        self.encoder = SimpleCLIPTextEncoder()

    def generate_sprite(self, prompt="knight", seed=42):
        return generate_arcade_sprite(prompt, seed)

    def create_sprite_sheet(self, frames):
        if not frames:
            return Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        width, height = frames[0].size
        sheet = Image.new("RGBA", (width * len(frames), height), (0, 0, 0, 0))
        for i, frame in enumerate(frames):
            sheet.paste(frame, (i * width, 0))
        return sheet

    def img2sprite(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGBA")
        image = image.resize((64, 64), Image.NEAREST)
        return quantize_to_signature_palette(image)
