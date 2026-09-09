import torch
import numpy as np
from PIL import Image
from pixel_art_engine.clip_text import SimpleCLIPTextEncoder
from pixel_art_engine.model import PixelSpriteEncoder, PixelSpriteGenerator
from pixel_art_engine.procedural import generate_arcade_sprite
from pixel_art_engine.palette import quantize_to_pixel_art, SIGNATURE_PALETTE

class PixelSpriteEngine:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.text_encoder = SimpleCLIPTextEncoder().to(self.device)
        self.encoder = PixelSpriteEncoder(latent_dim=64).to(self.device)
        self.generator = PixelSpriteGenerator(latent_dim=64, condition_dim=32).to(self.device)

    def generate_sprite(self, prompt="knight", seed=42):
        archetype = "knight"
        p_lower = str(prompt).lower()
        if "wizard" in p_lower or "mage" in p_lower:
            archetype = "wizard"
        elif "monster" in p_lower or "orc" in p_lower or "dragon" in p_lower:
            archetype = "monster"
        elif "robot" in p_lower or "mech" in p_lower:
            archetype = "robot"

        return generate_arcade_sprite(archetype=archetype, color_theme="red", pose="idle", frame=0)

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
        return quantize_to_pixel_art(image)
