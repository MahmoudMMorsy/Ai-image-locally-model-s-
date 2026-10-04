import torch
import numpy as np
from PIL import Image
from pixel_art_engine.clip_text import SimpleCLIPTextEncoder
from pixel_art_engine.model import PixelSpriteEncoder, PixelSpriteGenerator
from pixel_art_engine.procedural import generate_arcade_sprite
from pixel_art_engine.palette import (
    quantize_to_pixel_art,
    SIGNATURE_PALETTE,
    GAMEBOY_PALETTE,
    NES_PALETTE
)

class PixelSpriteEngine:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.encoder = PixelSpriteEncoder(latent_dim=64).to(self.device)
        self.generator = PixelSpriteGenerator(latent_dim=64, condition_dim=32).to(self.device)
        self.clip_encoder = SimpleCLIPTextEncoder()

        self.encoder.eval()
        self.generator.eval()

    def generate_sprite(self, prompt="knight", seed=42, style="signature"):
        torch.manual_seed(seed)
        np.random.seed(seed)

        archetype = "knight"
        p_lower = str(prompt).lower()
        if "wizard" in p_lower or "mage" in p_lower:
            archetype = "wizard"
        elif "monster" in p_lower or "orc" in p_lower or "dragon" in p_lower:
            archetype = "monster"
        elif "robot" in p_lower or "mech" in p_lower:
            archetype = "robot"

        color_theme = "blue"
        if "red" in p_lower:
            color_theme = "red"
        elif "green" in p_lower:
            color_theme = "green"
        elif "gold" in p_lower or "yellow" in p_lower:
            color_theme = "gold"

        raw_sprite = generate_arcade_sprite(archetype=archetype, color_theme=color_theme, pose="idle", frame=0)

        # Select target palette
        if style.lower() == "gameboy" or "gameboy" in p_lower or "gb" in p_lower:
            target_palette = GAMEBOY_PALETTE
        elif style.lower() == "nes" or "nes" in p_lower:
            target_palette = NES_PALETTE
        else:
            target_palette = SIGNATURE_PALETTE

        return quantize_to_pixel_art(raw_sprite, size=(64, 64), palette=target_palette)

    def create_sprite_sheet(self, frames):
        if not frames:
            return Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        width, height = frames[0].size
        sheet = Image.new("RGBA", (width * len(frames), height), (0, 0, 0, 0))
        for i, frame in enumerate(frames):
            sheet.paste(frame, (i * width, 0))
        return sheet

    def img2sprite(self, image, style="signature"):
        if isinstance(image, str):
            image = Image.open(image).convert("RGBA")

        palette_map = {
            "gameboy": GAMEBOY_PALETTE,
            "nes": NES_PALETTE,
            "signature": SIGNATURE_PALETTE
        }
        target_pal = palette_map.get(style.lower(), SIGNATURE_PALETTE)
        return quantize_to_pixel_art(image, size=(64, 64), palette=target_pal)
