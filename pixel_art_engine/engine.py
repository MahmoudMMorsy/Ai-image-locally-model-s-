"""
High-Level Interface for Pixel Art Sprite Generation
Coordinates neural generator, text condition mapping, and palette quantization.
"""
import torch
import numpy as np
from PIL import Image
import os
from .model import PixelSpriteGenerator, PixelSpriteEncoder
from .palette import quantize_to_pixel_art, SIGNATURE_PALETTE
from .diffusion import PixelDiffusion
from .procedural import generate_arcade_sprite


ARCHETYPES = {
    "knight": 0,
    "warrior": 1,
    "wizard": 2,
    "mage": 3,
    "rogue": 4,
    "archer": 5,
    "ninja": 6,
    "cyborg": 7,
    "robot": 8,
    "monster": 9,
    "orc": 10,
    "goblin": 11,
    "skeleton": 12,
    "hero": 13,
    "villain": 14,
    "alien": 15
}


class PixelSpriteEngine:
    """
    Main engine for generating 64x64 Pixel Art Sprites.
    Fast, lightweight, CPU-friendly.
    """
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.diffusion = PixelDiffusion(timesteps=30, device=self.device)

        # Load pre-trained diffusion weights
        diff_weights_path = os.path.join(os.path.dirname(__file__), "pixel_diffusion_weights.pt")
        if os.path.exists(diff_weights_path) and os.path.getsize(diff_weights_path) > 0:
            try:
                self.diffusion.model.load_state_dict(torch.load(diff_weights_path, map_location=self.device))
            except Exception:
                pass

        self.diffusion.model.eval()
        # Keep encoder/generator for img2img identity encoding
        self.encoder = PixelSpriteEncoder(latent_dim=64).to(self.device)
        self.generator = PixelSpriteGenerator(latent_dim=64, condition_dim=32).to(self.device)
        self.encoder.eval()
        self.generator.eval()

    def _text_to_condition(self, prompt: str) -> torch.Tensor:
        """Converts text prompt into a 32-dimensional condition vector."""
        cond = torch.zeros(1, 32, device=self.device)
        prompt_lower = prompt.lower()

        # Match archetype
        for key, idx in ARCHETYPES.items():
            if key in prompt_lower:
                cond[0, idx] = 1.0
                break

        # Match color hints
        colors = {
            "red": 16, "blue": 17, "green": 18, "yellow": 19,
            "purple": 20, "black": 21, "white": 22, "gold": 23,
            "dark": 24, "light": 25, "fire": 26, "ice": 27
        }
        for color, idx in colors.items():
            if color in prompt_lower:
                cond[0, idx] = 1.0

        return cond

    def generate_sprite(self, prompt: str = "pixel knight character", seed: int = None) -> Image.Image:
        """
        Generates a 64x64 pixel art sprite character from a prompt and seed.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        condition = self._text_to_condition(prompt)

        # Detect archetype and main color
        matched_arch = "knight"
        for key in ARCHETYPES:
            if key in prompt.lower():
                matched_arch = key
                break

        matched_color = "blue"
        for col in ["red", "green", "gold", "yellow", "purple", "dark", "blue"]:
            if col in prompt.lower():
                matched_color = col
                break

        # High-definition retro arcade character rendering
        arcade_img = generate_arcade_sprite(archetype=matched_arch, color_theme=matched_color, pose="idle")

        # Neural diffusion refinement
        with torch.no_grad():
            diff_sample = self.diffusion.sample(shape=(1, 4, 64, 64), condition=condition, seed=seed)
            diff_sample = torch.clamp((diff_sample + 1.0) * 127.5, 0, 255)
            diff_arr = diff_sample.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        # Blend arcade archetype with neural diffusion features
        arc_arr = np.array(arcade_img, dtype=np.float32)
        diff_arr_f = diff_arr.astype(np.float32)

        # Alpha mask for character shape
        mask = (arc_arr[:, :, 3:4] > 0).astype(np.float32)
        blended_arr = (arc_arr * 0.85 + diff_arr_f * 0.15 * mask).astype(np.uint8)
        blended_arr[:, :, 3] = arc_arr[:, :, 3] # preserve sharp outline transparency

        raw_img = Image.fromarray(blended_arr, mode="RGBA")
        return quantize_to_pixel_art(raw_img, size=(64, 64), palette=SIGNATURE_PALETTE)
