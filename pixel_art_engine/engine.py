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
from .procedural import generate_procedural_sprite, pretrain_engine_weights


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
        self.generator = PixelSpriteGenerator(latent_dim=64, condition_dim=32).to(self.device)
        self.encoder = PixelSpriteEncoder(latent_dim=64).to(self.device)

        # Load pre-trained weights if available and valid (>0 bytes), or initialize quickly
        weights_path = os.path.join(os.path.dirname(__file__), "pretrained_weights.pt")
        if os.path.exists(weights_path) and os.path.getsize(weights_path) > 0:
            try:
                checkpoint = torch.load(weights_path, map_location=self.device)
                self.generator.load_state_dict(checkpoint["generator"])
                self.encoder.load_state_dict(checkpoint["encoder"])
            except Exception:
                pretrain_engine_weights(self.generator, self.encoder, device=self.device, epochs=5)
        else:
            pretrain_engine_weights(self.generator, self.encoder, device=self.device, epochs=5)

        self.generator.eval()
        self.encoder.eval()

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

        z = torch.randn(1, 64, device=self.device)
        condition = self._text_to_condition(prompt)

        # Detect archetype and main color
        matched_arch = "knight"
        for key in ARCHETYPES:
            if key in prompt.lower():
                matched_arch = key
                break

        matched_color = "blue"
        for col in ["red", "green", "gold", "purple", "dark", "blue"]:
            if col in prompt.lower():
                matched_color = col
                break

        # Blend procedural structural anchor with neural output
        proc_img = generate_procedural_sprite(matched_arch, matched_color, seed=seed if seed is not None else 42)
        proc_arr = np.array(proc_img, dtype=np.float32) / 255.0
        t_proc = torch.from_numpy(proc_arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            z_latent = self.encoder(t_proc)
            raw_tensor = self.generator(z_latent, condition)

        blended_tensor = 0.7 * t_proc + 0.3 * raw_tensor
        arr = (blended_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        raw_img = Image.fromarray(arr, mode="RGBA")

        return quantize_to_pixel_art(raw_img, size=(64, 64), palette=SIGNATURE_PALETTE)
