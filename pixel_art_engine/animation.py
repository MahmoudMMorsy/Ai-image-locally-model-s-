"""
Neural Character Animation Generator
Uses latent space interpolation and neural decoder inference for smooth character action animations.
"""
import torch
import numpy as np
from PIL import Image
import imageio
from typing import List, Tuple
from .palette import quantize_to_pixel_art, SIGNATURE_PALETTE
from .model import PixelSpriteEncoder, PixelSpriteGenerator


ACTION_LATENT_OFFSETS = {
    "idle": [0.0, 0.02, 0.05, 0.02],
    "walk": [-0.1, 0.05, 0.1, -0.05],
    "run": [-0.2, 0.1, 0.2, -0.1],
    "attack": [-0.15, -0.3, 0.35, 0.1],
    "jump": [0.15, -0.4, -0.2, 0.05]
}


class SpriteAnimationGenerator:
    """
    Generates consistent sequential animation frames using neural latent perturbations.
    """
    def __init__(self, engine=None, device="cpu"):
        self.device = torch.device(device)
        if engine is not None:
            self.encoder = engine.encoder
            self.generator = engine.generator
        else:
            self.encoder = PixelSpriteEncoder(latent_dim=64).to(self.device)
            self.generator = PixelSpriteGenerator(latent_dim=64, condition_dim=32).to(self.device)

        self.encoder.eval()
        self.generator.eval()

    def generate_animation(
        self,
        base_character: Image.Image,
        action: str = "run",
        num_frames: int = 4
    ) -> Tuple[List[Image.Image], Image.Image, bytes]:
        """
        Generates a sequence of animation frames for the specified action.
        """
        if base_character.mode != "RGBA":
            base_character = base_character.convert("RGBA")

        base_64 = base_character.resize((64, 64), Image.Resampling.NEAREST)
        arr = np.array(base_64, dtype=np.float32) / 255.0
        t_base = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        action_name = action.lower()
        offsets = ACTION_LATENT_OFFSETS.get(action_name, ACTION_LATENT_OFFSETS["run"])

        frames = []
        with torch.no_grad():
            z_latent = self.encoder(t_base)
            cond = torch.zeros(1, 32, device=self.device)

            for i in range(num_frames):
                offset_val = offsets[i % len(offsets)]

                # Apply neural latent perturbation for action pose synthesis
                motion_dir = torch.sin(torch.linspace(0, np.pi, 64, device=self.device)) * offset_val
                z_frame = z_latent + motion_dir.unsqueeze(0)

                rec_tensor = self.generator(z_frame, cond)

                # Blend with base character tensor for strict identity preservation
                blended_t = 0.8 * t_base + 0.2 * rec_tensor
                blended_arr = (blended_t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)

                frame_raw = Image.fromarray(blended_arr, mode="RGBA")
                frame_quant = quantize_to_pixel_art(frame_raw, size=(64, 64), palette=SIGNATURE_PALETTE)
                frames.append(frame_quant)

        # Create Sprite Sheet
        sheet = Image.new("RGBA", (64 * len(frames), 64), (0, 0, 0, 0))
        for idx, f in enumerate(frames):
            sheet.paste(f, (idx * 64, 0))

        # Create Animated GIF
        gif_io = imageio.mimsave(
            imageio.RETURN_BYTES,
            [np.array(f) for f in frames],
            format="GIF",
            duration=0.15,
            loop=0
        )

        return frames, sheet, gif_io
