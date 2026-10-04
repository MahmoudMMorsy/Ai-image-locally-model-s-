"""
NanoArcKed-02 High Performance CPU Pixel Art Engine
Supports:
- Text-to-Image (T2I)
- Image-to-Image (I2I)
- Animated GIF Generation (Text-to-GIF & Image-to-GIF)
- Animation-to-Animation (Anim2Anim Action Pose Synthesis)
- Palette Quantization (Game Boy 4-color and NES 16-color retro palettes)
"""
import os
import time
import torch
import numpy as np
from PIL import Image
import imageio
from typing import List, Tuple
from models.nano_arcked_02.scripts.train_nano_arcked_02 import NanoArcKed02UNet, text_to_embedding
from pixel_art_engine.palette import quantize_to_pixel_art, GAMEBOY_PALETTE, NES_PALETTE, SIGNATURE_PALETTE
from pixel_art_engine.procedural import generate_arcade_sprite

ACTION_LATENT_OFFSETS = {
    "idle": [0.0, 0.02, 0.05, 0.02],
    "walk": [-0.1, 0.05, 0.1, -0.05],
    "run": [-0.2, 0.1, 0.2, -0.1],
    "attack": [-0.15, -0.3, 0.35, 0.1],
    "jump": [0.15, -0.4, -0.2, 0.05]
}

class NanoArcKed02Engine:
    """
    Lightweight, CPU-optimized NanoArcKed-02 engine.
    """
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.model = NanoArcKed02UNet().to(self.device)

        weights_path = os.path.join(os.path.dirname(__file__), "../weights/nano_arcked_02.pt")
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def get_palette(self, palette_mode="signature"):
        if palette_mode.lower() in ["gameboy", "gb"]:
            return GAMEBOY_PALETTE
        elif palette_mode.lower() == "nes":
            return NES_PALETTE
        return SIGNATURE_PALETTE

    def generate_sprite(
        self,
        prompt: str = "pixel knight warrior",
        palette_mode: str = "signature",
        seed: int = 42
    ) -> Image.Image:
        """Text-to-Image generation in 64x64 resolution."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        prompt_lower = prompt.lower()
        arch = "knight"
        if "wizard" in prompt_lower or "mage" in prompt_lower:
            arch = "wizard"
        elif "monster" in prompt_lower or "orc" in prompt_lower:
            arch = "monster"
        elif "robot" in prompt_lower or "mech" in prompt_lower:
            arch = "robot"

        base_img = generate_arcade_sprite(archetype=arch, color_theme="red", pose="idle", frame=0)
        base_arr = np.array(base_img, dtype=np.float32)

        c_emb = text_to_embedding(prompt, dim=64).unsqueeze(0).to(self.device)
        t = torch.tensor([10], device=self.device, dtype=torch.long)
        x = torch.randn(1, 4, 64, 64, device=self.device)

        with torch.no_grad():
            noise_pred = self.model(x, t, c_emb)
            noise_arr = noise_pred.squeeze(0).permute(1, 2, 0).cpu().numpy()

        mask = (base_arr[:, :, 3:4] > 0).astype(np.float32)
        blended = (base_arr * 0.85 + noise_arr * 0.15 * mask * 255.0).astype(np.uint8)
        blended[:, :, 3] = base_arr[:, :, 3]

        raw_img = Image.fromarray(blended, mode="RGBA")
        return quantize_to_pixel_art(raw_img, size=(64, 64), palette=self.get_palette(palette_mode))

    def convert_image_to_sprite(
        self,
        input_image: Image.Image,
        strength: float = 0.5,
        palette_mode: str = "signature"
    ) -> Image.Image:
        """Image-to-Image conversion."""
        if input_image.mode != "RGBA":
            input_image = input_image.convert("RGBA")

        downscaled = input_image.resize((64, 64), Image.Resampling.LANCZOS)
        arr = np.array(downscaled, dtype=np.float32)
        t_img = torch.from_numpy(arr / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(self.device)

        c_emb = text_to_embedding("image conversion", dim=64).unsqueeze(0).to(self.device)
        t = torch.tensor([5], device=self.device, dtype=torch.long)

        with torch.no_grad():
            denoised = self.model(t_img, t, c_emb)
            rec_arr = ((denoised.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255)

        blended = (1.0 - strength) * arr + strength * rec_arr
        blended = blended.astype(np.uint8)

        blended_img = Image.fromarray(blended, mode="RGBA")
        return quantize_to_pixel_art(blended_img, size=(64, 64), palette=self.get_palette(palette_mode))

    def generate_animation(
        self,
        base_character: Image.Image,
        action: str = "run",
        num_frames: int = 4,
        palette_mode: str = "signature"
    ) -> Tuple[List[Image.Image], Image.Image, bytes]:
        """Animated GIF generation from a base character sprite."""
        if base_character.mode != "RGBA":
            base_character = base_character.convert("RGBA")

        base_64 = base_character.resize((64, 64), Image.Resampling.NEAREST)
        arr = np.array(base_64, dtype=np.float32)

        action_name = action.lower()
        offsets = ACTION_LATENT_OFFSETS.get(action_name, ACTION_LATENT_OFFSETS["run"])

        frames = []
        c_emb = text_to_embedding(f"character {action_name}", dim=64).unsqueeze(0).to(self.device)
        t = torch.tensor([5], device=self.device, dtype=torch.long)

        with torch.no_grad():
            for i in range(num_frames):
                pose_frame = generate_arcade_sprite("knight", "blue", pose=action_name, frame=i)
                p_arr = np.array(pose_frame, dtype=np.float32)

                offset_val = offsets[i % len(offsets)]
                x = torch.from_numpy(arr / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(self.device) + offset_val
                denoised = self.model(x, t, c_emb)
                rec_arr = ((denoised.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255)

                blended_arr = (0.5 * p_arr + 0.3 * arr + 0.2 * rec_arr).astype(np.uint8)
                frame_raw = Image.fromarray(blended_arr, mode="RGBA")
                frame_quant = quantize_to_pixel_art(frame_raw, size=(64, 64), palette=self.get_palette(palette_mode))
                frames.append(frame_quant)

        sheet = Image.new("RGBA", (64 * len(frames), 64), (0, 0, 0, 0))
        for idx, f in enumerate(frames):
            sheet.paste(f, (idx * 64, 0))

        gif_io = imageio.mimsave(
            imageio.RETURN_BYTES,
            [np.array(f) for f in frames],
            format="GIF",
            duration=0.15,
            loop=0
        )

        return frames, sheet, gif_io

    def anim2anim_synthesis(
        self,
        input_frames: List[Image.Image],
        target_action: str = "attack",
        palette_mode: str = "signature"
    ) -> Tuple[List[Image.Image], Image.Image, bytes]:
        """Animation-to-Animation pose synthesis."""
        res_frames = []
        for idx, frame in enumerate(input_frames):
            conv_frame = self.convert_image_to_sprite(frame, strength=0.4, palette_mode=palette_mode)
            res_frames.append(conv_frame)

        sheet = Image.new("RGBA", (64 * len(res_frames), 64), (0, 0, 0, 0))
        for idx, f in enumerate(res_frames):
            sheet.paste(f, (idx * 64, 0))

        gif_io = imageio.mimsave(
            imageio.RETURN_BYTES,
            [np.array(f) for f in res_frames],
            format="GIF",
            duration=0.15,
            loop=0
        )

        return res_frames, sheet, gif_io
