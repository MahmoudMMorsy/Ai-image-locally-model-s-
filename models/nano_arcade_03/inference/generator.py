"""
Nano Arcade 03 Inference Generator Module
Provides Text-to-Image, Image-to-Image (Img2Img), and GIF Animation generation for 64x64 pixel sprites.
"""
import os
import io
import torch
import numpy as np
from PIL import Image
import imageio
from pixel_art_engine.palette import quantize_to_pixel_art, SIGNATURE_PALETTE
from models.nano_arcade_03.scripts.train_nano_arcade import NanoArcade03UNet, text_to_embedding

class NanoArcade03Generator:
    def __init__(self, weights_path="models/nano_arcade_03/weights/nano_arcade_03.pt", device="cpu"):
        self.device = torch.device(device)
        self.model = NanoArcade03UNet().to(self.device)

        if weights_path and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.eval()

    def text_to_sprite(self, prompt: str, seed: int = 42, steps: int = 5) -> Image.Image:
        torch.manual_seed(seed)
        c_emb = text_to_embedding(prompt, dim=64).unsqueeze(0).to(self.device)
        x = torch.randn(1, 4, 64, 64, device=self.device)

        with torch.no_grad():
            for t_val in reversed(range(1, steps + 1)):
                t_tensor = torch.tensor([float(t_val)], device=self.device)
                pred = self.model(x, t_tensor, c_emb)
                x = (x + pred) / 2.0

        arr = (x.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        raw_img = Image.fromarray(arr, mode="RGBA")
        return quantize_to_pixel_art(raw_img, size=(64, 64), palette=SIGNATURE_PALETTE)

    def image_to_sprite(self, input_image: Image.Image, strength: float = 0.5) -> Image.Image:
        if input_image.mode != "RGBA":
            input_image = input_image.convert("RGBA")
        resized = input_image.resize((64, 64), Image.Resampling.NEAREST)

        arr = np.array(resized, dtype=np.float32) / 255.0
        t_img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        c_emb = torch.zeros(1, 64, device=self.device)
        t = torch.tensor([5.0], device=self.device)

        with torch.no_grad():
            rec_tensor = self.model(t_img, t, c_emb)

        blended = (1.0 - strength) * t_img + strength * rec_tensor
        blended_arr = (blended.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        blended_img = Image.fromarray(blended_arr, mode="RGBA")
        return quantize_to_pixel_art(blended_img, size=(64, 64), palette=SIGNATURE_PALETTE)

    def generate_animation_gif(self, prompt: str, num_frames: int = 4) -> bytes:
        frames = []
        for f in range(num_frames):
            frame_prompt = f"{prompt}, action frame {f}"
            frame_img = self.text_to_sprite(frame_prompt, seed=42 + f)
            frames.append(np.array(frame_img))

        buf = io.BytesIO()
        imageio.mimwrite(buf, frames, format="GIF", duration=0.15)
        return buf.getvalue()
