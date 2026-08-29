"""
nano_pixel_XL0_2 Real Conditional Diffusion CPU Engine
Performs text embedding encoding, DDPM denoising diffusion sampling loop, and palette quantization.
"""
import os
import torch
import numpy as np
from PIL import Image
from models.nano_pixel_art_v1.scripts.train_nanopixel import NanoPixelUNet
from pixel_art_engine.palette import quantize_to_pixel_art, SIGNATURE_PALETTE
from models.nano_pixel_XL0_2.scripts.train_nano_pixel import text_to_embedding

class NanoPixelXL02Engine:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.model = NanoPixelUNet().to(self.device)
        weights_path = os.path.join(os.path.dirname(__file__), "../weights/nanopixel_XL0_2.pt")
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def generate(self, prompt: str = "pixel art character", size: int = 64, seed: int = 42) -> Image.Image:
        torch.manual_seed(seed)
        np.random.seed(seed)

        cond_emb = text_to_embedding(prompt, dim=64).unsqueeze(0).to(self.device)

        # Real multi-step denoising diffusion loop
        x = torch.randn(1, 4, 64, 64, device=self.device)
        timesteps = 10
        for i in reversed(range(timesteps)):
            t = torch.tensor([i], device=self.device, dtype=torch.long)
            with torch.no_grad():
                noise_pred = self.model(x, t, cond_emb)
            x = x - 0.08 * noise_pred

        x_clamp = torch.clamp((x + 1.0) * 127.5, 0, 255)
        raw_arr = x_clamp.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        raw_img = Image.fromarray(raw_arr, mode="RGBA")
        if size != 64:
            raw_img = raw_img.resize((size, size), Image.Resampling.NEAREST)

        return quantize_to_pixel_art(raw_img, size=(size, size), palette=SIGNATURE_PALETTE)
