import os
import torch
import numpy as np
from PIL import Image
from pixel_art_engine.clip_text import SimpleCLIPTextEncoder
from models.nano_pixel_mom.model import NanoPixelMomUNet, NanoPixelMomDecoder
from pixel_art_engine.palette import quantize_to_pixel_art, SIGNATURE_PALETTE

class NanoPixelMomGenerator:
    """
    Inference Engine for Nano Pixel MOM (Massive Output Model).
    Loads trained NanoPixelMomUNet and NanoPixelMomDecoder weights and executes
    neural diffusion denoising + VAE decoding directly on CPU.
    """
    def __init__(self, unet_path="models/nano_pixel_mom/weights/nano_pixel_mom_unet.pt",
                 decoder_path="models/nano_pixel_mom/weights/nano_pixel_mom_decoder.pt",
                 device="cpu"):
        self.device = torch.device(device)
        self.text_encoder = SimpleCLIPTextEncoder()
        self.unet = NanoPixelMomUNet().to(self.device)
        self.decoder = NanoPixelMomDecoder().to(self.device)

        if os.path.exists(unet_path):
            self.unet.load_state_dict(torch.load(unet_path, map_location=self.device))
        if os.path.exists(decoder_path):
            self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))

        self.unet.eval()
        self.decoder.eval()

    def generate(self, prompt="pixel knight warrior", steps=5, seed=42):
        torch.manual_seed(seed)
        text_embed = self.text_encoder(prompt)
        latent = torch.randn(1, 4, 32, 32, device=self.device)

        with torch.no_grad():
            denoised = latent
            for step in range(steps, 0, -1):
                t = torch.tensor([[float(step)]], device=self.device)
                denoised = self.unet(latent, t, text_embed)
                latent = latent - 0.1 * denoised

            # Decode the denoised latent representation to RGB/RGBA
            rgb_tensor = self.decoder(denoised)

        img_np = (rgb_tensor[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        img_raw = Image.fromarray(img_np, mode="RGBA")
        return quantize_to_pixel_art(img_raw, size=(64, 64), palette=SIGNATURE_PALETTE)
