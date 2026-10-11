import os
import torch
import numpy as np
from PIL import Image
from models.nano_pixel_mom.model import NanoPixelMomUNet, NanoPixelMomDecoder

class NanoPixelMomEngine:
    def __init__(self, weights_dir="models/nano_pixel_mom/weights"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.unet = NanoPixelMomUNet().to(self.device)
        self.decoder = NanoPixelMomDecoder().to(self.device)

        unet_path = os.path.join(weights_dir, "nano_pixel_mom_unet.pt")
        decoder_path = os.path.join(weights_dir, "nano_pixel_mom_decoder.pt")

        if os.path.exists(unet_path):
            self.unet.load_state_dict(torch.load(unet_path, map_location=self.device))
        if os.path.exists(decoder_path):
            self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))

        self.unet.eval()
        self.decoder.eval()

    def generate_sprite(self, prompt="character", seed=42, width=64, height=64):
        torch.manual_seed(seed)
        np.random.seed(seed)

        latent = torch.randn(1, 4, height, width, device=self.device)
        cond = torch.randn(1, 64, device=self.device)
        t = torch.tensor([10], device=self.device).long()

        with torch.no_grad():
            pred_noise = self.unet(latent, t, cond)
            decoded = self.decoder(pred_noise)

        # Denormalize [-1, 1] to [0, 255]
        img_np = decoded.squeeze(0).cpu().permute(1, 2, 0).numpy()
        img_np = np.clip((img_np * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)

        # Ensure RGBA image
        img = Image.fromarray(img_np, mode="RGBA")
        return img

    def create_sprite_sheet(self, frames):
        w, h = frames[0].size
        sheet = Image.new("RGBA", (w * len(frames), h), (0, 0, 0, 0))
        for idx, frame in enumerate(frames):
            sheet.paste(frame, (idx * w, 0))
        return sheet
