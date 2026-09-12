"""
Image-to-Sprite Converter Module (img2img)
Converts any external image/photo/drawing into a 64x64 pixel art sprite character,
preserving silhouette, colors, and identity.
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from .palette import quantize_to_pixel_art, SIGNATURE_PALETTE
from .model import PixelSpriteEncoder, PixelSpriteGenerator


class ImageToSpriteConverter:
    """
    Converts any input image into a 64x64 pixel art sprite matching
    the engine's signature pixel style and identity.
    """
    def __init__(self, engine=None, device="cpu"):
        self.device = torch.device(device)
        if engine is not None:
            self.encoder = getattr(engine, 'sprite_encoder', engine.encoder)
            self.generator = engine.generator
        else:
            self.encoder = PixelSpriteEncoder(latent_dim=64).to(self.device)
            self.generator = PixelSpriteGenerator(latent_dim=64, condition_dim=32).to(self.device)

        self.encoder.eval()
        self.generator.eval()

    def convert_image_to_sprite(self, input_image: Image.Image, strength: float = 0.5) -> Image.Image:
        """
        Takes any input PIL Image (photo, avatar, sketch, drawing) and converts it
        into a 64x64 signature pixel art character.
        """
        if input_image.mode != "RGBA":
            input_image = input_image.convert("RGBA")

        # 1. Downsample to 64x64 with high-quality Lanczos or Bilinear for color blending
        downscaled = input_image.resize((64, 64), Image.Resampling.LANCZOS)

        # 2. Extract tensor for encoder
        arr = np.array(downscaled, dtype=np.float32) / 255.0 # (64, 64, 4)
        t_img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device) # (1, 4, 64, 64)

        with torch.no_grad():
            # Latent encoding of input character
            z_latent = self.encoder(t_img)

            # Neural reconstruction from latent
            cond = torch.zeros(1, 32, device=self.device)
            rec_tensor = self.generator(z_latent, cond)

        # Blend original structure with neural latent reconstruction based on strength
        blended_t = (1.0 - strength) * t_img + strength * rec_tensor
        blended_arr = (blended_t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)

        blended_img = Image.fromarray(blended_arr, mode="RGBA")

        # 3. Apply exact signature pixel art quantization and sharp pixel grid alignment
        sprite_img = quantize_to_pixel_art(blended_img, size=(64, 64), palette=SIGNATURE_PALETTE)
        return sprite_img
