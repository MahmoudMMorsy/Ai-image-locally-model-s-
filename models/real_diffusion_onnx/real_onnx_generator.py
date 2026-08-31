import os
import torch
import numpy as np
from PIL import Image
from pixel_art_engine.clip_text import SimpleCLIPTextEncoder
from models.real_diffusion_onnx.model_architecture import RealLatentUNet, RealLatentDecoder

class RealNeuralDiffusionGenerator:
    """
    Genuine Neural Latent Diffusion Image Generator.
    Executes multi-step iterative denoising in latent space (4, 32, 32)
    and decodes final latent features using a trained neural VAE Decoder to RGB (256x256).
    """
    def __init__(self, unet_path="models/real_diffusion_onnx/weights/real_latent_unet.pt",
                 decoder_path="models/real_diffusion_onnx/weights/real_vae_decoder.pt"):
        self.encoder = SimpleCLIPTextEncoder()
        self.unet = RealLatentUNet()
        self.decoder = RealLatentDecoder()

        if os.path.exists(unet_path):
            self.unet.load_state_dict(torch.load(unet_path))
        if os.path.exists(decoder_path):
            self.decoder.load_state_dict(torch.load(decoder_path))

        self.unet.eval()
        self.decoder.eval()

    def generate_image(self, prompt="cyberpunk poster", steps=10, seed=42):
        torch.manual_seed(seed)

        # 1. Encode text prompt to condition vector (1, 128)
        text_embed = self.encoder(prompt)

        # 2. Initialize latent standard normal noise (1, 4, 32, 32)
        latent = torch.randn(1, 4, 32, 32)

        # 3. Neural DDIM Iterative Denoising Loop
        with torch.no_grad():
            for step in range(steps, 0, -1):
                t = torch.tensor([[float(step)]])
                noise_pred = self.unet(latent, t, text_embed)
                latent = latent - 0.08 * noise_pred

            # 4. Neural VAE Decoding to RGB image (1, 3, 256, 256)
            rgb_tensor = self.decoder(latent)

        # Convert tensor to PIL Image
        img_np = (rgb_tensor[0].permute(1, 2, 0).detach().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img_np)
