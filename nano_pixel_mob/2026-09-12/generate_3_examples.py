import os
import sys
import torch
import numpy as np
from PIL import Image

base_dir = os.path.dirname(os.path.abspath(__file__))

from pixel_llm import PixelLLM
from discrete_diffusion import ConvDiscreteDiffusion
from vqvae_prior import VQVAEModel, LatentPriorTransformer

def colorize_and_upscale(indices, palette):
    img_arr = np.zeros((32, 32, 4), dtype=np.uint8)
    for r in range(32):
        for c in range(32):
            idx = indices[r, c]
            if 1 <= idx <= 16:
                color = palette[min(idx-1, len(palette)-1)]
                img_arr[r, c] = [color[0], color[1], color[2], 255]
    img = Image.fromarray(img_arr, mode='RGBA')
    return img.resize((64, 64), Image.NEAREST)

def generate_3_examples_per_model():
    dataset_path = os.path.join(base_dir, "dataset", "arcade_sprites_32x32.pt")
    data = torch.load(dataset_path)
    palettes = data["palettes"].numpy()

    # 1. Model 1
    m1 = PixelLLM()
    m1.load_state_dict(torch.load(os.path.join(base_dir, "1_pixel_llm", "pixel_llm_weights.pt")))
    dir1 = os.path.join(base_dir, "1_pixel_llm")
    for i in range(3):
        palette = palettes[i % len(palettes)]
        indices = m1.generate(temperature=0.85, top_k=8)
        img = colorize_and_upscale(indices, palette)
        img.save(os.path.join(dir1, f"example_0{i+1}.png"))

    # 2. Model 2
    m2 = ConvDiscreteDiffusion()
    m2.load_state_dict(torch.load(os.path.join(base_dir, "2_discrete_diffusion", "discrete_diffusion_weights.pt")))
    dir2 = os.path.join(base_dir, "2_discrete_diffusion")
    for i in range(3):
        palette = palettes[(i+3) % len(palettes)]
        indices = m2.sample(steps=10)
        img = colorize_and_upscale(indices, palette)
        img.save(os.path.join(dir2, f"example_0{i+1}.png"))

    # 3. Model 3
    m3_vqvae = VQVAEModel()
    m3_vqvae.load_state_dict(torch.load(os.path.join(base_dir, "3_vqvae_prior", "vqvae_weights.pt")))
    m3_prior = LatentPriorTransformer()
    m3_prior.load_state_dict(torch.load(os.path.join(base_dir, "3_vqvae_prior", "prior_weights.pt")))
    dir3 = os.path.join(base_dir, "3_vqvae_prior")
    for i in range(3):
        palette = palettes[(i+6) % len(palettes)]
        sampled_codes = m3_prior.generate()
        indices = m3_vqvae.decode_indices(sampled_codes)[0].cpu().numpy()
        img = colorize_and_upscale(indices, palette)
        img.save(os.path.join(dir3, f"example_0{i+1}.png"))

if __name__ == "__main__":
    generate_3_examples_per_model()
