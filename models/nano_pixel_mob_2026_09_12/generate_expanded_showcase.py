import os
import sys
import torch
import numpy as np
from PIL import Image
import imageio
import importlib.util

def load_module_from_path(module_name, filepath):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

mod1 = load_module_from_path("pixel_llm_mod", "models/nano_pixel_mob_2026_09_12/1_pixel_llm/pixel_llm.py")
mod2 = load_module_from_path("diffusion_mod", "models/nano_pixel_mob_2026_09_12/2_discrete_diffusion/discrete_diffusion.py")
mod3 = load_module_from_path("vqvae_mod", "models/nano_pixel_mob_2026_09_12/3_vqvae_prior/vqvae_prior.py")

PixelLLM = mod1.PixelLLM
ConvDiscreteDiffusion = mod2.ConvDiscreteDiffusion
VQVAEModel = mod3.VQVAEModel
LatentPriorTransformer = mod3.LatentPriorTransformer

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

def generate_expanded_showcase():
    print("Generating expanded 32-character showcase in examples/2026-09-12_nano_pixel_mob/...")
    dataset_path = "models/nano_pixel_mob_2026_09_12/dataset/arcade_sprites_32x32.pt"
    data = torch.load(dataset_path)
    palettes = data["palettes"].numpy()

    examples_dir = "examples/2026-09-12_nano_pixel_mob"
    os.makedirs(examples_dir, exist_ok=True)

    # Load Models
    m1 = PixelLLM()
    m1.load_state_dict(torch.load("models/nano_pixel_mob_2026_09_12/1_pixel_llm/pixel_llm_weights.pt"))

    m2 = ConvDiscreteDiffusion()
    m2.load_state_dict(torch.load("models/nano_pixel_mob_2026_09_12/2_discrete_diffusion/discrete_diffusion_weights.pt"))

    m3_vqvae = VQVAEModel()
    m3_vqvae.load_state_dict(torch.load("models/nano_pixel_mob_2026_09_12/3_vqvae_prior/vqvae_weights.pt"))
    m3_prior = LatentPriorTransformer()
    m3_prior.load_state_dict(torch.load("models/nano_pixel_mob_2026_09_12/3_vqvae_prior/prior_weights.pt"))

    models = [
        ("1_pixel_llm", lambda: m1.generate(temperature=0.85, top_k=8)),
        ("2_discrete_diffusion", lambda: m2.sample(steps=10)),
        ("3_vqvae_prior", lambda: m3_vqvae.decode_indices(m3_prior.generate())[0].cpu().numpy())
    ]

    all_32_images = []

    for model_name, gen_fn in models:
        model_out_dir = os.path.join(examples_dir, model_name)
        os.makedirs(model_out_dir, exist_ok=True)

        imgs = []
        for i in range(12): # 12 per model
            palette = palettes[i % len(palettes)]
            indices = gen_fn()
            img_64 = colorize_and_upscale(indices, palette)
            img_64.save(os.path.join(model_out_dir, f"character_{i+1:02d}.png"))
            imgs.append(img_64)
            if len(all_32_images) < 32:
                all_32_images.append(img_64)

        # Model sprite sheet (3x4 grid)
        sheet = Image.new("RGBA", (64 * 4, 64 * 3), (0, 0, 0, 0))
        for idx, img in enumerate(imgs):
            r = idx // 4
            c = idx % 4
            sheet.paste(img, (c * 64, r * 64))
        sheet.save(os.path.join(model_out_dir, "sprite_sheet.png"))
        imageio.mimsave(os.path.join(model_out_dir, "preview.gif"), [np.array(im) for im in imgs], fps=3, loop=0)

    # Master 32-Character Sprite Sheet (4x8 grid)
    master_sheet = Image.new("RGBA", (64 * 8, 64 * 4), (20, 20, 25, 255))
    for idx, img in enumerate(all_32_images[:32]):
        r = idx // 8
        c = idx % 8
        master_sheet.paste(img, (c * 64, r * 64), mask=img)
    master_sheet.save(os.path.join(examples_dir, "master_32_characters_sheet.png"))
    imageio.mimsave(os.path.join(examples_dir, "master_preview.gif"), [np.array(im) for im in all_32_images[:32]], fps=3, loop=0)

    readme_content = """# Nano Pixel Mob - Showcase Results (2026-09-12)

هذا الأرشيف يحتوي على صور وتنائج النماذج التوليدية لـ **Nano Pixel Mob**:

1. `1_pixel_llm/`: نتائج نموذج PixelLLM التوليدي (12 شخصية 64x64 + Sprite Sheet + GIF).
2. `2_discrete_diffusion/`: نتائج نموذج Discrete Masked Diffusion (12 شخصية 64x64 + Sprite Sheet + GIF).
3. `3_vqvae_prior/`: نتائج نموذج VQ-VAE + Neural Prior (12 شخصية 64x64 + Sprite Sheet + GIF).
4. `master_32_characters_sheet.png`: تجميعة 32 شخصية فريدة مولدة بالذكاء الاصطناعي بالكامل على الـ CPU.
5. `master_preview.gif`: عرض متحرك للشخصيات الـ 32.
"""
    with open(os.path.join(examples_dir, "README.md"), "w") as f:
        f.write(readme_content)

    print("Expanded showcase generated successfully.")

if __name__ == "__main__":
    generate_expanded_showcase()
