"""
NanoPixel-v1 Pixel-Space UNet Diffusion Trainer
Trains directly in Pixel Space (64x64 / 128x128) without lossy VAE compression.
Uses exact trigger prefix: 'pixel art, 16-bit, gba style, <character_type>'
AdamW optimizer with LR=1e-5 and Cosine Annealing scheduler.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
from models.nano_pixel_art_v1.scripts.train_nanopixel import NanoPixelUNet, CharbonnierLoss, PaletteConsistencyLoss
from models.nano_pixel_art_v1.scripts.export_onnx import export_onnx

TRIGGER_PREFIX = "pixel art, 16-bit, gba style"
PROC_64_DIR = "models/nano_pixel_art_v1/dataset/processed_64x64"
PROC_128_DIR = "models/nano_pixel_art_v1/dataset/processed_128x128"

def caption_to_prompt_embedding(caption: str, dim: int = 64) -> torch.Tensor:
    """Encodes prompt caption string containing trigger prefix into 64-dim conditioning vector."""
    if not caption.startswith(TRIGGER_PREFIX):
        caption = f"{TRIGGER_PREFIX}, {caption}"

    vec = torch.zeros(dim)
    words = caption.lower().replace(',', '').split()
    for idx, word in enumerate(words):
        h = sum(ord(c) for c in word)
        vec[h % dim] += 1.0 / (idx + 1)
    norm = torch.norm(vec)
    return vec / (norm + 1e-6)

def train_pixel_space_diffusion(learning_rate: float = 1e-5, max_epochs: int = 100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[NanoPixel-v1] Training in Pixel Space on {device} with LR={learning_rate}...")

    model = NanoPixelUNet().to(device)
    charbonnier = CharbonnierLoss()
    palette_loss = PaletteConsistencyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    # Load image-caption pairs
    dataset = []
    for proc_dir in [PROC_64_DIR, PROC_128_DIR]:
        if not os.path.exists(proc_dir):
            continue
        for f in os.listdir(proc_dir):
            if f.endswith('.png'):
                img_path = os.path.join(proc_dir, f)
                txt_path = os.path.join(proc_dir, f.replace('.png', '.txt'))

                caption = "character, idle, full body, isolated background"
                if os.path.exists(txt_path):
                    with open(txt_path, 'r') as tf:
                        caption = tf.read().strip()

                im = Image.open(img_path).convert('RGBA').resize((64, 64), Image.Resampling.NEAREST)
                arr = np.array(im, dtype=np.float32) / 127.5 - 1.0 # [-1, 1] pixel space
                t_img = torch.from_numpy(arr).permute(2, 0, 1)
                t_cond = caption_to_prompt_embedding(caption, dim=64)
                dataset.append((t_img, t_cond))

    print(f"[NanoPixel-v1] Loaded {len(dataset)} pixel-space training samples.")

    if len(dataset) > 0:
        model.train()
        batch_size = 16
        imgs = torch.stack([item[0] for item in dataset]).to(device)
        conds = torch.stack([item[1] for item in dataset]).to(device)
        num_items = len(imgs)

        for epoch in range(1, max_epochs + 1):
            perm = torch.randperm(num_items)
            total_loss = 0.0
            batches = 0

            for i in range(0, num_items, batch_size):
                idx = perm[i:i+batch_size]
                x0 = imgs[idx]
                c_emb = conds[idx]
                b_sz = x0.size(0)

                t = torch.randint(0, 20, (b_sz,), device=device).long()
                noise = torch.randn_like(x0)

                # Standard DDPM forward diffusion step in Pixel Space
                alpha_t = 1.0 - (t.float() / 20.0 + 0.01)[..., None, None, None]
                xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(1.0 - alpha_t) * noise

                pred_noise = model(xt, t, c_emb)
                loss = charbonnier(pred_noise, noise) + 0.05 * palette_loss(pred_noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batches += 1

            scheduler.step()
            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{max_epochs}] Pixel-Space Loss: {total_loss/max(1, batches):.4f}")

    weights_dir = "models/nano_pixel_art_v1/weights"
    os.makedirs(weights_dir, exist_ok=True)
    pt_path = os.path.join(weights_dir, "nanopixel_v1.pt")
    torch.save(model.state_dict(), pt_path)
    print(f"[NanoPixel-v1] Saved pixel-space model weights to {pt_path}")

    export_onnx()

if __name__ == "__main__":
    train_pixel_space_diffusion()
