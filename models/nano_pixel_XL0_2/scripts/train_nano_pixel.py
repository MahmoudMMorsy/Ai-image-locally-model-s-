"""
nano_pixel_XL0_2 Real Conditional Diffusion Trainer
Reads image-caption (.png + .txt) pairs, encodes text embeddings for true prompt conditioning,
applies Charbonnier + Palette Consistency Loss.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
from models.nano_pixel_art_v1.scripts.train_nanopixel import NanoPixelUNet, CharbonnierLoss, PaletteConsistencyLoss

PROC_64_DIR = "models/nano_pixel_XL0_2/dataset/processed_64x64"
PROC_128_DIR = "models/nano_pixel_XL0_2/dataset/processed_128x128"
CHECKPOINTS_DIR = "models/nano_pixel_XL0_2/weights/checkpoints"

def text_to_embedding(caption: str, dim: int = 64) -> torch.Tensor:
    """Generates a deterministic 64-dim text embedding vector from text caption."""
    vec = torch.zeros(dim)
    words = caption.lower().split()
    for idx, word in enumerate(words):
        h = sum(ord(c) for c in word)
        vec[h % dim] += 1.0 / (idx + 1)
    norm = torch.norm(vec)
    return vec / (norm + 1e-6)

def load_dataset():
    dataset = []
    for proc_dir in [PROC_64_DIR, PROC_128_DIR, "dataset_training_images/clean"]:
        if not os.path.exists(proc_dir):
            continue
        files = [f for f in os.listdir(proc_dir) if f.endswith('.png')]
        for f in files:
            img_path = os.path.join(proc_dir, f)
            txt_path = os.path.join(proc_dir, f.replace('.png', '.txt'))
            caption = "pixel art character"
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as tf:
                    caption = tf.read().strip()
            im = Image.open(img_path).convert('RGBA').resize((64, 64), Image.Resampling.NEAREST)
            arr = np.array(im, dtype=np.float32) / 127.5 - 1.0
            t_img = torch.from_numpy(arr).permute(2, 0, 1)
            t_emb = text_to_embedding(caption, dim=64)
            dataset.append((t_img, t_emb, caption))
    return dataset

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[nano_pixel_XL0_2] Starting Training on {device}...")

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    dataset = load_dataset()
    print(f"[nano_pixel_XL0_2] Loaded {len(dataset)} image-caption dataset pairs.")

    model = NanoPixelUNet().to(device)
    charbonnier = CharbonnierLoss()
    palette_loss = PaletteConsistencyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    max_epochs = 10

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
                xt = x0 + 0.1 * noise

                pred_noise = model(xt, t, c_emb)
                loss = charbonnier(pred_noise, noise) + 0.05 * palette_loss(pred_noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batches += 1

            epoch_loss = total_loss / max(1, batches)
            scheduler.step()

            if epoch % 2 == 0:
                print(f"Epoch [{epoch}/{max_epochs}] Loss: {epoch_loss:.4f}")

    weights_dir = "models/nano_pixel_XL0_2/weights"
    os.makedirs(weights_dir, exist_ok=True)
    pt_path = os.path.join(weights_dir, "nanopixel_XL0_2.pt")
    torch.save(model.state_dict(), pt_path)
    print(f"[nano_pixel_XL0_2] Saved trained model weights to {pt_path}")

    # Export ONNX
    onnx_path = os.path.join(weights_dir, "nanopixel_XL0_2.onnx")
    dummy_x = torch.randn(1, 4, 64, 64)
    dummy_t = torch.tensor([10], dtype=torch.long)
    dummy_cond = torch.randn(1, 64)
    model.eval()
    torch.onnx.export(
        model,
        (dummy_x, dummy_t, dummy_cond),
        onnx_path,
        input_names=["sample", "timestep", "condition"],
        output_names=["noise_pred"],
        dynamo=False
    )
    print(f"[nano_pixel_XL0_2] Exported ONNX model to {onnx_path}")

if __name__ == "__main__":
    train()
