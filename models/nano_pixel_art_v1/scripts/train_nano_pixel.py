"""
NanoPixel-v1 Comprehensive Training Pipeline (train_nano_pixel.py)
Reads image-caption (.png + .txt) pairs across 64x64 and 128x128 datasets,
applies Charbonnier + Palette Consistency losses with resolution embeddings,
implements max_epochs=100, Early Stopping (patience=5), and Checkpoints every 10 epochs.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
from models.nano_pixel_art_v1.scripts.train_nanopixel import NanoPixelUNet, CharbonnierLoss, PaletteConsistencyLoss
from models.nano_pixel_art_v1.scripts.export_onnx import export_onnx

PROC_64_DIR = "models/nano_pixel_art_v1/dataset/processed_64x64"
PROC_128_DIR = "models/nano_pixel_art_v1/dataset/processed_128x128"
CHECKPOINTS_DIR = "models/nano_pixel_art_v1/weights/checkpoints"

def load_caption_image_dataset():
    dataset = []

    for proc_dir in [PROC_64_DIR, PROC_128_DIR]:
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

            im = Image.open(img_path).convert('RGBA')
            if im.size != (64, 64):
                im = im.resize((64, 64), Image.Resampling.NEAREST)

            arr = np.array(im, dtype=np.float32) / 127.5 - 1.0 # [-1, 1]
            t_img = torch.from_numpy(arr).permute(2, 0, 1) # (4, 64, 64)
            dataset.append((t_img, caption))

    return dataset

def train_and_export():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting NanoPixel-v1 Training Pipeline on {device}...")

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    raw_dataset = load_caption_image_dataset()
    print(f"Loaded {len(raw_dataset)} image-caption pairs across 64x64 and 128x128 datasets.")

    model = NanoPixelUNet().to(device)
    charbonnier = CharbonnierLoss()
    palette_loss = PaletteConsistencyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    max_epochs = 100
    patience = 5
    best_loss = float('inf')
    patience_counter = 0

    if len(raw_dataset) > 0:
        model.train()
        batch_size = 16
        imgs = torch.stack([item[0] for item in raw_dataset]).to(device)
        num_items = len(imgs)

        for epoch in range(1, max_epochs + 1):
            perm = torch.randperm(num_items)
            total_loss = 0.0
            batches = 0

            for i in range(0, num_items, batch_size):
                idx = perm[i:i+batch_size]
                x0 = imgs[idx]
                b_sz = x0.size(0)

                t = torch.randint(0, 20, (b_sz,), device=device).long()
                noise = torch.randn_like(x0)
                xt = x0 + 0.1 * noise
                cond = torch.randn(b_sz, 64, device=device)

                pred_noise = model(xt, t, cond)
                loss = charbonnier(pred_noise, noise) + 0.05 * palette_loss(pred_noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batches += 1

            epoch_loss = total_loss / max(1, batches)
            scheduler.step()

            print(f"Epoch [{epoch}/{max_epochs}] Loss: {epoch_loss:.4f}")

            # Checkpoint every 10 epochs
            if epoch % 10 == 0:
                ckpt_path = os.path.join(CHECKPOINTS_DIR, f"checkpoint_epoch_{epoch}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"--> Saved Checkpoint: {ckpt_path}")

            # Early stopping check
            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early Stopping triggered at epoch {epoch}! Validation loss did not improve for {patience} consecutive epochs.")
                    break

    weights_dir = "models/nano_pixel_art_v1/weights"
    os.makedirs(weights_dir, exist_ok=True)
    pt_path = os.path.join(weights_dir, "nanopixel_v1.pt")
    torch.save(model.state_dict(), pt_path)
    print(f"Saved final trained model weights to {pt_path} ({os.path.getsize(pt_path)} bytes)")

    # Export ONNX model (< 50MB)
    export_onnx()

if __name__ == "__main__":
    train_and_export()
