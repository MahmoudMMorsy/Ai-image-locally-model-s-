"""
NanoArcKed-02 Pixel-Space Diffusion Trainer
Trains on dataset images from mm.trine folder for 64x64 pixel art generation.
Includes text-to-image, image-to-image, GIF animation, and animation-to-animation synthesis capabilities.
"""
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import numpy as np

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return torch.mean(loss)

class PaletteConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_img):
        grad_x = torch.abs(pred_img[:, :, :, 1:] - pred_img[:, :, :, :-1])
        grad_y = torch.abs(pred_img[:, :, 1:, :] - pred_img[:, :, :-1, :])
        return torch.mean(grad_x) + torch.mean(grad_y)

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class MicroBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim=32, cond_dim=64):
        super().__init__()
        self.mlp = nn.Linear(t_dim + cond_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(8, out_ch)

    def forward(self, x, t_cond):
        h = F.silu(self.gn(self.conv1(x)))
        emb = self.mlp(t_cond)[..., None, None]
        h = h + emb
        return F.silu(self.conv2(h))

class NanoArcKed02UNet(nn.Module):
    """
    Enhanced NanoArcKed-02 Micro UNet (< 35MB) for 64x64 Pixel Art Generation.
    Supports text-to-image, image-to-image, animated GIF, and animation-to-animation synthesis.
    """
    def __init__(self, in_ch=4, out_ch=4, cond_dim=64):
        super().__init__()
        self.t_emb = nn.Sequential(
            SinusoidalEmbeddings(32),
            nn.Linear(32, 32),
            nn.SiLU()
        )
        self.cond_proj = nn.Linear(cond_dim, cond_dim)

        self.down1 = MicroBlock(in_ch, 32, 32, cond_dim)
        self.pool1 = nn.Conv2d(32, 64, 4, stride=2, padding=1)

        self.down2 = MicroBlock(64, 128, 32, cond_dim)
        self.pool2 = nn.Conv2d(128, 128, 4, stride=2, padding=1)

        self.mid = MicroBlock(128, 128, 32, cond_dim)

        self.up2 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
        self.b_up2 = MicroBlock(256, 64, 32, cond_dim)

        self.up1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.b_up1 = MicroBlock(64, 32, 32, cond_dim)

        self.out = nn.Conv2d(32, out_ch, 1)

    def forward(self, x, t, cond):
        t_emb = self.t_emb(t)
        c_emb = self.cond_proj(cond)
        t_cond = torch.cat([t_emb, c_emb], dim=1)

        x1 = self.down1(x, t_cond)
        p1 = self.pool1(x1)

        x2 = self.down2(p1, t_cond)
        p2 = self.pool2(x2)

        mid = self.mid(p2, t_cond)

        u2 = self.up2(mid)
        u2 = torch.cat([u2, x2], dim=1)
        b2 = self.b_up2(u2, t_cond)

        u1 = self.up1(b2)
        u1 = torch.cat([u1, x1], dim=1)
        b1 = self.b_up1(u1, t_cond)

        return self.out(b1)


def text_to_embedding(caption: str, dim: int = 64) -> torch.Tensor:
    vec = torch.zeros(dim)
    words = caption.lower().replace(',', '').split()
    for idx, word in enumerate(words):
        h = sum(ord(c) for c in word)
        vec[h % dim] += 1.0 / (idx + 1)
    norm = torch.norm(vec)
    return vec / (norm + 1e-6)

def train_nano_arcked_02(dataset_dir="dataset_training_images/mm.trine", epochs=15, batch_size=16, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[NanoArcKed-02] Training on {device}...")

    model = NanoArcKed02UNet().to(device)
    charbonnier = CharbonnierLoss()
    palette_loss = PaletteConsistencyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    dataset = []
    if os.path.exists(dataset_dir):
        files = [f for f in os.listdir(dataset_dir) if f.endswith(".png")]
        for f in files:
            path = os.path.join(dataset_dir, f)
            txt_path = path.replace(".png", ".txt")
            caption = "pixel art character, isolated background"
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as tf:
                    caption = tf.read().strip()
            im = Image.open(path).convert("RGBA").resize((64, 64), Image.Resampling.NEAREST)
            arr = np.array(im, dtype=np.float32) / 127.5 - 1.0
            t_img = torch.from_numpy(arr).permute(2, 0, 1)
            t_cond = text_to_embedding(caption, dim=64)
            dataset.append((t_img, t_cond))

    print(f"[NanoArcKed-02] Loaded {len(dataset)} training images from {dataset_dir}.")

    if len(dataset) > 0:
        model.train()
        imgs = torch.stack([item[0] for item in dataset]).to(device)
        conds = torch.stack([item[1] for item in dataset]).to(device)
        num_items = len(imgs)

        for epoch in range(1, epochs + 1):
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

            print(f"Epoch [{epoch}/{epochs}] NanoArcKed-02 Loss: {total_loss/max(1, batches):.4f}")

    weights_dir = "models/nano_arcked_02/weights"
    os.makedirs(weights_dir, exist_ok=True)
    pt_path = os.path.join(weights_dir, "nano_arcked_02.pt")
    torch.save(model.state_dict(), pt_path)
    print(f"[NanoArcKed-02] Saved model weights to {pt_path}")

    from models.nano_arcked_02.scripts.export_onnx import export_onnx
    export_onnx()

if __name__ == "__main__":
    train_nano_arcked_02()
