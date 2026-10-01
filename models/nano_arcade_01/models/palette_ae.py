"""
Palette-based autoencoder for sharp pixel art.
1) Extract global palette from dataset
2) Encode images as palette indices
3) Tiny network predicts index maps
4) Decode via palette lookup → always sharp
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path


def extract_palette(image_paths, n_colors=32, size=64):
    """K-means style palette from a sample of images (numpy only)."""
    pixels = []
    for p in image_paths[: min(200, len(image_paths))]:
        img = Image.open(p).convert("RGBA").resize((size, size), Image.NEAREST)
        arr = np.asarray(img).reshape(-1, 4).astype(np.float32)
        # keep non-transparent
        mask = arr[:, 3] > 10
        if mask.any():
            pixels.append(arr[mask])
    pix = np.concatenate(pixels, 0)
    # simple reservoir + iterative kmeans
    rng = np.random.RandomState(42)
    if len(pix) > 50000:
        pix = pix[rng.choice(len(pix), 50000, replace=False)]
    # init centers
    idx = rng.choice(len(pix), n_colors, replace=False)
    centers = pix[idx].copy()
    for _ in range(15):
        # assign
        d = ((pix[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        labels = d.argmin(1)
        for k in range(n_colors):
            m = labels == k
            if m.any():
                centers[k] = pix[m].mean(0)
    centers[:, 3] = np.where(centers[:, 3] < 128, 0, 255)
    return centers.astype(np.float32)  # (K, 4)


def images_to_indices(imgs: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """imgs (B,4,H,W) in [0,1], palette (K,4) in [0,255] → indices (B,H,W)"""
    B, _, H, W = imgs.shape
    flat = (imgs.permute(0, 2, 3, 1).reshape(-1, 4) * 255.0)  # N 4
    pal = palette.to(imgs.device)  # K 4
    d = ((flat[:, None, :] - pal[None, :, :]) ** 2).sum(-1)  # N K
    idx = d.argmin(1).view(B, H, W)
    return idx


def indices_to_images(idx: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """idx (B,H,W), palette (K,4) → (B,4,H,W) in [0,1]"""
    pal = palette.to(idx.device) / 255.0
    return F.embedding(idx, pal).permute(0, 3, 1, 2).contiguous()


class PaletteNet(nn.Module):
    """
    Predicts palette-index logits from a continuous latent.
    Or acts as autoencoder on index maps via embedding.
    """

    def __init__(self, n_colors=32, latent=64, base=32):
        super().__init__()
        self.n_colors = n_colors
        self.emb = nn.Embedding(n_colors, 16)
        # encoder on embedded indices
        self.enc = nn.Sequential(
            nn.Conv2d(16, base, 4, 2, 1), nn.ReLU(True),       # 32
            nn.Conv2d(base, base * 2, 4, 2, 1), nn.ReLU(True),  # 16
            nn.Conv2d(base * 2, base * 2, 4, 2, 1), nn.ReLU(True),  # 8
            nn.Flatten(),
            nn.Linear(base * 2 * 8 * 8, latent),
        )
        self.dec_fc = nn.Linear(latent, base * 2 * 8 * 8)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base * 2, base * 2, 4, 2, 1), nn.ReLU(True),  # 16
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1), nn.ReLU(True),      # 32
            nn.ConvTranspose2d(base, n_colors, 4, 2, 1),  # 64 logits
        )
        self._base = base
        self.latent = latent

    def encode(self, idx):
        x = self.emb(idx).permute(0, 3, 1, 2)  # B 16 H W
        return self.enc(x)

    def decode_logits(self, z):
        h = self.dec_fc(z).view(-1, self._base * 2, 8, 8)
        return self.dec(h)  # B K H W

    def forward(self, idx):
        z = self.encode(idx)
        logits = self.decode_logits(z)
        return logits, z

    @torch.no_grad()
    def sample(self, z_bank, n=1, noise=0.4, device=None):
        device = device or z_bank.device
        i1 = torch.randint(0, z_bank.size(0), (n,), device=device)
        i2 = torch.randint(0, z_bank.size(0), (n,), device=device)
        a = torch.rand(n, 1, device=device) * 0.5 + 0.25
        z = a * z_bank[i1] + (1 - a) * z_bank[i2]
        z = z + torch.randn_like(z) * noise
        logits = self.decode_logits(z)
        idx = logits.argmax(1)
        return idx
