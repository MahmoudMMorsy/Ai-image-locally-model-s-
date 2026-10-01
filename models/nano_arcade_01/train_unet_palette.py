#!/usr/bin/env python3
"""U-Net style palette AE — strong reconstruction on user data, CPU only."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from models.palette_ae import indices_to_images


class PaletteUNet(nn.Module):
    """Encode index map → latent → decode with skips for sharp recon."""

    def __init__(self, n_colors: int = 48, latent: int = 128, base: int = 48):
        super().__init__()
        self.n_colors = n_colors
        self.latent = latent
        self.emb = nn.Embedding(n_colors, 16)
        # encoder
        self.e1 = nn.Sequential(nn.Conv2d(16, base, 3, 1, 1), nn.ReLU(True))  # 64
        self.e2 = nn.Sequential(nn.Conv2d(base, base * 2, 4, 2, 1), nn.ReLU(True))  # 32
        self.e3 = nn.Sequential(nn.Conv2d(base * 2, base * 4, 4, 2, 1), nn.ReLU(True))  # 16
        self.e4 = nn.Sequential(nn.Conv2d(base * 4, base * 4, 4, 2, 1), nn.ReLU(True))  # 8
        self.fc = nn.Linear(base * 4 * 8 * 8, latent)
        self.fc_up = nn.Linear(latent, base * 4 * 8 * 8)
        # decoder with skips
        self.d4 = nn.Sequential(nn.ConvTranspose2d(base * 4, base * 4, 4, 2, 1), nn.ReLU(True))  # 16
        self.d3 = nn.Sequential(nn.ConvTranspose2d(base * 8, base * 2, 4, 2, 1), nn.ReLU(True))  # 32
        self.d2 = nn.Sequential(nn.ConvTranspose2d(base * 4, base, 4, 2, 1), nn.ReLU(True))  # 64
        self.out = nn.Conv2d(base * 2, n_colors, 3, 1, 1)
        self._base = base

    def encode(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.emb(idx).permute(0, 3, 1, 2)
        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        z = self.fc(x4.flatten(1))
        return z

    def decode_logits(self, z: torch.Tensor, skips=None) -> torch.Tensor:
        """If skips is None, decode from latent only (generation)."""
        b = self._base
        h = self.fc_up(z).view(-1, b * 4, 8, 8)
        h = self.d4(h)  # 16
        if skips is not None:
            h = torch.cat([h, skips["x3"]], 1)
        else:
            h = torch.cat([h, h], 1)  # fake skip
        h = self.d3(h)  # 32
        if skips is not None:
            h = torch.cat([h, skips["x2"]], 1)
        else:
            h = torch.cat([h, h], 1)
        h = self.d2(h)  # 64
        if skips is not None:
            h = torch.cat([h, skips["x1"]], 1)
        else:
            h = torch.cat([h, h], 1)
        return self.out(h)

    def forward(self, idx: torch.Tensor):
        x = self.emb(idx).permute(0, 3, 1, 2)
        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        z = self.fc(x4.flatten(1))
        skips = {"x1": x1, "x2": x2, "x3": x3}
        logits = self.decode_logits(z, skips)
        return logits, z

    def reconstruct(self, idx: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(idx)
        return logits.argmax(1)


def save_samples(model, IDX, pal, ep: int, out_dir: Path, show: Path):
    model.eval()
    with torch.no_grad():
        # recon pairs
        pairs = []
        for i in [0, 20, 50, 100, 150, 250]:
            if i >= len(IDX):
                break
            orig = indices_to_images(IDX[i : i + 1], pal)[0]
            pred = model.reconstruct(IDX[i : i + 1])
            rec = indices_to_images(pred, pal)[0]
            o = (orig.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            r = (rec.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pair = Image.new("RGBA", (192, 96), (20, 20, 20, 255))
            pair.paste(Image.fromarray(o, "RGBA").resize((96, 96), Image.NEAREST), (0, 0))
            pair.paste(Image.fromarray(r, "RGBA").resize((96, 96), Image.NEAREST), (96, 0))
            pairs.append(pair)
        rg = Image.new("RGBA", (192, 96 * len(pairs)), (15, 15, 15, 255))
        for i, t in enumerate(pairs):
            rg.paste(t, (0, i * 96))
        rg.save(out_dir / f"unet_recon_ep{ep:03d}.png")
        rg.save(show / "palette_recon_pair.png")

        # generation via latent mix
        Z = model.encode(IDX[: min(500, len(IDX))])
        gens = []
        for k in range(16):
            a, b = torch.randint(0, Z.size(0), (2,))
            alpha = 0.4 + 0.2 * torch.rand(1)
            z = alpha * Z[a] + (1 - alpha) * Z[b] + torch.randn_like(Z[0]) * 0.15
            logits = model.decode_logits(z.unsqueeze(0), skips=None)
            pred = logits.argmax(1)
            img = indices_to_images(pred, pal)[0]
            arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            gens.append(Image.fromarray(arr, "RGBA").resize((96, 96), Image.NEAREST))
        gg = Image.new("RGBA", (4 * 96, 4 * 96), (15, 15, 15, 255))
        for i, t in enumerate(gens):
            gg.paste(t, ((i % 4) * 96, (i // 4) * 96))
        gg.save(out_dir / f"unet_gen_ep{ep:03d}.png")
        gg.save(show / "palette_full_grid.png")
    print(f"  samples ep{ep} saved", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--start", type=int, default=1)
    args = ap.parse_args()

    out = Path("checkpoints")
    show = Path("/home/workdir/artifacts/show_results")
    show.mkdir(parents=True, exist_ok=True)

    IDX = torch.load("checkpoints/full_indices.pt", map_location="cpu", weights_only=False)
    pal = torch.load("checkpoints/palette48_full.pt", map_location="cpu", weights_only=False)
    print(f"IDX={IDX.shape}", flush=True)

    model = PaletteUNet(n_colors=48, latent=128, base=48)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"params={nparams}", flush=True)

    ckpt_path = out / "palette_full_best.pt"
    if ckpt_path.exists():
        try:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if ck.get("arch") == "PaletteUNet":
                model.load_state_dict(ck["model"])
                args.start = int(ck.get("ep", 0)) + 1
                print(f"resumed ep {ck.get('ep')}", flush=True)
        except Exception as e:
            print(f"no resume: {e}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(TensorDataset(IDX), batch_size=args.batch, shuffle=True, drop_last=True)

    for ep in range(args.start, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for (batch,) in loader:
            logits, _ = model(batch)
            loss = F.cross_entropy(logits, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * batch.size(0)
            n += batch.size(0)
        avg = tot / max(n, 1)
        print(f"ep {ep:03d} loss={avg:.4f}", flush=True)

        if ep % 5 == 0 or ep == args.start:
            torch.save(
                {
                    "model": model.state_dict(),
                    "n_colors": 48,
                    "latent": 128,
                    "base": 48,
                    "ep": ep,
                    "loss": avg,
                    "arch": "PaletteUNet",
                    "params": nparams,
                },
                ckpt_path,
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "n_colors": 48,
                    "latent": 128,
                    "base": 48,
                    "ep": ep,
                    "arch": "PaletteUNet",
                },
                out / f"unet_ep{ep:03d}.pt",
            )
            save_samples(model, IDX, pal, ep, out, show)

    print("TRAIN DONE", flush=True)


if __name__ == "__main__":
    main()
