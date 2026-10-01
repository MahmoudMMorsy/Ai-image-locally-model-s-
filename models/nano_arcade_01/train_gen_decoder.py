#!/usr/bin/env python3
"""Train latent-only decoder for neural generation. CPU, resilient, resumes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from models.palette_ae import indices_to_images
from train_unet_palette import PaletteUNet


class LatentDecoder(nn.Module):
    """Decode z → palette logits without skip connections."""

    def __init__(self, n_colors=48, latent=128, base=48):
        super().__init__()
        self.fc = nn.Linear(latent, base * 4 * 8 * 8)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base * 4, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(base, n_colors, 3, 1, 1),
        )
        self._base = base

    def forward(self, z):
        h = self.fc(z).view(-1, self._base * 4, 8, 8)
        return self.net(h)


def main():
    out = Path("checkpoints")
    show = Path("/home/workdir/artifacts/show_results")
    show.mkdir(parents=True, exist_ok=True)

    IDX = torch.load("checkpoints/full_indices.pt", map_location="cpu", weights_only=False)
    pal = torch.load("checkpoints/palette48_full.pt", map_location="cpu", weights_only=False)
    ckpt = torch.load("checkpoints/palette_full_best.pt", map_location="cpu", weights_only=False)

    encoder = PaletteUNet(n_colors=48, latent=128, base=48)
    if "model" in ckpt:
        encoder.load_state_dict(ckpt["model"], strict=False)
    elif "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"], strict=False)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    SUB = min(1200, len(IDX))
    idx_sub = IDX[:SUB]

    bank_path = Path("checkpoints/latent_bank.pt")
    if bank_path.exists() and bank_path.stat().st_size > 1000:
        Z_full = torch.load(bank_path, map_location="cpu", weights_only=False)
        Z = Z_full[:SUB] if Z_full.size(0) >= SUB else Z_full
        SUB = Z.size(0)
        idx_sub = IDX[:SUB]
        print(f"loaded bank Z={Z.shape}", flush=True)
    else:
        print(f"building latent bank n={SUB}...", flush=True)
        zs = []
        with torch.no_grad():
            for i in range(0, SUB, 8):
                z = encoder.encode(idx_sub[i : i + 8])
                zs.append(z)
        Z = torch.cat(zs, 0)
        torch.save(Z, bank_path)
        print(f"Z={Z.shape}", flush=True)

    dec = LatentDecoder(n_colors=48, latent=128, base=48)
    start_ep = 1
    if "decoder" in ckpt:
        try:
            dec.load_state_dict(ckpt["decoder"])
            start_ep = int(ckpt.get("ep", 0)) + 1
            print(f"resumed decoder ep={ckpt.get('ep')} loss={ckpt.get('loss')}", flush=True)
        except Exception as e:
            print(f"fresh decoder: {e}", flush=True)

    opt = torch.optim.Adam(dec.parameters(), lr=8e-4)
    print(f"dec params={sum(p.numel() for p in dec.parameters())} start={start_ep}", flush=True)

    ds = TensorDataset(Z, idx_sub)
    loader = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)

    def save_gen(ep, avg_loss):
        dec.eval()
        with torch.no_grad():
            pairs = []
            for i in [0, 30, 60, 100, 150, 200]:
                if i >= SUB:
                    continue
                orig = indices_to_images(idx_sub[i : i + 1], pal)[0]
                pred = dec(Z[i : i + 1]).argmax(1)
                rec = indices_to_images(pred, pal)[0]
                o = (orig.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                r = (rec.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                pair = Image.new("RGBA", (192, 96), (20, 20, 20, 255))
                pair.paste(Image.fromarray(o, "RGBA").resize((96, 96), Image.NEAREST), (0, 0))
                pair.paste(Image.fromarray(r, "RGBA").resize((96, 96), Image.NEAREST), (96, 0))
                pairs.append(pair)
            if pairs:
                rg = Image.new("RGBA", (192, 96 * len(pairs)), (15, 15, 15, 255))
                for i, t in enumerate(pairs):
                    rg.paste(t, (0, i * 96))
                rg.save(show / "dec_recon_pair.png")
                rg.save(out / f"gen_dec_recon_ep{ep:03d}.png")

            gens = []
            for k in range(16):
                a, b = torch.randint(0, Z.size(0), (2,))
                alpha = 0.45 + 0.25 * torch.rand(1)
                z = alpha * Z[a] + (1 - alpha) * Z[b] + torch.randn_like(Z[0]) * 0.08
                pred = dec(z.unsqueeze(0)).argmax(1)
                img = indices_to_images(pred, pal)[0]
                arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                gens.append(Image.fromarray(arr, "RGBA").resize((96, 96), Image.NEAREST))
            gg = Image.new("RGBA", (4 * 96, 4 * 96), (15, 15, 15, 255))
            for i, t in enumerate(gens):
                gg.paste(t, ((i % 4) * 96, (i // 4) * 96))
            gg.save(show / "neural_gen_grid.png")
            gg.save(out / f"gen_dec_gen_ep{ep:03d}.png")
        print(f"  samples ep{ep} loss={avg_loss:.4f}", flush=True)

    max_ep = 40
    for ep in range(start_ep, max_ep + 1):
        dec.train()
        tot, n = 0.0, 0
        for zb, ib in loader:
            logits = dec(zb)
            loss = F.cross_entropy(logits, ib)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * zb.size(0)
            n += zb.size(0)
        avg = tot / max(n, 1)
        print(f"ep {ep:03d} loss={avg:.4f}", flush=True)
        if ep % 5 == 0 or ep == start_ep:
            model_state = ckpt.get("model", ckpt.get("encoder"))
            torch.save(
                {
                    "decoder": dec.state_dict(),
                    "encoder": model_state,
                    "model": model_state,
                    "n_colors": 48,
                    "latent": 128,
                    "base": 48,
                    "ep": ep,
                    "loss": avg,
                    "arch": "UNet+LatentDecoder",
                    "unet_loss": ckpt.get("unet_loss", 0.006),
                },
                "checkpoints/palette_full_best.pt",
            )
            torch.save({"decoder": dec.state_dict(), "ep": ep, "loss": avg}, out / f"latent_dec_ep{ep:03d}.pt")
            save_gen(ep, avg)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
