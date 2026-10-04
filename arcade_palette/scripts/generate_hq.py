#!/usr/bin/env python3
"""HQ skip-mix + Hybrid generation for Arcade Palette model (CPU)."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

class PaletteUNet(nn.Module):
    def __init__(self, n_colors=48, latent=128, base=48):
        super().__init__()
        self.n_colors = n_colors
        self.emb = nn.Embedding(n_colors, 4)
        self.e1 = nn.Sequential(nn.Conv2d(4, base, 3, 1, 1), nn.ReLU(True), nn.Conv2d(base, base, 3, 1, 1), nn.ReLU(True))
        self.e2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(base, base*2, 3, 1, 1), nn.ReLU(True), nn.Conv2d(base*2, base*2, 3, 1, 1), nn.ReLU(True))
        self.e3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(base*2, base*4, 3, 1, 1), nn.ReLU(True), nn.Conv2d(base*4, base*4, 3, 1, 1), nn.ReLU(True))
        self.e4 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(base*4, base*4, 3, 1, 1), nn.ReLU(True))
        self.fc = nn.Linear(base*4*8*8, latent)
        self.fc_up = nn.Linear(latent, base*4*8*8)
        self.d3 = nn.Sequential(nn.ConvTranspose2d(base*8, base*4, 4, 2, 1), nn.ReLU(True), nn.Conv2d(base*4, base*4, 3, 1, 1), nn.ReLU(True))
        self.d2 = nn.Sequential(nn.ConvTranspose2d(base*6, base*2, 4, 2, 1), nn.ReLU(True), nn.Conv2d(base*2, base*2, 3, 1, 1), nn.ReLU(True))
        self.d1 = nn.Sequential(nn.ConvTranspose2d(base*3, base, 4, 2, 1), nn.ReLU(True), nn.Conv2d(base, base, 3, 1, 1), nn.ReLU(True))
        self.out = nn.Conv2d(base, n_colors, 1)

    def encode_skips(self, idx):
        x = self.emb(idx).permute(0, 3, 1, 2)
        x1 = self.e1(x); x2 = self.e2(x1); x3 = self.e3(x2); x4 = self.e4(x3)
        z = self.fc(x4.flatten(1))
        return z, {"x1": x1, "x2": x2, "x3": x3}

    def decode_logits(self, z, skips):
        h = self.fc_up(z).view(-1, 192, 8, 8)
        h = torch.cat([h, skips["x3"]], 1)
        h = self.d3(h)
        h = torch.cat([h, skips["x2"]], 1)
        h = self.d2(h)
        h = torch.cat([h, skips["x1"]], 1)
        h = self.d1(h)
        return self.out(h)

class SP(nn.Module):
    def __init__(self, latent=128):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent, 384), nn.ReLU(True), nn.Linear(384, 192*8*8))
        self.to_s3 = nn.Sequential(nn.ConvTranspose2d(192, 192, 4, 2, 1), nn.ReLU(True), nn.Conv2d(192, 192, 3, 1, 1), nn.ReLU(True))
        self.to_s2 = nn.Sequential(nn.ConvTranspose2d(192, 96, 4, 2, 1), nn.ReLU(True), nn.Conv2d(96, 96, 3, 1, 1), nn.ReLU(True))
        self.to_s1 = nn.Sequential(nn.ConvTranspose2d(96, 48, 4, 2, 1), nn.ReLU(True), nn.Conv2d(48, 48, 3, 1, 1), nn.ReLU(True))
    def forward(self, z):
        h = self.fc(z).view(-1, 192, 8, 8)
        x3 = self.to_s3(h); x2 = self.to_s2(x3); x1 = self.to_s1(x2)
        return {"x1": x1, "x2": x2, "x3": x3}

def indices_to_images(idx, palette):
    pal = palette.to(idx.device) / 255.0
    return F.embedding(idx, pal).permute(0, 3, 1, 2).contiguous()

def to_pil(idx_t, pal):
    img = indices_to_images(idx_t, pal)[0]
    return Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), "RGBA")

def coverage(idx):
    flat = idx.view(-1).cpu().numpy()
    vals, counts = np.unique(flat, return_counts=True)
    return float((flat != vals[counts.argmax()]).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--mode", choices=["hq", "hybrid", "both"], default="both")
    args = ap.parse_args()
    ck = Path(args.ckpt_dir)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    pal = torch.load(ck / "palette48_full.pt", map_location="cpu", weights_only=False)
    uck = torch.load(ck / "palette_full_best_0.45.pt", map_location="cpu", weights_only=False)
    meta = torch.load(ck / "filter_meta.pt", map_location="cpu", weights_only=False)
    Z = torch.load(ck / "latent_bank_filt.pt", map_location="cpu", weights_only=False)
    IDX = torch.load(ck / "full_indices_u8.pt", map_location="cpu", weights_only=False)
    sel = torch.cat([torch.arange(int(meta["n_orig"])), meta["keep_abs"].long()])
    IDX_f = IDX[sel].long()
    N = min(2200, Z.size(0)); Z = Z[:N]; IDX_f = IDX_f[:N]

    unet = PaletteUNet(n_colors=48, latent=128, base=48)
    sd = uck.get("encoder") or uck.get("model")
    unet.load_state_dict(sd, strict=False)
    unet.eval()

    sp = None
    sp_path = ck / "skip_predictor_slim.pt"
    if sp_path.exists():
        spck = torch.load(sp_path, map_location="cpu", weights_only=False)
        sp = SP(); sp.load_state_dict(spck["skip_predictor"], strict=False); sp.eval()
        print(f"SP ep={spck.get('ep')} loss={spck.get('loss')}")

    Zn = F.normalize(Z, dim=1)

    def encode_skips(idx):
        return unet.encode_skips(idx)

    if args.mode in ("hq", "both"):
        hq = []; tries = 0
        with torch.no_grad():
            while len(hq) < args.n and tries < 600:
                tries += 1
                a = torch.randint(0, N, (1,)).item()
                sims = (Zn[a:a+1] @ Zn.T).squeeze(0); sims[a] = -1
                topv, topi = sims.topk(min(18, N-1))
                valid = [int(topi[j]) for j in range(len(topv)) if float(topv[j]) >= 0.57]
                if len(valid) < 2: continue
                b = valid[torch.randint(0, min(7, len(valid)), (1,)).item()]
                alpha = 0.5 + 0.35 * torch.rand(1).item()
                za, sa = encode_skips(IDX_f[a:a+1]); zb, sb = encode_skips(IDX_f[b:b+1])
                z = alpha * za + (1 - alpha) * zb + torch.randn_like(za) * 0.01
                skips = {k: alpha * sa[k] + (1 - alpha) * sb[k] for k in sa}
                pred = unet.decode_logits(z, skips).argmax(1)
                if coverage(pred) < 0.15: continue
                hq.append(to_pil(pred, pal))
        for i, im in enumerate(hq):
            im.resize((256, 256), Image.NEAREST).save(out / f"hq_{i:02d}.png")
        print(f"HQ saved {len(hq)} -> {out}")

    if args.mode in ("hybrid", "both") and sp is not None:
        hyb = []
        with torch.no_grad():
            for _ in range(80):
                j = torch.randint(0, N, (1,)).item()
                z = Z[j:j+1] + torch.randn(1, 128) * 0.035
                sk_n = sp(z)
                nn_i = int((F.normalize(z, dim=1) @ Zn.T).squeeze(0).argmax())
                _, sk_r = encode_skips(IDX_f[nn_i:nn_i+1])
                a = 0.3 + 0.35 * torch.rand(1).item()
                skips = {k: a * sk_n[k] + (1 - a) * sk_r[k] for k in sk_n}
                pred = unet.decode_logits(z, skips).argmax(1)
                if coverage(pred) < 0.15: continue
                hyb.append(to_pil(pred, pal))
                if len(hyb) >= args.n: break
        for i, im in enumerate(hyb):
            im.resize((256, 256), Image.NEAREST).save(out / f"hybrid_{i:02d}.png")
        print(f"Hybrid saved {len(hyb)} -> {out}")

if __name__ == "__main__":
    main()
