#!/usr/bin/env python3
"""Phase 2: short masked-diffusion train on palette indices (CPU-friendly smoke → longer runs).

  python arcade_t2i/phase2_train_maskdiff.py --repo-root . --epochs 5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyMaskUNet(nn.Module):
    """Small discrete denoiser over 49 tokens (48 colors + MASK=48)."""

    def __init__(self, n_tokens: int = 49, dim: int = 64):
        super().__init__()
        self.emb = nn.Embedding(n_tokens, dim)
        self.t_emb = nn.Embedding(16, dim)
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, n_tokens, 1),
        )

    def forward(self, x_idx: torch.Tensor, t_id: torch.Tensor) -> torch.Tensor:
        # x_idx [B,H,W] long
        h = self.emb(x_idx).permute(0, 3, 1, 2)  # B,C,H,W
        te = self.t_emb(t_id).view(-1, h.size(1), 1, 1)
        return self.net(h + te)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-n", type=int, default=0, help="0 = all")
    args = ap.parse_args()
    root = args.repo_root.resolve()
    ckpt_dir = root / "arcade_t2i" / "checkpoints"
    idx_path = ckpt_dir / "indices_unified_u8.pt"
    if not idx_path.exists():
        idx_path = ckpt_dir / "indices_full_u8.pt"
    if not idx_path.exists():
        raise SystemExit("missing indices_*.pt — run phase1 first")

    data = torch.load(idx_path, map_location="cpu", weights_only=True)
    if args.max_n > 0:
        data = data[: args.max_n]
    n = data.shape[0]
    print(f"train N={n} shape={tuple(data.shape)}")
    MASK = 48
    model = TinyMaskUNet()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    t0 = time.time()
    last = 0.0
    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(n)
        total, steps = 0.0, 0
        for i in range(0, n, args.batch):
            b = data[perm[i : i + args.batch]].long()
            if b.numel() == 0:
                continue
            # random mask ratio
            ratio = torch.empty(b.size(0), 1, 1).uniform_(0.3, 0.9)
            mask = torch.rand_like(b.float()) < ratio
            x_in = b.clone()
            x_in[mask] = MASK
            t_id = (ratio.view(-1) * 15).long().clamp(0, 15)
            logits = model(x_in, t_id)
            loss = F.cross_entropy(logits.permute(0, 2, 3, 1)[mask], b[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        last = total / max(steps, 1)
        print(f"ep{ep} ce={last:.4f}")

    out = ckpt_dir / "maskdiff_unified_latest.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "ce": last,
            "n": n,
            "epochs": args.epochs,
            "time_s": time.time() - t0,
        },
        out,
    )
    meta = {"ce": last, "n": n, "epochs": args.epochs, "path": str(out)}
    (ckpt_dir / "maskdiff_unified_meta.json").write_text(json.dumps(meta, indent=2))
    print("saved", out, meta)


if __name__ == "__main__":
    main()
