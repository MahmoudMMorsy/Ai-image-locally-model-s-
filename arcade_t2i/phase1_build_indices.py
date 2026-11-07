#!/usr/bin/env python3
"""Phase 1: quantize manifest images to palette48 indices [N,64,64] uint8.

Requires local checkout of image files listed in manifest.

  python arcade_t2i/phase1_build_indices.py --repo-root .
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ARCADE_PALETTE_48 = [
    [0, 0, 0], [15, 15, 25], [30, 30, 45], [50, 50, 70],
    [80, 80, 100], [120, 120, 140], [170, 170, 190], [220, 220, 240], [255, 255, 255],
    [120, 0, 0], [180, 20, 20], [230, 50, 50], [255, 100, 100],
    [100, 40, 0], [170, 70, 0], [230, 120, 20], [255, 170, 50], [255, 220, 100],
    [30, 80, 0], [60, 140, 10], [100, 200, 30], [160, 240, 70],
    [0, 60, 80], [0, 120, 140], [20, 180, 200], [80, 230, 245],
    [20, 20, 100], [50, 50, 180], [100, 100, 230], [160, 160, 255],
    [80, 10, 100], [140, 30, 160], [200, 70, 220], [240, 140, 255],
    [100, 20, 60], [170, 40, 100], [230, 90, 150], [255, 160, 200],
    [60, 40, 20], [110, 75, 40], [160, 110, 60], [210, 160, 100],
    [40, 20, 10], [80, 50, 20], [130, 90, 50], [180, 140, 90], [220, 185, 140], [245, 215, 180],
]


def quantize_rgba(img: Image.Image, palette: torch.Tensor) -> torch.Tensor:
    img = img.convert("RGBA").resize((64, 64), Image.Resampling.NEAREST)
    a = np.asarray(img, dtype=np.float32)
    rgb = torch.from_numpy(a[:, :, :3])
    alpha = torch.from_numpy(a[:, :, 3])
    diff = rgb.unsqueeze(2) - palette.view(1, 1, 48, 3)
    idx = (diff ** 2).sum(-1).argmin(-1).to(torch.uint8)
    idx[alpha < 128] = 0
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="default arcade_t2i/manifests/manifest_unified.jsonl",
    )
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()
    root = args.repo_root.resolve()
    man = args.manifest or (root / "arcade_t2i" / "manifests" / "manifest_unified.jsonl")
    if not man.exists():
        # fallback to older full manifest
        alt = root / "arcade_t2i" / "manifests" / "manifest_full.jsonl"
        if alt.exists():
            man = alt
        else:
            raise SystemExit(f"manifest not found: {man}")

    rows = [json.loads(ln) for ln in man.read_text(encoding="utf-8").splitlines() if ln.strip()]
    print(f"manifest lines: {len(rows)}")
    palette = torch.tensor(ARCADE_PALETTE_48, dtype=torch.float32)
    ckpt_dir = root / "arcade_t2i" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    indices = []
    kept_rows = []
    fails = 0
    for i, r in enumerate(rows):
        p = root / r["path"]
        if not p.exists():
            fails += 1
            continue
        try:
            with Image.open(p) as im:
                indices.append(quantize_rgba(im, palette).numpy())
                kept_rows.append(r)
        except Exception:
            fails += 1
            continue
        if (i + 1) % args.batch == 0:
            print(f"  quantized {len(indices)} / scanned {i+1}")

    if len(indices) < 1000:
        print(f"HARD STOP: kept={len(indices)} fails={fails}")
        raise SystemExit(2)

    t = torch.from_numpy(np.stack(indices, axis=0))
    torch.save(t, ckpt_dir / "indices_unified_u8.pt")
    torch.save(palette / 255.0, ckpt_dir / "palette48.pt")
    kept_path = root / "arcade_t2i" / "manifests" / "manifest_unified_kept.jsonl"
    with kept_path.open("w", encoding="utf-8") as f:
        for r in kept_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"N": int(t.shape[0]), "fails": fails, "shape": list(t.shape)}
    (ckpt_dir / "indices_unified_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print("saved", ckpt_dir / "indices_unified_u8.pt")


if __name__ == "__main__":
    main()
