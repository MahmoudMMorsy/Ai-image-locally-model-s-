# Arcade Palette Pixel Generator (64x64)

Lightweight **CPU-only** generative model for arcade-style pixel art sprites.

## Status (v1 - Oct 2026)

| Mode | Status | Notes |
|------|--------|-------|
| HQ skip-mix (random) | Working | Nearest-neighbor latent mix + coverage filter |
| Hybrid neural | Working | SP + real skips blend |
| Image to Image | Working | Latent interpolation |
| Same-character animation | Working | High-similarity nearest poses |
| Pure neural (SP alone) | Weak | Blobs without bank |
| Text to Image (nano) | Not ready | Needs captions + more train |

**SkipPredictor:** ep82 · focal CE ≈ **0.174** · pixel acc ≈ **75%**  
**PaletteUNet recon:** CE ≈ 0.45

## Requirements

```
torch>=2.0
Pillow>=9.0
numpy>=1.20
```

CPU is enough. GPU optional for training only — weights stay CPU-compatible (`map_location="cpu"`).

## Checkpoints

Place weights under `arcade_palette/checkpoints/`:

| File | Role |
|------|------|
| `palette48_full.pt` | 48-color palette |
| `palette_full_best_0.45.pt` | PaletteUNet decoder |
| `skip_predictor_slim.pt` | SkipPredictor (neural) |
| `latent_bank_filt.pt` | Latent bank |
| `full_indices_u8.pt` | Palette index maps |
| `filter_meta.pt` | Filter metadata |

Use local packages: `ARCADE_CORE_A.zip` + `ARCADE_CORE_B.zip`

## Generate

```bash
cd arcade_palette
python scripts/generate_hq.py --ckpt_dir checkpoints --out outputs --mode both --n 16
```

## Taxonomy

- pixel_type: 24 · gender: 4 · class: 12 · weapon: 8 · species: 7
- Missing: undead, cleric, mob — add more data + captions for T2I
