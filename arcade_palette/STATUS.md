# Training status — 2026-10-05

Branch: `Meta.data` · folder: `arcade_palette/`

## Metrics
| Component | Value |
|-----------|-------|
| SkipPredictor | ep **85** · best focal CE **0.174** · acc ~75% |
| PaletteUNet | CE ~0.45 |
| Device | CPU (weights CPU-compatible) |

## Working
- HQ skip-mix
- Hybrid neural
- Image→Image
- Same-character animation

## Not ready
- Pure neural (SP alone)
- Text→Image (nano)

## Checkpoints (local / Release)
Place under `arcade_palette/checkpoints/`:
- palette48_full.pt
- palette_full_best_0.45.pt
- skip_predictor_slim.pt
- latent_bank_filt.pt
- full_indices_u8.pt
- filter_meta.pt

Packages: ARCADE_CORE_A.zip + ARCADE_CORE_B.zip

## Last train log
```
SP ep 083 focal=0.1863 acc=74.9%
SP ep 084 focal=0.2061 acc=71.4%
SP ep 085 focal=0.1770 acc=74.2%
BEST=0.1740
```

## Generate
```bash
python arcade_palette/scripts/generate_hq.py --ckpt_dir arcade_palette/checkpoints --out out --mode both
```
