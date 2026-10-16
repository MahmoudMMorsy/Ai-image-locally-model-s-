# arcade_t2i — Text-to-Image (64×64 arcade sprites)

## Goal
Readable AR/EN text→image for classes: knight, mage, robot, cat, dragon, …
Quality gate: samples must be recognizable characters (not noise/blobs). **NOT PASSED yet.**

## Layout
- `model_t2i.py` — TextCVAE
- `train_strict.py` — full-N=15755 strict training
- `text_lang.py` + `bilingual_vocab.json` — AR/EN
- `checkpoints/`
  - `t2i_cvae_strict_best.pt` — CVAE best (ep~13, loss~0.45, recon_acc~0.82)
  - `maskdiff_best.pt` — discrete mask diffusion (ep~6, loss~0.44)
  - `indices_full_u8.pt` — 15755×64×64 palette indices
  - `palette48.pt`, `latent_bank.pt`
- `samples/` — eval grids (STRICT_*, MASKDIFF_*)
- `PLAN_STRICT.md` — acceptance criteria

## Status (honest)
| Model | Metric | Value | Visual gate |
|-------|--------|-------|-------------|
| Jules old | loss | 2.44 | FAIL (noise) |
| TextCVAE strict | loss / recon_acc | ~0.45 / ~0.82 | FAIL (blobs) |
| MaskDiff | loss | ~0.44 | FAIL (not characters yet) |

Train on full manifest N=15755. Class-balanced sampling used for minority classes (knight, etc.).

## Next
Continue MaskDiff + CVAE until generation is readable. Push every checkpoint/sample update here.
