# Plan: from-scratch arcade generator (2026-10-06)

Owner intent: generate a character that is not a copy of a training image, at the quality of the trained sprites, using the images themselves.

## Data (use all of it)

| Path | Images | Use |
|------|--------|-----|
| `mm.trine/` | ~5109 PNG, ~111 characters, multi-pose | PRIMARY. Required. |
| `examples/` | ~199 | optional extra |
| `dataset_clean.zip` | 101 | smoke test only. Forbidden as main set. |
| `dataset_raw.zip` | 10 | ignore for product training |
| `nano arcked dat/` | ~4 real PNGs | do not treat as the dataset |

## Stages

1. Index every `mm.trine` PNG to 64x64 and a shared 48-color palette. Save `indices` + palette. Dedup exact hashes but keep pose variants (different frame of same character id).
2. Train discrete masked diffusion on those indices. Architecture can start from `nano_pixel_mob/.../discrete_diffusion.py` but must be 64x64 and trained on the full index set.
3. Sample from a full mask. Save grids under `examples/from_scratch/`. Reject a run if mean unique colors per image is under 6.
4. Only after step 3 looks like characters: add captions from filename groups (character hash, pose tag) and a small conditioning vector. Not before.
5. Animation: group by the hash in `NNNNN__HASH__...png` and sample pose sequences of the same hash.

## Current honest status

- HQ skip-mix on a ~3500 latent bank produces recognizable in-distribution characters. That is retrieval-style blend, not from-scratch.
- Discrete diffusion training was started and is not yet at character quality (early samples are sparse noise). Continue training; do not publish those as success.
- Repo models under `models/nano_pixel_*` and `models/real_diffusion_onnx` were fine-tuned on the 101-image clean set. Do not use them as the product generator until they are retrained on `mm.trine`.

## For Jules

If you pick this up: implement stage 1 and stage 2, print the image count, and open a PR. Do not switch the data root back to `dataset_clean`.
