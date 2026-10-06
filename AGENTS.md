# AGENTS.md — instructions for Jules and any coding agent

Read this before any training or generation task.

## Goal

Build a real 64x64 arcade pixel-art generator that can sample a character **from scratch** (pure neural, no nearest-neighbor copy of a training sprite) at a quality close to the training images.

## Mandatory data rule

DO NOT train on `dataset_clean.zip` (101 images) or `dataset_raw.zip` (10 images) as the primary set.

Primary data is **`mm.trine/`**: about **5100 PNG files, mostly 64x64, about 111 characters, each with multiple poses**. Also use `examples/` (~199) only as extra, never as a replacement.

Every training script must:
1. Count images before training and print the count.
2. Abort if the count is under 1000 for the main arcade model.
3. Log the data root path in the commit / training log.

The daily pipeline that says "Verified 101 cleaned dataset images" is the wrong path for the product model. Keep those small runs only for smoke tests.

## Correct model path

1. Quantize every sprite to a shared 48-color palette (PaletteNet / index map). Sharpness comes from predicting color indices, not RGB regression.
2. Train a **discrete masked diffusion** (or a strong palette-index UNet) on **all** index maps from `mm.trine`.
3. Inference must start from a full mask token grid and unmask. No skip-mix, no copying a neighbor latent, for the "from scratch" path.
4. Skip-mix / HQ neighbor blend is allowed only as a separate production fallback, and must be labeled as such in showcases.
5. Same-character animation = frames of one character id (filename hash in `mm.trine`), not a morph between two different characters.

Reference code already in the repo: `nano_pixel_mob/2026-09-12/2_discrete_diffusion/discrete_diffusion.py`. Extend it to 64x64 and 48 colors and train it on `mm.trine`, do not leave it as a 32x32 demo on a tiny set.

## What is not the goal

- Do not claim text-to-image until captions exist for the full `mm.trine` set.
- Do not ship showcases that are procedural body-part stitching.
- Do not overwrite `mm.trine` images.

## Done when

- Training log shows N >= 4000 images from `mm.trine`.
- A sample grid is generated from a full mask (no bank lookup) and a human can recognize characters.
- `PLAN_FROM_SCRATCH.md` checklist is updated with the real epoch and loss.
