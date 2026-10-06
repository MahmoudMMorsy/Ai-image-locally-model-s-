# AGENTS.md — instructions for Jules and other coding agents

Read this file before any training, generation, or dataset change.

## Goal

Build a real 64x64 arcade pixel-art generator that can produce a **new character that is not a copy of a training image**, at quality close to the training sprites.

This is neural generation from the images (palette-index discrete diffusion / VQ prior), **not** procedural body-part stitching and **not** nearest-neighbor remix of the latent bank as the final product.

## Dataset rule (mandatory)

DO NOT train on `dataset_clean/` or `dataset_clean.zip` alone.
That set has only **~101 images**. Daily pipeline logs that say "Verified 101 cleaned dataset images" are the wrong data source for the quality target.

Use **all** real sprites:

| Source | Approx count | Use |
|--------|----------------|-----|
| `mm.trine/` | **~5104 PNG**, mostly 64x64, **111 characters**, each with multiple poses | PRIMARY training set |
| `examples/` | ~199 | optional extra, after filter to 64x64 arcade style |
| `All-data-Trine/` | ~22 | optional |
| `nano arcked dat/` | ~4 real PNGs | ignore for training |
| `dataset_clean.zip` | 101 | do **not** use as the only set |

Filter rules:
- Keep 64x64 (or nearest-resize with NEAREST, never bilinear).
- Drop near-empty frames (transparent coverage too low).
- Group frames by character hash in the filename (`NNNNN__<hash>__...`) so animation stays on the **same character**.
- Do not train only on augmentations of one character.

## What is already working (do not replace with stubs)

- Palette of 48 colors + index reconstruction is the sharpness path.
- HQ skip-mix from a latent bank produces clear characters **inside the training distribution**. It is a baseline, not the from-scratch model.
- Pure neural samples from a few epochs of discrete diffusion are still noise. That path must be trained longer on the full set.

## Correct model path

1. Quantize every kept sprite to the shared 48-color palette. Train on **indices**, not RGB regression.
2. Discrete masked diffusion (or VQ-VAE + prior) over 64x64 palette indices.
   Reference sketch: `nano_pixel_mob/2026-09-12/2_discrete_diffusion/`.
3. Conditioning later: character tags from filename groups (pose / weapon) then short text. Text-to-image is not ready until captions exist.
4. Inference must start from a full mask / noise tokens and unmask. No nearest-neighbor lookup at sample time for the "from scratch" checkpoint.
5. Acceptance: samples are coherent 64x64 characters, not copies of a single training file, coverage not a flat color, and a held-out character hash is not reproduced pixel-identical.

## Do not

- Do not log "trained" after 10 steps on `torch.randn`.
- Do not treat `models/nano_pixel_3A_xl` 3-conv stub trained on random tensors as the production model.
- Do not shrink the dataset to 101 images to make a daily job finish.
- Do not generate animation by morphing two different characters. Animation = pose sequence of one hash.

## Collaboration

- Plan: `PLAN_FROM_SCRATCH.md`
- Issues labeled `jules` are tasks for Jules (Google Jules).
- Comment on the issue if data counts disagree; do not silently switch back to the 101-image set.
