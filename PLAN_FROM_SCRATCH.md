# Plan: from-scratch 64x64 arcade character generator

Owner intent: a character that does **not** already exist in the set, generated from the images, at quality close to trained sprites.

Jules (Google Jules, https://jules.google) should follow this file. Assign work by labeling the tracking issue `jules`.

## Status (honest)

| Capability | Status |
|------------|--------|
| Clear characters inside the training style (HQ skip-mix) | works |
| Same-character pose sequence from `mm.trine` groups | data exists; filter tiny frames |
| New character from pure neural sampling | **not ready** — short discrete-diffusion runs still look like noise |
| Free text-to-image | not ready — no real captions |

## Phase 0 — data (blocking)

1. Index every PNG under `mm.trine/` (~5104 files).
2. Parse character id from `NNNNN__<12hex>__...`.
3. Keep frames that are 64x64 or can be NEAREST-resized to 64x64, with enough opaque pixels.
4. Write `data/manifest_full.json`: path, character_id, pose tag, width, height, opaque_ratio.
5. Report counts in the issue. Expected: thousands of frames, ~111 characters, not 101.

`dataset_clean.zip` (101 images) is a smoke subset only. Production training must not use it as the sole set.

## Phase 1 — palette indices

1. Fit or reuse a 48-color palette on the full manifest.
2. Save `data/indices_u8.pt` shaped `[N,64,64]` and `data/palette48.pt`.
3. Reconstruction check: index → palette → PNG must look like the source (nearest).

## Phase 2 — discrete masked diffusion (the from-scratch model)

Architecture direction (CPU-friendly, can grow):

- Embedding over 49 tokens (48 colors + MASK).
- Small conv U-Net, time embedding.
- Loss: cross-entropy on masked pixels (optionally light weight on unmasked).
- Train on **all** manifest indices, class-balanced by character_id so one hash does not dominate.

Sampling:

- Start from all MASK.
- Unmask high-confidence tokens over T steps.
- No latent-bank nearest neighbor in this sampler.

Stop condition for a checkpoint worth showing:

- masked CE clearly below early-epoch noise (~2.3 on 48-way is chance-ish; aim well under 1.0 on masked pixels).
- samples use many palette colors and a readable silhouette.
- not a pixel copy of one training image.

Existing sketch: `nano_pixel_mob/2026-09-12/2_discrete_diffusion/discrete_diffusion.py`.
Existing weights under `models/nano_pixel_*` were mostly fit on the 101-image loop or random tensors. Do not present them as the full-data model.

## Phase 3 — same-character animation

- Sample or retrieve a character embedding / hash.
- Generate or select only frames of that hash (or conditioned on that id).
- Export GIF. Reject sequences that change identity.

## Phase 4 — text (after 2 works)

- Captions from character groups: class, weapon, pose — not generic "arcade pixel character".
- Condition the diffusion on a small text embedding.
- Only then claim text-to-image.

## Phase 5 — package

- Weights + palette + `generate.py` that runs on CPU.
- Sample grid committed under `examples/from_scratch/`.
- Do not upload multi-hundred-MB binaries via text push; use GitHub Releases for large `.pt`.

## Coordination

- Grok maintains this plan and the data-count constraint.
- Jules implements phases on a branch and opens a PR.
- If a run uses fewer than 1000 images, stop and comment. That is the wrong dataset.
