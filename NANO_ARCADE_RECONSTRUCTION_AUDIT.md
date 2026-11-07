# Nano Arcade Reconstruction Audit

## Execution date

2026-11-07

## What was tested

1. Existing `nano_pixel_mob/2026-09-12/3_vqvae_prior/vqvae_weights.pt` on its bundled dataset.
2. Existing latent prior on 16 deterministic seeds.
3. A replacement dataset built from isolated RGBA examples.
4. A 300-epoch CPU VQ-VAE reconstruction run on the replacement dataset.

## Finding 1 — bundled dataset was not diverse

The bundled file `dataset/arcade_sprites_32x32.pt` reports 1,000 records, but inspection found only **4 unique exact images** and 4 unique palette rows. Therefore the earlier 100% reconstruction score was not evidence of generalization. It was memorization of a tiny repeated set.

## Finding 2 — latent prior failed the random-generation gate

The supplied `prior_weights.pt` was sampled with seeds 0–15. The output grid showed essentially one repeated red/blue character with minor changes. Metrics also showed a fixed 8-color output pattern across all samples. The prior is rejected as a production generator.

Evidence:

- `/home/ubuntu/nano_arcade/vqvae_prior_random16.png`
- `/home/ubuntu/nano_arcade/vqvae_prior_random16.json`

## Finding 3 — replacement RGBA dataset exposed a decoder/data problem

A builder was added at:

```text
nano_pixel_mob/2026-09-12/build_rgba_dataset.py
```

It collected 96 RGBA examples with 94 unique indexed images, normalized with nearest-neighbor resize, and reserved index 0 for transparency. A palette-index alignment bug was fixed so palette index 0 is transparent and visible colors begin at index 1.

Two CPU VQ-VAE runs were performed:

| Run | Epochs | Validation accuracy | Minimum sample accuracy | Gate |
|---|---:|---:|---:|---|
| initial replacement run | 60 | 0.7070 | 0.0078 | FAIL |
| extended run, before palette correction | 300 | 0.7533 | 0.4648 | FAIL |
| corrected palette run | 300 | 0.7780 | 0.5010 | FAIL |

Evidence for the final corrected run:

- `/home/ubuntu/nano_arcade/vqvae_rgba_examples_fixed_gate.png`
- `/home/ubuntu/nano_arcade/vqvae_rgba_examples_fixed_gate.json`
- `/home/ubuntu/nano_arcade/vqvae_rgba_examples_fixed.pt`

The visual grid still contains sparse/noisy inputs and incomplete reconstructions. The checkpoint is **not accepted** for prior training.

## Decision

Do not train the spatial prior yet. The correct next stage is dataset repair:

1. Apply the strict foreground occupancy and connected-component gates to every candidate.
2. Reject sparse or near-empty crops and any image with more than one subject.
3. Preserve source alpha and use a single global palette/index contract for the training run.
4. Build a clean train/validation split from genuinely readable 32×32 or 64×64 characters.
5. Repeat reconstruction until the visual gate passes; only then train the prior.

No production-quality model is claimed by this audit. The failed samples are retained as evidence so future iterations do not mistake memorization or a saved PNG for successful generation.
