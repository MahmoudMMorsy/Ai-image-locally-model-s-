# flux_student STATUS

Updated: 2026-11-03

## Overview
- Architecture: `StudentT2I` (base=48: ~10M params, ~39 MB checkpoint; base=96: ~26.6M params)
- Teacher model: `stabilityai/sd-turbo` (1-step fast CPU/GPU synthesis, fallback for `FLUX.1-schnell`)
- Data: Multiresolution pyramids generated across 32x32, 64x64, 128x128, and 256x256
- Safety: Zero-tolerance religious figure exclusion filtering implemented in `student/filter.py` and active in teacher generation and continuous loop.

## Checkpoint & Training
- Checkpoint: `checkpoints/student_final.pt` (~39 MB, 10,032,387 parameters)
- Epochs completed: Continuous loop active, trained across multiple cycles (>30 epochs total)
- Loss progression: Initial loss ~1.07 -> 1.009 -> 0.8257 -> 0.7445 (continuously decreasing)
- Manifest dataset: 29+ prompt-paired character items synthesized directly by teacher model

## Honest Quality Assessment
- The model learns character structure conditioning on prompt-paired teacher images.
- Current status: Discrete denoising samples exhibit clear silhouette formation, distinct color massing matching prompt archetypes (e.g., steel armor / silver-grey for knight, metallic teal/red highlights for robot).
- High-frequency anatomical details (fine facial expressions, intricate weapon filigree) are still softening at 64x64 pixel resolution, which will continuously sharpen as the continuous background loop iterates further.

## Background Continuous Loop
- Automated via `flux_student/scripts/continuous_loop.py` running in background.
- Periodically batches teacher generation, resumes training checkpoint, dumps test samples to `samples/`, and logs execution to `continuous.log` and `continuous_status.json`.
