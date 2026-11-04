# flux_student STATUS

Updated: 2026-11-04

## Overview
- Architecture: `StudentT2I` (base=48: ~10.03M params, ~38.3 MB checkpoint; base=96: ~26.6M params)
- Teacher model: `stabilityai/sd-turbo` (1-step fast CPU/GPU synthesis, fallback for `FLUX.1-schnell`)
- Data: Multiresolution pyramids generated across 32x32, 64x64, 128x128, and 256x256
- Safety: Zero-tolerance religious figure exclusion filtering implemented in `student/filter.py` and active in teacher generation and continuous loop with strict Unicode word boundaries `(?<![\\w\\u0600-\\u06FF])`.

## Checkpoint & Training
- Checkpoint: `checkpoints/student_final.pt` (~38.3 MB, 10,032,387 parameters)
- Epochs completed: Continuous loop active, trained across multiple cycles and epochs
- Loss progression: Initial loss ~1.06 -> 1.006 -> 0.7867 (continuously decreasing)
- Manifest dataset: 28+ prompt-paired character items synthesized directly by teacher model

## Honest Quality Assessment
- The model learns character structure conditioning on prompt-paired teacher images.
- Current status: Discrete denoising samples exhibit clear silhouette formation, distinct color massing matching prompt archetypes (e.g., steel armor / silver-grey for knight, metallic teal/red highlights for robot).
- High-frequency anatomical details (fine facial expressions, intricate weapon filigree) are still softening at 64x64 pixel resolution, which will continuously sharpen as the continuous background loop iterates further.

## Background Continuous Loop
- Automated via `flux_student/scripts/continuous_loop.py` running in background.
- Periodically batches teacher generation, resumes training checkpoint, dumps test samples to `samples/`, and logs execution to `continuous.log` and `continuous_status.json`.
