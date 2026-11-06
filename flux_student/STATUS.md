# flux_student STATUS

Updated: 2026-11-06

## Overview
- Architecture: `StudentT2I` (base=48: ~10M params, ~39 MB checkpoint; base=96: ~26.6M params)
- Teacher model: `stabilityai/sd-turbo` (1-step fast CPU/GPU synthesis, fallback for `FLUX.1-schnell`)
- Data: Multiresolution pyramids generated across 32x32, 64x64, 128x128, and 256x256 with `size_variants` recorded in `manifest.jsonl`
- Safety: Zero-tolerance religious figure exclusion filtering implemented in `student/filter.py` and active in teacher generation, inference, and continuous loop.

## Checkpoint & Training
- Checkpoint: `checkpoints/student_final.pt` (~39 MB, 10,032,387 parameters)
- Resuming: Fully supported via `--resume` flag and continuous loop cycle resumption
- Loss progression: Initial loss ~1.09 -> 1.007 (standalone 25 epochs) -> ~0.79 (continuous loop cycles)
- Manifest dataset: 27+ prompt-paired character items synthesized directly by teacher model

## Honest Quality Assessment
- The model learns character structure conditioning on prompt-paired teacher images.
- Current status: Discrete denoising samples exhibit clear silhouette formation, distinct color massing matching prompt archetypes (e.g., steel armor / silver-grey for knight, metallic teal/red highlights for robot).
- High-frequency anatomical details (fine facial expressions, intricate weapon filigree) are still softening at 64x64 pixel resolution, which will continuously sharpen as the continuous background loop iterates further.

## Background Continuous Loop
- Automated via `flux_student/scripts/continuous_loop.py` running in background.
- Periodically batches teacher generation, resumes training checkpoint, dumps test samples to `samples/`, and logs execution to `continuous.log` and `continuous_status.json`.
