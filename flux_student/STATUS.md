# flux_student STATUS

Updated: 2026-11-10

## Overview
- Architecture: `StudentT2I` (base=48: ~10.03M params, ~38.32 MB checkpoint; base=96: ~26.6M params)
- Teacher model: `stabilityai/sd-turbo` (1-step fast CPU/GPU synthesis, fallback for `FLUX.1-schnell`)
- Data: Multiresolution pyramids generated across 32x32, 64x64, 128x128, and 256x256
- Safety: Zero-tolerance religious figure exclusion filtering implemented in `student/filter.py` (with regex word/Unicode boundary checks) and active across teacher generation, continuous loop, and inference.

## Checkpoint & Training
- Checkpoint: `flux_student/checkpoints/student_final.pt` (38.32 MB, 10,032,387 parameters)
- Epochs completed: 25 initial epochs + background continuous loop cycles active
- Loss progression: Initial loss ~1.0701 -> 1.0129 -> 0.9960 -> 0.7519 (consistently converging downward)
- Manifest dataset: 34+ prompt-paired character items synthesized directly by teacher model into `data/manifest.jsonl` and multiresolution pyramids

## Honest Quality Assessment
- The student model successfully runs end-to-end forward denoising and inference sampling.
- Current visual quality: Early denoising samples exhibit emerging color masses and silhouette clusters corresponding to character classes (e.g. silver/dark steel palette for knights, metallic dark tones for robots), but still retain prominent diffusion noise at 64x64 resolution on CPU before extensive multi-hour training.
- The training loss is actively and consistently declining (down to ~0.7519), showing proper neural weight optimization.

## Background Continuous Loop
- Automated via `flux_student/scripts/continuous_loop.py` running in background (PID recorded in `continuous.pid`).
- Continuously batches teacher generation, resumes training checkpoint, dumps test samples to `samples/`, and logs execution to `continuous.log` and `continuous_status.json`.
