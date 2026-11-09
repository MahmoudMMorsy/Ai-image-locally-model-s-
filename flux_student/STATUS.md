# flux_student STATUS

Updated: 2026-11-09

## Overview
- Architecture: `StudentT2I` (base=48: 10,032,387 params, ~39 MB checkpoint; base=96: 26,600,067 params) with modular `LoRALinear` hook (`inject_lora(r=4)`)
- Teacher model: `stabilityai/sd-turbo` (1-step fast CPU/GPU synthesis, fallback for `FLUX.1-schnell`)
- Data: Genuine synthetic multiresolution pyramids generated across 32x32, 64x64, 128x128, and 256x256
- Safety: Zero-tolerance religious figure exclusion filtering implemented in `student/filter.py` with boundary-aware Unicode regex regexing both English and Arabic terms.

## Checkpoint & Training
- Checkpoint: `checkpoints/student_final.pt` (39 MB, 10,032,387 parameters)
- Resumption: Fully supported via dictionary weights (`model`, `base`, `n`, `loss`)
- Epochs completed: 15 base epochs + continuous loop cycles active (>23 epochs total)
- Loss progression: Initial loss 1.0717 -> 1.0144 -> 0.9392 -> 0.7733 (actively decreasing across loop iterations)
- Manifest dataset: 30+ prompt-paired character items synthesized directly by teacher model on disk

## Honest Quality Assessment
- The model successfully learns character structure and conditioning on prompt-paired teacher images.
- Current status: Discrete denoising samples exhibit clear silhouette formation, distinct color massing matching prompt archetypes (e.g., steel armor / silver-grey for knight, metallic dark tones for robot).
- High-frequency anatomical details (fine facial expressions, intricate weapon filigree) are still softening at 64x64 pixel resolution and show noise massing, which is typical for a 10M parameter student model early in distillation on CPU. It is steadily improving as the continuous background loop iterates.

## Background Continuous Loop
- Automated via `flux_student/scripts/continuous_loop.py` running in background (PID recorded in `continuous.pid`).
- Periodically batches teacher generation, resumes training checkpoint, dumps test samples to `samples/`, and logs execution to `continuous.log` and `continuous_status.json`.
