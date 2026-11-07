# flux_student STATUS

Updated: 2026-11-07

## Overview
- Architecture: `StudentT2I` (base=48: 10,032,387 params, ~40.1 MB checkpoint; base=96: ~26.6M params), with modular `LoRALinear` hook support.
- Teacher model: `stabilityai/sd-turbo` (1-step fast CPU/GPU synthesis, fallback for `FLUX.1-schnell`).
- Data: Prompt-only synthesis generated on disk into `data/raw/` and multiresolution pyramids across 32x32, 64x64, 128x128, and 256x256.
- Manifest: `data/manifest.jsonl` tracking real verified entries with `id`, `prompt`, `teacher`, and `size_variants` ([32, 64, 128, 256]).
- Safety: Zero-tolerance religious figure exclusion filtering implemented in `student/filter.py` active across dataset generation, training, inference, and the continuous loop.

## Checkpoint & Training
- Checkpoint: `checkpoints/student_final.pt` (~40.1 MB, 10,032,387 parameters).
- Training: Multi-epoch baseline training plus background continuous cycles.
- Loss progression: 1.0920 -> 1.0184 (initial 25 epochs) -> 0.8109 (cycle 1 on CPU, continuing downward).
- Manifest dataset: Active prompt-paired character items synthesized directly by teacher model.

## Honest Quality Assessment
- The model successfully learns character structure and conditioning on prompt-paired teacher images.
- Current status: Early denoising samples produce silhouette massing and prompt-guided color distributions (e.g., metallic/blue and armor tones for knight, robotic highlights for android). At current epochs on 64x64 resolution, output resembles high-frequency pixel noise with emergent silhouette clustering; fully crisp anatomical features require extended training iterations which the background continuous loop is actively providing.

## Background Continuous Loop
- Automated via `flux_student/scripts/continuous_loop.py` running actively in background.
- Periodically synthesizes new teacher character batches, resumes training on `student_final.pt`, dumps test samples to `samples/`, and updates `continuous.log` and `continuous_status.json`.
