# flux_student STATUS

Updated: 2026-11-10

## Overview
- Architecture: `StudentT2I` (base=48: ~10.03M params, ~40 MB checkpoint; base=96: ~26.6M params) with modular LoRA adapter hook (`inject_lora(r=4)`).
- Teacher model: `stabilityai/sd-turbo` (1-step fast CPU/GPU synthesis, fallback for `FLUX.1-schnell`) generating full-body characters from text only.
- Data: Multi-resolution pyramids generated strictly across 32x32, 64x64, 128x128, and 256x256 recorded in `data/manifest.jsonl` with fields: `id`, `prompt`, `teacher`, `size_variants`.
- Safety: Zero-tolerance religious figure exclusion filtering implemented in `student/filter.py` and active in teacher generation and continuous loop.

## Checkpoint & Training
- Checkpoint: `checkpoints/student_final.pt` (40.18 MB, 10,032,387 parameters).
- Epochs completed: Initial standalone training (15 epochs) + continuous loop cycles active (>30 epochs total).
- Loss progression: Standalone epoch 1 (1.1212) -> epoch 15 (1.0272) -> continuous loop cycle 1 (1.0011) -> cycle 2 (0.9109) -> cycle 3 (0.8670).
- Manifest dataset: Prompt-paired full-body character items synthesized directly by teacher model (growing unattended via continuous loop).

## Honest Quality Assessment
- The model learns character structure conditioning on prompt-paired teacher images.
- Current status: Denoising samples at early-to-mid training iterations exhibit global character silhouette formation and distinct color massing matching prompt archetypes (e.g., silver-grey / metallic steel tones for knight, metallic red / mechanical highlights for robot android).
- High-frequency anatomical details (sharp facial expressions, intricate weapon filigree) remain soft/noisy at 64x64 pixel resolution, typical of early diffusion/denoiser training on CPU with a small batch count, continuously sharpening as the continuous loop adds synthetic teacher samples and trains further epochs.

## Background Continuous Loop
- Automated via `flux_student/scripts/continuous_loop.py` running in background (PID recorded in `continuous.pid`).
- Periodically batches teacher generation, resumes training checkpoint, dumps test samples to `samples/`, and logs execution to `continuous.log` and `continuous_status.json`.
