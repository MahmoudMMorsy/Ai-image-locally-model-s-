# flux_student STATUS

Updated: 2026-11-07

## Overview
- Architecture: `StudentT2I` (base=48: ~10.03M params, ~39 MB checkpoint; base=96: ~26.6M params)
- Teacher model: `stabilityai/sd-turbo` (1-step fast CPU/GPU synthesis, fallback for `FLUX.1-schnell`)
- Data: Multiresolution pyramids generated across 32x32, 64x64, 128x128, and 256x256
- Safety: Zero-tolerance religious figure exclusion filtering implemented in `student/filter.py` with Unicode word boundaries and active in teacher generation and continuous loop.

## Checkpoint & Training
- Checkpoint: `checkpoints/student_final.pt` (39 MB, 10,032,387 parameters)
- Epochs completed: Initial 25 epochs + active continuous loop cycles (8 epochs per cycle)
- Loss progression: Initial loss 1.0930 -> 1.0114 (initial training) -> 0.7541 (continuous cycle 1)
- Manifest dataset: 27+ prompt-paired character items synthesized directly by teacher model

## Honest Quality Assessment
- The model learns character structure conditioning on prompt-paired teacher images.
- Current status: At 25 epochs on CPU, samples show initial global color massing and coarse spatial clustering, but fine anatomical details remain noisy and coarse without high step counts or larger iterations. The continuous training loop actively refines parameter convergence.
- Contrast with Teacher: Teacher outputs provide coherent character figures and poses; student denoiser is actively learning representation mappings.

## Background Continuous Loop
- Automated via `flux_student/scripts/continuous_loop.py` running in background.
- Periodically batches teacher generation, resumes training checkpoint, dumps test samples to `samples/`, and logs execution to `continuous.log` and `continuous_status.json`.
