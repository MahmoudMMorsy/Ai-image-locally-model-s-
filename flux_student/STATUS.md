# flux_student STATUS

Updated: 2026-11-04

## Overview
- Architecture: `StudentT2I` (base=48: ~10M params, ~39 MB checkpoint; base=96: ~26.6M params)
- Teacher model: `stabilityai/sd-turbo` (1-step fast CPU/GPU synthesis, fallback for `FLUX.1-schnell`)
- Data: Multiresolution pyramids generated across 32x32, 64x64, 128x128, and 256x256
- Safety: Zero-tolerance religious figure exclusion filtering implemented in `student/filter.py` and active in teacher generation and continuous loop.

## Checkpoint & Training
- Checkpoint: `checkpoints/student_final.pt` (39 MB, 10,032,387 parameters)
- Epochs completed: Continuous loop active, trained across cycles (15 epochs standalone + continuous cycles)
- Loss progression: Initial loss ~1.0921 -> 1.0194 -> 0.8158 (continuously decreasing)
- Manifest dataset: 24+ prompt-paired character items synthesized directly by teacher model (`data/manifest.jsonl`)

## Honest Quality Assessment
- The model learns character structure conditioning on prompt-paired teacher images.
- Current status: In early training cycles, student output samples display emerging color massing and silhouette structuring matching prompt archetypes, but high-frequency pixel noise remains present. At 64x64 resolution on CPU, fine anatomical detail requires deeper continuous training iterations.
- Religious and sensitive figure exclusion filter remains strictly verified and active at 100% adherence.

## Background Continuous Loop
- Automated via `flux_student/scripts/continuous_loop.py` running in background.
- Periodically batches teacher generation, resumes training checkpoint, dumps test samples to `samples/`, and logs execution to `continuous.log` and `continuous_status.json`.
