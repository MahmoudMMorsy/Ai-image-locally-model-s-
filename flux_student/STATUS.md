# flux_student STATUS

Updated: 2026-11-02

## Overview
- Architecture: `StudentT2I` (base=48: ~10M params, 38.3 MB checkpoint; base=96: ~26.6M params)
- Teacher model: `stabilityai/sd-turbo` (1-step fast CPU/GPU synthesis, fallback for `FLUX.1-schnell`)
- Data: Multiresolution pyramids generated across 32x32, 64x64, 128x128, and 256x256
- Safety: Zero-tolerance religious figure exclusion filtering implemented in `student/filter.py`

## Checkpoint & Training
- Checkpoint: `checkpoints/student_final.pt` (~38.3 MB, 10,032,387 parameters)
- Epochs completed: 5 epochs (CPU batch size 4)
- Final loss: ~1.0242

## Honest Quality Assessment
- The model currently learns early pixel noise conditioning on prompt-paired teacher images.
- Early stage: Samples exhibit coherent color clustering and general figure silhouette placement without full high-definition anatomy yet, which is expected at early training iterations (<50 epochs, N=8..16).
- Progressive improvement is automated via the unattended `continuous_loop.py` script.

## Next Milestones
1. Let unattended `continuous_loop.py` run to accumulate more teacher samples and reduce loss below 0.5.
2. Scale up to base=96 (~26.6M params) when dataset size exceeds 200 pairs.
