# flux_student STATUS

Updated: 2026-11-02

## Overview
- Architecture: `StudentT2I` (base=48: ~10.03M params, ~38.3 MB checkpoint; base=96: ~26.6M params)
- Teacher model: `stabilityai/sd-turbo` (fast synthesis, fallback for `FLUX.1-schnell`)
- Data: Multiresolution pyramids generated across 32x32, 64x64, 128x128, and 256x256 (21 manifest items)
- Safety: Zero-tolerance religious figure exclusion filtering implemented in `student/filter.py` with boundary protection against substring false positives.
- Taxonomy Integration: Synced 21 synthesized full-body character items into `nano arcked dat` 7-level taxonomy hierarchy with complete 1:1 image/JSON metadata sidecars.

## Checkpoint & Training
- Checkpoint: `checkpoints/student_final.pt` (~38.3 MB, 10,032,387 parameters)
- Epochs completed: 10 epochs (CPU batch size 4, resumed)
- Final loss: ~1.0191

## Honest Quality Assessment
- The model currently learns early pixel noise conditioning on prompt-paired teacher images.
- Current Quality Status: At early training iterations (<50 epochs, N=21), samples exhibit initial color clustering and figure silhouette placement with visible grain and diffuse noise; fine anatomical details (distinct limbs, facial features) require hundreds more training cycles and further dataset expansion.
- Generated samples are exported to `samples/sample_knight.png`, `samples/sample_robot.png`, and `samples/sample_arabic_knight.png`.

## Next Milestones
1. Let unattended `continuous_loop.py` run continuously to accumulate more teacher samples and reduce loss below 0.5.
2. Scale up to base=96 (~26.6M params) when dataset size exceeds 200 pairs.
