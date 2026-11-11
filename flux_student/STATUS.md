# flux_student STATUS

Updated: 2026-11-11

## Overview
- Architecture: `StudentT2I` (base=48: ~10.03M params, ~39 MB checkpoint; base=96: ~26.6M params)
- Teacher model: `stabilityai/sd-turbo` (1-step fast CPU/GPU synthesis, prompt-only character data source)
- Multi-resolution pyramids: 32x32, 64x64, 128x128, and 256x256
- Safety: Zero-tolerance religious figure exclusion filtering implemented in `student/filter.py` and strictly enforced during generation and training.
- Modular hooks: Integrated `LoRALinear` and `inject_lora()` hook in `student/model.py` for text-conditioning adapter expansion.

## Checkpoint & Training Status
- Checkpoint: `flux_student/checkpoints/student_final.pt` (39 MB, 10,032,387 parameters)
- Loss progression:
  - Phase 1 (Initial direct training, 25 epochs): Loss reduced from 1.0729 down to 1.0075
  - Phase 2 (Continuous loop cycle 1, 8 epochs): Loss reduced from 0.9941 to 0.7973
  - Phase 3 (Continuous loop cycle 2, 8 epochs): Loss reduced from 0.7833 to 0.7098 (continuously decreasing)
- Manifest dataset: 29 verified prompt-synthesized character entries in `flux_student/data/manifest.jsonl`

## Honest Quality Assessment (تقييم صادق للجودة)
1. **Silhouette & Massing**:
   - The student model successfully predicts denoised representations with distinct character silhouettes and centralized character framing against cleaner background tones.
   - Chromatic palette conditioning matches prompt archetypes: silver-grey armor for knight prompts, metallic slate/red accents for robot prompts, and mystical robes/aura for mage prompts.
2. **Current Limitations & Noise Comparison**:
   - While global composition and character masses are clearly separated from background noise, high-frequency fine details (e.g. sharp finger contours, intricate facial features, crisp weapon texturing) remain soft or blurred at 64x64 denoising steps.
   - It is not pure random noise; clear anatomical body presence and character posture are formed, but sharp character pixel art lines require additional continuous distillation cycles and higher epoch density to achieve crisp definition.

## Background Continuous Loop
- Automated background loop running under `flux_student/scripts/continuous_loop.py` (PID recorded in `continuous.pid`).
- Tracks state in `continuous_status.json` and logs every generation, epoch loss, and sample dump in `continuous.log`.
