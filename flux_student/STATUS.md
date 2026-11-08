# flux_student STATUS

Updated: 2026-11-08

## Overview
- Architecture: `StudentT2I` (base=48: 10,032,387 params, ~39 MB checkpoint; base=96: 26,600,067 params)
- LoRA Extension: Hook available via `model.inject_lora(r=4, scale=1.0)` targeting conditioning projection.
- Teacher Model: `stabilityai/sd-turbo` (1-step fast CPU/GPU synthesis, fallback for `FLUX.1-schnell`)
- Data: Prompt-only synthetic character assets with multiresolution pyramids generated across 32x32, 64x64, 128x128, and 256x256.
- Manifest schema: `{"id": str, "prompt": str, "teacher": str, "size_variants": [32, 64, 128, 256]}`.
- Safety: Zero-tolerance religious figure exclusion filtering implemented in `student/filter.py` and strictly enforced in teacher generation, continuous loop, and inference.

## Checkpoint & Training
- Primary Checkpoint: `checkpoints/student_final.pt` (~39 MB, 10,032,387 parameters)
- Epochs Completed: 30 manual epochs + continuous loop active (completed cycle 5, >38 total epochs)
- Loss Progression: Initial 1.0738 -> 1.0042 (manual) -> 0.8364 (continuous loop cycle 5)
- Manifest dataset: 24+ prompt-paired character items synthesized directly by teacher model

## Honest Quality Assessment
- The model learns character structure conditioning on prompt-paired teacher images.
- Current status: At 30 epochs (loss ~1.00), samples show diffuse high-frequency noise patterns with emerging overall hue/tint bias corresponding to prompt keywords.
- As continuous loop training progresses past cycle 5 (loss ~0.83), character silhouette contours and color massing (e.g. steel grey for armor, mechanical cyan/red accents for robot) begin solidifying, though fine features at 64x64 remain abstract/noisy until further epochs elapse.

## Continuous Background Loop
- Active background process: PID tracked in `continuous.pid` and `continuous_status.json`.
- Automatically synthesizes new character prompt batches from teacher, resumes checkpoint, trains for 8 epochs per cycle, saves `student_final.pt`, dumps samples to `samples/`, and logs execution to `continuous.log` and `continuous_status.json`.
