# flux_student TRAINING LOG

### 2026-11-02 Pipeline Execution Run
- **Teacher**: `stabilityai/sd-turbo` (CPU fp32 text-to-image synthesis)
- **Prompts**: Seed matrix in `scripts/prompts_seed.txt` containing mandatory bilingual full-body character prompts (knight, mage, robot, ninja, monster, warrior, etc.).
- **Dataset generated**:
  - `data/raw/`: 20 PNGs (256x256)
  - `data/resized/32/`: 20 PNGs (32x32)
  - `data/resized/64/`: 20 PNGs (64x64)
  - `data/resized/128/`: 20 PNGs (128x128)
  - `data/resized/256/`: 20 PNGs (256x256)
  - `data/manifest.jsonl`: 20 JSON entries with `id`, `prompt`, `teacher`.
- **Student Model Training**:
  - Model: `StudentT2I` (base=48, 10.03M parameters)
  - Epochs: 25 initial epochs
  - Loss progression: 1.0937 -> 1.0156
  - Output checkpoint: `checkpoints/student_final.pt` (38.32 MB)
- **Continuous Distillation Loop**:
  - Process started in background.
  - Cycle 1: added 4 teacher samples, 8 training epochs, final loss: `0.8357`.
  - Checkpoint updated: `checkpoints/student_final.pt`.
  - Samples generated: `samples/loop_gen_*.png`, `samples/sample_knight.png`, `samples/sample_robot.png`, `samples/sample_mage.png`.
