# Training log

- 2026-10-30: Pipeline bootstrap, 20 teacher imgs (sd-turbo)
- 2026-11-02:
  - Added strict religious figure exclusion filter `student/filter.py` and test suite `tests/test_pipeline.py`.
  - Added inference CLI `scripts/inference.py` supporting multiresolution output (32, 64, 128, 256).
  - Seed prompts expanded with 50 diverse character archetypes (English & Arabic).
  - 8 teacher character images synthesized with `stabilityai/sd-turbo` and processed into 32/64/128/256 pyramids.
  - Student model (base=48, 10.03M params) trained 5 epochs on CPU; loss ~1.024.
  - Checkpoint saved to `checkpoints/student_final.pt` (38.3 MB).
  - Inference samples generated in `samples/sample_knight.png` and `samples/sample_robot.png`.
