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
- 2026-11-02 (Update):
  - Expanded prompts list to 73 diverse archetypes in English and Arabic.
  - Synthesized teacher images via `stabilityai/sd-turbo` to 21 items across 32/64/128/256 pyramids.
  - Synced 21 generated character assets into `nano arcked dat` with matching JSON sidecars (2,877 total pairs).
  - Resumed Student model training on CPU for 10 epochs; loss reached ~1.0191.
  - Verified checkpoint at `checkpoints/student_final.pt` (~39MB, 10.03M params).
  - Inferred new samples: `sample_knight.png`, `sample_robot.png`, `sample_arabic_knight.png`.
