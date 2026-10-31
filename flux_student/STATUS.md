# flux_student STATUS

Updated: 2026-10-31

## Metrics
| Item | Value |
|------|-------|
| Teacher images | 20 (sd-turbo → 32/64/128/256) |
| Student | StudentT2I base=48 (~10M params, ~40MB fp32 ckpt) |
| Train loss (noise MSE) | **~0.49** (was ~1.0 at start) |
| Device | CPU |
| Augmentation | horizontal flip |

## Repo files
- `flux_student/student/model.py`
- `flux_student/scripts/generate_teacher_data.py`
- `flux_student/scripts/train_student.py`
- `flux_student/scripts/prompts_seed.txt`
- `flux_student/README.md`

## Honest assessment
Pipeline works end-to-end. **Quality still weak** for real characters — need 500–5000 teacher pairs and/or mix of arcade sprites from `mm.trine`.

## Next continuous steps
1. Expand teacher set when HF/pip network stable
2. Mix repo pixel sprites for style phase
3. Scale base→96 after n≥100
4. User LoRA on top of student checkpoint
