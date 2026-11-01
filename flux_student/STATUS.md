# STATUS — flux_student
Updated: 2026-11-02

## Delivery checklist
| Item | Status |
|------|--------|
| student/model.py | OK ~10M params (base=48) |
| generate_teacher_data.py | OK |
| train_student.py | OK |
| continuous_loop.py | OK |
| prompts_seed.txt | character-only AR/EN |
| data/manifest | 26 teacher images (sd-turbo + prompt) |
| checkpoints/student_final.pt | ~40MB local |
| DATA_POLICY | prompt-only teacher |

## Metrics
- n_data: 26
- last train: 30 epochs, loss ~0.37–0.39
- Teacher: sd-turbo prompt-only
- Quality: still noisy at n=26; need 500+ pairs

## Commands
```bash
python scripts/generate_teacher_data.py --n 32 --teacher sd-turbo
python scripts/train_student.py --size 64 --epochs 30 --cpu --base 48
nohup python3 -u scripts/continuous_loop.py >> continuous.log 2>&1 &
```
