# STATUS — flux_student
Updated: 2026-11-04

## Pipeline complete (code)
| Component | Path |
|-----------|------|
| Student model | student/model.py (~10M params) |
| Teacher gen | scripts/generate_teacher_data.py |
| Train | scripts/train_student.py, train_once.py |
| Continuous loop | scripts/continuous_loop.py |
| Safety filter | student/filter.py |
| Prompts | scripts/prompts_seed.txt |

## Local metrics
- Teacher images: 26 (sd-turbo, prompt-only)
- Checkpoint: ~40 MB
- Last train: 20 epochs, loss ≈ 0.383

## Repo main (Jules)
- manifest ~36, cycle 4, PR #139/#142 merged

## Honest blockers
- Need more teacher pairs (hundreds+) for clear characters
- pip/HF 502 blocks expanding teacher set in some sessions

## Commands
```bash
python scripts/generate_teacher_data.py --n 32 --teacher sd-turbo
python scripts/train_once.py
nohup python3 -u scripts/continuous_loop.py >> continuous.log 2>&1 &
```
