# STATUS
Updated: 2026-11-02

- n_data: 26 (sd-turbo + prompt only)
- last train: 20 epochs, loss=0.3817
- StudentT2I base=48 ~10M params
- continuous_loop: running when sandbox alive
- Honest quality: samples still noisy; need 500+ teacher images

## Commands
```bash
python scripts/train_once.py
nohup python3 -u scripts/continuous_loop.py >> continuous.log 2>&1 &
```
