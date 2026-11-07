# EXECUTION STATUS
Updated: 2026-11-07

## Latest train
- N indices: 2000
- epochs: 20
- CE: **0.327**
- free-gen from full MASK: still weak (blob)
- hybrid (partial mask + model fill): readable characters

## Honest gate
PASS for data path / hybrid recovery. FAIL for pure free neural character from empty mask.

Next: longer train + class conditioning or larger UNet on full 5104.
