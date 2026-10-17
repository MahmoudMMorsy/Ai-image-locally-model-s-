# STATUS — 2026-10-17 (honest, updated)

## What works
- **Inpainting** (partial mask 20–50%): character structure recovers — **PASS** for inpaint path.
- Data: auto-reviewed N≈8230; bilingual AR/EN; 48-color palette.
- MaskDiff curriculum A→B done; **C in progress** (ep~12).
- Softmax free-gen sampler (replaces argmax black collapse).

## What does NOT work yet
- **Free neural generation**: faint silhouettes / noise — **GATE FAIL** (not training-quality sprites).
- Jules `eval_accepted_*.png` remain colored noise; do not treat as PASS.

## Repo updates absorbed (no replace)
- PR dataset aggregation in `nano arcked dat` noted; training continues on reviewed indices already built.
- STATUS/TRAINING_LOG append-only progress; prior honest FAIL kept.

## Rules
1. Noise/blobs = FAIL.
2. Real pixel indices only.
3. Log CE + visual path every run.

## Next
- Finish C/D; only Gate PASS when free-gen shows clear knight/mage/robot/dragon.
