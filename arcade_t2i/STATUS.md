# STATUS — 2026-10-17 (honest)

## What works
- **Inpainting** (partial mask 20–50%): character structure recovers — PASS for inpaint path.
- Data pipeline: auto-reviewed N≈8230 sprites, bilingual AR/EN captions, 48-color palette.
- MaskDiff curriculum A→B→C→D running (CE was ~0.44 on Stage A).

## What does NOT work yet
- **Free neural generation** (100% mask / text→image from noise): still color blobs / noise — **GATE FAIL**.
- Jules prior `eval_accepted_*.png` images are colored noise; claiming PASS on those violates `PLAN_STRICT.md`.

## Rules (do not change)
1. Noise/blobs = FAIL. Do not write PASS without recognizable knight/mage/robot/dragon.
2. Train on real pixel indices, not empty taxonomy folders.
3. Log CE + **visual** sample path every run.

## Current best checkpoint (local handoff)
- `maskdiff_fix_best.pt` — stage A CE≈0.44, then B/C/D in progress 2026-10-17.
- Goal: free-gen readable 64×64 arcade sprites from AR/EN text.

## Next
- Finish curriculum C/D on MaskDiff.
- Only mark Gate PASS when free-gen samples show clear character silhouettes.
