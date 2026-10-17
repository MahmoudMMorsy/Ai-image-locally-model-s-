# STATUS — 2026-10-17 (synced to repo)

## What works
- **Inpainting** (partial mask): clear character recovery — **PASS**
- Data: N≈8230 auto-reviewed, bilingual AR/EN, palette48
- Curriculum A→B→C→D completed; **D+ centroid** in progress
- Softmax multinomial sampler (fixed argmax→black collapse)
- Class prototypes (`class_prototypes.pt`) for free-gen init

## Metrics (local → to be uploaded)
| Stage | Epoch | CE | Notes |
|-------|-------|-----|-------|
| A | 3 | 0.44 | inpaint solid |
| B | 8 | ~0.62 | |
| C | 14 | ~1.33 | high mask |
| D | 18 | ~1.41 | pure free weak |
| **D+** | **23** | **0.77** | centroid+noise refine |

## Free-gen visual
- Before: black / sparse dots
- D+ g20–g22: humanoid/robot silhouettes visible (centroid path)
- **Still GATE FAIL** vs training-quality sprites (not claimed PASS)

## Jules eval_accepted_*.png
- Remain colored noise; do not treat as PASS (PLAN_STRICT)

## PR #107
- `nano_arcade_data/` platform taxonomy (posters/logos) — noted, not character sprite train set

## Artifacts (local handoff paths)
- `checkpoints/maskdiff_fix_best.pt` (~2MB) epoch 23 CE 0.77
- `checkpoints/class_prototypes.pt`
- `samples/repo_Dp_g20.png`, `repo_Dp_g22.png`, `MASKDIFF_EVAL_NOW.png`

## Next
- Continue D+ until free-gen readable; upload weights when gate approaches PASS
- Only mark Gate PASS with clear knight/mage/robot/dragon free-gen
