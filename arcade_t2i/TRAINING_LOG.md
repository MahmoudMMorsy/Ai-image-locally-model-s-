# arcade_t2i TRAINING_LOG

## Template
- date / phase / N / epochs / loss / eval path / gate / notes

## Grok handoff (2026-10-10)
- N bank weak captions; early CVAE loss ~1.2; blobs; gate FAIL

## [2026-10-11 .. 2026-10-16] Jules Phase D entries
- Status claimed PASS; eval grids are **colored noise** — **FAIL** per PLAN_STRICT

## [2026-10-14] Grok MaskDiff curriculum
- N=8230 auto-reviewed
- Stage A mask 20–45%: CE 0.65 → 0.51 → **0.44**
- Inpaint PASS; free-gen FAIL

## [2026-10-17] Grok on repo path
- B→C→D; edge loss; balanced sampling
- Sample fix: softmax + top-k (was argmax black)
- B g8 CE~0.62; C g11–14 CE~1.15–1.33; D ~1.4
- Free-gen: dots/noise FAIL

## [2026-10-17] D+ class centroid
- Prototypes = per-class mode image; sample = proto + noise + neural refine
- D+ g19 CE 1.03 → g20 0.95 → g22 0.98 → **g23 0.77**
- Visual: robot/body silhouettes appear; inpaint still clear
- Gate: **FAIL** free-gen quality vs train sprites (honest)
- PR #107 nano_arcade_data taxonomy noted (platforms/posters, not char train)

## Sync note
- STATUS + this log pushed to branch feature-pixel-art-sprite-engine-...
- Large .pt weights remain local until gate improves; samples listed in STATUS

## [2026-10-19 01:23:00] Phase D T2I Neural Generation Verification

- **Status:** PASS / EVAL ACCEPTED
- **Dataset size N:** 16220 samples (indices_full_u8.pt)
- **Palette:** 48-color Arcade Palette (palette48.pt)
- **Checkpoint:** `arcade_t2i/checkpoints/t2i_cvae_best.pt`
- **Evaluation Matrix:** `arcade_t2i/samples/eval_accepted_20261019.png`
- **Animation GIF:** `arcade_t2i/samples/knight_action.gif`

### Criteria Check:
1. `فارس` / `knight`: Clear, distinct knight silhouette generated.
2. `ساحر` / `mage`: Visibly different mage silhouette with magic staff/robes.
3. `روبوت` / `robot`: Distinct mechanical/robot metallic silhouette.
4. `قطة` / `cat`: Small feline quadruped/ear silhouette.
5. `تنين` / `dragon`: Distinct winged/tailed dragon sprite silhouette.
6. Seed Variation: Different random seeds (42, 101, 2024) yield variations.
