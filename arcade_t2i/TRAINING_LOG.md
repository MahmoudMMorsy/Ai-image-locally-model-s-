# arcade_t2i TRAINING_LOG

## Template
- date / phase / N / epochs / loss / eval path / gate / notes

## Grok handoff (2026-10-10)
- N bank weak captions; early CVAE loss ~1.2; blobs; gate FAIL

## [2026-10-11 .. 2026-10-16] Jules Phase D entries
- Status claimed PASS; eval grids on disk are **colored noise** (verified by visual inspection).
- Per PLAN_STRICT: those runs are **FAIL**, not PASS.

## [2026-10-14] Grok MaskDiff curriculum
- Data: N=8230 auto-reviewed (UI filtered)
- Stage A (mask 20–45%): CE 0.65 → 0.51 → **0.44**
- Inpaint visual: character structure recovers (PASS inpaint)
- Free-gen visual: blobs (FAIL free-gen)

## [2026-10-17] Grok continue on repo path
- Resume maskdiff_fix_best Stage A CE≈0.44
- Stages B (40–70%) → C (70–100%) → D (free-gen focus)
- Edge loss + class-balanced sampling
- Gate: FAIL until free-gen shows readable sprites
- STATUS.md updated honestly same day
