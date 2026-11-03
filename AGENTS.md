# AGENTS.md — instructions for Jules and other coding agents

Read this file + **UNIFIED_PIPELINE.md** before any training, generation, or dataset change.

## Goal

Build a real 64x64 arcade pixel-art generator that can produce a **new character that is not a copy of a training image**, at quality close to the training sprites.

Neural generation (palette-index discrete diffusion / T2I), **not** procedural stitching and **not** nearest-neighbor remix as the final product.

## Unified entry

See **`UNIFIED_PIPELINE.md`** for the single ordered plan:
1. Index ALL repo sprites → `arcade_t2i/manifests/manifest_unified.jsonl`
2. Palette48 + indices
3. Train discrete / T2I on full N in `arcade_t2i/`
4. `flux_student/` is **helper distill only** — does not replace full-data training

## Dataset rule (mandatory)

DO NOT train on `dataset_clean/` or `dataset_clean.zip` alone (~101 images).

Use **all** real sprites:

| Source | Approx count | Use |
|--------|----------------|-----|
| `mm.trine/` | **~5100 PNG**, ~111 characters | PRIMARY |
| `nano arcked dat/` | large (official_cc0 + data + classes) | PRIMARY |
| `nano_arcade_data/` | ~2.5k | after 64x64 filter |
| `examples/` | ~199 | optional |
| `dataset_clean.zip` | 101 | do **not** use as the only set |

Hard stop if N < 1000 after indexing. Aim N >= 15000 when full repo is available.

## Do not

- Do not log "trained" after 10 steps on `torch.randn`.
- Do not treat incomplete `models/nano_pixel_*` stubs as production.
- Do not shrink the dataset to 101 images to finish a daily job.
- Do not claim text-to-image PASS without readable knight/mage/robot samples.
- Do not present flux_student 26–36 teacher pairs as full from-scratch quality.

## Collaboration

- Plan: `UNIFIED_PIPELINE.md`, `PLAN_FROM_SCRATCH.md`, `arcade_t2i/JULES_T2I_PLAN.md`
- Issues labeled `jules` → Jules implements on a branch + PR
- Comment if data counts disagree; do not silently switch back to the 101-image set.
