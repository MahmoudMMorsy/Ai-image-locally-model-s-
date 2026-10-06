# Parallel training status (2026-10-06)

## nano arcked dat — audit

README claims ~1692 images. On disk in the repo:

- Real PNG files found: **4** (under `seed_cc0/images/` only).
- Class folders (`warrior`, `knight`, `archer`, `mage`, ...): **JSON sidecars only**, no PNG.
- `metadata/`: ~999 JSON files.
- `dataset_index.csv` / `manifests/dataset.jsonl`: ~1692 rows (catalog), images not committed as binaries.

**Do not train on nano arcked dat as the image source.** It is taxonomy + manifests. Primary pixels remain **`mm.trine/` (~5109 PNG)**.

## Parallel work

| Track | Data | Status |
|-------|------|--------|
| HQ / bank target | filtered indices N=3505 from mm.trine lineage | production quality reference |
| Discrete Diff lite | same N=3505 | best CE ~0.93 (ep14), samples not character-quality yet |
| Discrete Diff UNet128 | same N=3505 | restarted larger arch, CE ~1.33 after 3 ep |

## Rules for Jules / agents

1. Train on `mm.trine` PNGs (count must be printed; abort if < 1000).
2. Ignore empty class trees under `nano arcked dat` for pixel training.
3. Use `nano arcked dat` only for taxonomy labels when matching files exist.
4. From-scratch path = discrete masked diffusion on palette indices; HQ skip-mix is separate and labeled.

See AGENTS.md and PLAN_FROM_SCRATCH.md.
