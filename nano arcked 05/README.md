# nano arcked 05

Synced package (2026-10-05) combining:
- Audit of repo folder `nano arcked dat`
- Seed CC0 images cleaned/previewed
- Link to production model: `arcade_palette` (PaletteUNet + SkipPredictor)
- Quality recommendations from `nano_pixel_mob` experiments

## Honest data status

| Source | Reality |
|--------|---------|
| `nano arcked dat` metadata | ~1045 JSON entries |
| Actual PNG files in that folder | **4** seed spritesheets only |
| Catalog `dataset_index.csv` | 40 rows (filenames, mostly **no image binary** in repo) |
| Local training bank (session) | **3505** latents / **4193** index maps — primary data |

**Conclusion:** Do not claim `nano arcked dat` is a full image dataset. Use taxonomy labels + seed previews only. Training quality depends on the existing arcade bank, not empty catalog paths.

## Seed images (cleaned)

Under `clean_seed/` (local package):
- game_character.png (90x90 sheet)
- missprincess.png (32x64)
- ghosts.png (240x192 sheet)
- rpg_sprite_walk.png (192x128 sheet)

These are **too few** and mostly sheets — not enough to replace the 64x64 arcade bank.

## Production model path

Use sibling folder on branch `Meta.data`:
- `arcade_palette/` — HQ skip-mix + Hybrid SP (best CE ~0.174)

## Repo tech that can improve quality

From `nano_pixel_mob/2026-09-12`:
1. **Discrete masked diffusion** — pure neural discrete tokens, faster CPU
2. **VQ-VAE + AR prior** — compressed 8x8 codes
3. **PixelLLM** — autoregressive palette tokens

Recommended next step: keep HQ path; optionally train discrete diffusion on existing `full_indices_u8.pt` for pure neural without inventing missing images.

## Taxonomy

- `taxonomy/catalog_full_index.json` — 40 catalog rows
- `taxonomy/catalog_64_arcade_style.json` — subset style/resolution filters

## Docs

- `docs/REPO_AUDIT.md` — full audit
- `docs/seed_analysis.json` — seed dimensions
