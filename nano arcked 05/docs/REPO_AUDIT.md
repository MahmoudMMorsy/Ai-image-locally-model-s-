# Repo audit — nano arcked dat + quality

```json
{
  "nano_arcked_dat": {
    "metadata_json_count": 1045,
    "actual_png_in_repo": 4,
    "verdict": "Catalog/metadata heavy; real images mostly NOT in repo.",
    "usable_now": "4 seed CC0 spritesheets — low count, mixed sizes."
  },
  "local_arcade_bank": {
    "latent_bank": 3505,
    "indices": 4193,
    "verdict": "Primary training data already in use."
  },
  "repo_tech_worth_borrowing": [
    "discrete masked diffusion",
    "VQ-VAE + prior",
    "PixelLLM autoregressive palette tokens",
    "Taxonomy type/gender/style/race/pose"
  ],
  "quality_plan": [
    "Keep PaletteUNet+SP HQ as production",
    "Do not dilute bank with 16x16 sheets without careful crop",
    "Optional: discrete diffusion on full_indices_u8.pt"
  ]
}
```
