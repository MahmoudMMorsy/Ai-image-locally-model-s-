# nano arcked dat

Pixel-art image collection for the local image-generation project.

## Collection rules

- Target: pixel art, especially 8-bit / 16-bit / arcade aesthetics.
- Priority resolutions: 32x32, 64x64, 128x128, plus small spritesheets that can be indexed.
- Prefer full-body characters with readable anatomy, then creatures, monsters, robots, items, weapons, vehicles, environments and effects.
- Preserve source provenance and license metadata for every collected asset.
- CC0/public-domain assets are the primary redistributable pool.
- Paid, copyrighted, game-rip, marketplace, and search-discovered candidates are also part of the research catalog, but they are tracked as **preview-only / rights-required** until the user acquires the necessary rights.
- Other licenses are tracked separately and are never mislabeled as CC0.
- Search engines and Pinterest are discovery layers, not proof of ownership. Every candidate must resolve to its original creator, game, product page, or rights holder.
- Creator notices that prohibit AI/ML use are treated as exclusion rules.
- Sacred/religious exclusion filter is applied before ingestion. The filter is intentionally conservative: uncertain items are quarantined instead of being silently included.
- The dataset is for artistic/research collection; training is outside this repository's collection pipeline.

## Image-first rule

**The repository is image-first.** Every asset that enters the mirror lane must exist as an actual image file in the repository. Metadata is never a substitute for the image.

- One real image file per collected asset, using PNG/WebP/GIF/JPG/JPEG as supplied or safely converted when the pipeline explicitly allows it.
- Metadata is stored separately under `metadata/<asset-id>.json` and is never required to sit beside the image.
- Manifests are separate indexes only. They do not replace image files.
- No fake placeholder files, text files pretending to be assets, or metadata-only entries in the image dataset.
- For a protected/paid asset that cannot legally be mirrored yet, the repository records a rights-research entry with its original preview/source URL instead of pretending that a missing binary is a collected image. Once redistribution rights are established, its actual image can move into the image pool.

## Layout

`images/<type>/<gender>/<style>/<race>/<source>/<pose>/<asset>.png`
`metadata/<asset-id>.json`
`manifests/dataset.jsonl`
`manifests/dataset.csv`
`manifests/quarantine.jsonl`

**Separation rule:** image, metadata, and manifest are three independent layers. The image is the primary dataset object; metadata only describes it.

The collector can be expanded with additional source catalogs without changing the metadata schema.

## Source strategy

We do **not** limit discovery to CC0. The project has two lanes:

1. **Mirror lane:** sources whose terms explicitly permit redistribution, such as verified CC0/public-domain collections. These may be downloaded into `images/` by automation.
2. **Rights-research lane:** paid stores, game sprite archives, image-search results, Pinterest, and other protected sources. These are cataloged with the exact page URL, creator/title, license status, acquisition status, and AI/ML restrictions. Protected binaries are not copied into this public repository before rights are established.

Examples now tracked include The Spriters Resource, CraftPix, Free Game Assets on itch.io, itch.io pixel-art asset discovery, GameDev Market, Unity Asset Store, Pinterest, and general image-search discovery.

## Current collection sources

The automated mirror currently uses a curated redistributable pool. The broader source catalog is intentionally larger and includes paid/protected sources for later rights acquisition and visual review.

The Tiddybub catalog states that its included packs are CC0 and preserves SOURCE.md provenance for each pack. OpenGameArt records are checked individually before being added to the source catalog.

## Important

This repository records provenance; it does not magically change the copyright/license of an asset. Always retain the original source and license information.
