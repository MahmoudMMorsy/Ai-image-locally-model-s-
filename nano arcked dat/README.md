# nano arcked dat

Pixel-art image collection for the local image-generation project.

## Collection rules

- Target: pixel art, especially 8-bit / 16-bit / arcade aesthetics.
- Priority resolutions: 32x32, 64x64, 128x128, plus small spritesheets that can be indexed.
- Prefer full-body characters with readable anatomy, then creatures, monsters, robots, items, weapons, vehicles, environments and effects.
- Preserve source provenance and license metadata for every collected asset.
- CC0/public-domain assets are the primary redistributable pool.
- Other licenses are tracked separately and are never mislabeled as CC0.
- Creator notices that prohibit AI/ML use are treated as exclusion rules.
- Sacred/religious exclusion filter is applied before ingestion. The filter is intentionally conservative: uncertain items are quarantined instead of being silently included.
- The dataset is for artistic/research collection; training is outside this repository's collection pipeline.

## Layout

`images/<type>/<gender>/<style>/<race>/<source>/<pose>/...`
`metadata/<asset-id>.json`
`manifests/dataset.jsonl`
`manifests/dataset.csv`
`manifests/quarantine.jsonl`

The collector can be expanded with additional source catalogs without changing the metadata schema.

## Current collection sources

The first automated collector uses a curated CC0 source pool from the public `Tiddybub/2d-assets` catalog plus explicitly verified CC0 OpenGameArt entries.

The Tiddybub catalog states that its included packs are CC0 and preserves SOURCE.md provenance for each pack. OpenGameArt records are checked individually before being added to the source catalog.

## Important

This repository records provenance; it does not magically change the copyright/license of an asset. Always retain the original source and license information.
