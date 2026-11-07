# EXECUTION STATUS
Updated: 2026-11-07

Branch: **`unified-pipeline-exec`**

## Done (synced)

| Step | Result |
|------|--------|
| **0 Index** | `arcade_t2i/manifests/manifest_unified.jsonl` — **N=5104** from mm.trine tree, **111 unique char_hash** |
| **1 Indices** | `indices_unified_u8.pt` — **N=2000** quantized (2000 images downloaded; 3104 paths not on disk yet) |
| **2 Train** | `maskdiff_unified_latest.pt` — CE **0.441** after 10 epochs on **800** samples (RAM-limited subset) |

### Report
```json
{"manifest_N": 5104, "unique_hashes": 111, "indices_N": 2000, "train_n": 800, "ce": 0.4407}
```

## Honest limits
- Full 5104 image blobs not all local; indices built on first 2000 downloaded.
- Training subset 800 due to sandbox RAM (OOM at batch 64 on 2000).
- CE 0.44 is **not** ship gate for free-gen knight/mage/robot — continue train on full N elsewhere (Jules/GPU).

## Next steps (no conflict)
1. Jules: full checkout → re-run phase1 on all 5104 → phase2 longer.
2. Eval samples under `arcade_t2i/samples/`.
3. Do **not** mass-merge PRs 128–147.

## Commands
```bash
python arcade_t2i/phase0_build_manifest.py --repo-root .
python arcade_t2i/phase1_build_indices.py --repo-root .
python arcade_t2i/phase2_train_maskdiff.py --repo-root . --epochs 30
```
