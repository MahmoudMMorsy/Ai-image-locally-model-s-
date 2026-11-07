# EXECUTION STATUS
Updated: 2026-11-07

Branch: **`unified-pipeline-exec`**

## Completed tooling (synced)
| Step | Artifact | Status |
|------|----------|--------|
| 0 Index script | `arcade_t2i/phase0_build_manifest.py` | ready (syntax fixed) |
| 1 Indices | `arcade_t2i/phase1_build_indices.py` | ready |
| 2 Train skeleton | `arcade_t2i/phase2_train_maskdiff.py` | ready |
| Docs | `UNIFIED_PIPELINE.md`, `JULES_PHASE0_PROMPT.md` | on branch |

## Verified counts (git tree, not local files)
- **mm.trine ≈ 5109 PNG**

## Run order (full checkout required for 0→1→2)
```bash
# clone branch
git clone -b unified-pipeline-exec --depth 1 <repo-url>
cd Ai-image-locally-model-s-

python arcade_t2i/phase0_build_manifest.py --repo-root .
python arcade_t2i/phase1_build_indices.py --repo-root .
python arcade_t2i/phase2_train_maskdiff.py --repo-root . --epochs 10
```

## Not done yet (needs full image blobs)
- Actual `manifest_unified.jsonl` with N≥1000 written from local files
- `indices_unified_u8.pt`
- Trained `maskdiff_unified_latest.pt` on full N
- Gate samples knight/mage/robot

## Conflict policy
Do **not** mass-merge open PRs #128–#147. Keep unified work on this branch only.
