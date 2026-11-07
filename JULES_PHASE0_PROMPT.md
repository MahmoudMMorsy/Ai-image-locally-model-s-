# Jules — Phase 0 only

```
Read UNIFIED_PIPELINE.md, AGENTS.md, EXECUTION_STATUS.md, arcade_t2i/JULES_T2I_PLAN.md.

Branch: unified-pipeline-exec (or open PR into it).
Work dir: repo root.

Task Phase 0 ONLY:
1. python arcade_t2i/phase0_build_manifest.py --repo-root .
2. Confirm N and per_folder in arcade_t2i/manifests/manifest_unified_report.json
3. Update EXECUTION_STATUS.md with real counts
4. Do NOT train yet. Do NOT use dataset_clean alone.
5. PR title: [jules][t2i] phase-0 — N=<count> — index only
Hard stop if N < 1000.
```
