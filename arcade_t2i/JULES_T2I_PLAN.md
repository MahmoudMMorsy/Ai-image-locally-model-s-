# Jules Plan: Complete Neural Text-to-Image (64x64 Arcade)

**Assignee:** Google Jules (https://jules.google)
**Repo:** `MahmoudMMorsy/Ai-image-locally-model-s-`
**Branch (default):** `feature-pixel-art-sprite-engine-10984173554244238923`
**Work directory:** `arcade_t2i/`
**Labels:** `jules`, `t2i`, `text-to-image`

**Owner goal:** Real neural **text to image** (Arabic + English to readable 64x64 arcade sprites). Not HQ skip-mix only. Not empty blobs.

---

## CRITICAL DATA RULE (updated 2026-10-11)

**Train on ALL image files in the repo. Target N approximately 16,330.**

| Folder | ~Images (repo scan) |
|--------|---------------------|
| `nano arcked dat/` | **8,485** |
| `mm.trine/` | **5,109** |
| `nano_arcade_data/` | **2,496** |
| `examples/` | **199** |
| `All-data-Trine/` | **22** |
| `nano_pixel_mob/` + root outputs | **~19** |
| **TOTAL** | **approx 16,330** |

### Mandatory sources (do not skip)

1. `mm.trine/**` — all PNG/JPG
2. `nano arcked dat/**` — including:
   - `nano arcked dat/official_cc0/` (~7234)
   - `nano arcked dat/data/` (~1237) — NEW batch (warrior, animal, knight, mage, robot, rogue, dragon, ...)
   - class folders: knight, mage, robot, warrior, ...
3. `nano_arcade_data/**`
4. `examples/**` (usable sprites; log any exclusions)
5. `All-data-Trine/**` character PNG/sheets

### Forbidden

- Training only on `dataset_clean.zip` (101 images)
- Training only on bank subset (~4k) while ignoring repo PNGs
- Claiming full data if **N < 15,000** after indexing (allow small loss for corrupt files; report count)

**Hard stop:** If after Phase A, `N < 1000` then stop and comment on the issue.
**Owner expectation:** Aim for **N >= 15,000** usable 64x64 frames after filter.

---

## 0. Where to work

| Item | Path |
|------|------|
| Project root | `arcade_t2i/` |
| Plan (this file) | `arcade_t2i/JULES_T2I_PLAN.md` |
| Code | `arcade_t2i/*.py`, `bilingual_vocab.json` |
| Manifests | `arcade_t2i/manifests/` |
| Checkpoints | `arcade_t2i/checkpoints/` |
| Eval | `arcade_t2i/samples/` |
| Log | `arcade_t2i/TRAINING_LOG.md` |

---

## 1. What Grok already finished (do not redo from zero)

| Phase | Status | Artifacts |
|-------|--------|-----------|
| Weak manifest + AR/EN captions | Done (bank only ~4211) | Must be **replaced** by full-repo manifest |
| Bilingual vocab + text_lang | Done | `bilingual_vocab.json`, `text_lang.py` |
| Compact Text-CVAE | Started | `model_t2i.py`, early checkpoint |
| Quality gate | **FAIL** | Blobs — not shippable |

---

## 2. Success criteria (hard gate)

1. `فارس` / `knight` → readable character (not solid blob)
2. `ساحر` / `mage` visibly different from knight
3. `robot` / `روبوت` visibly different from both
4. Seed variation for same prompt
5. Grid: `arcade_t2i/samples/eval_accepted_YYYYMMDD.png` + `TRAINING_LOG.md`

---

## 3. Phases (execute in order)

### Phase A — Index + captions on ALL ~16,330 images (blocking)

1. Walk repo for every `.png/.jpg/.jpeg/.webp/.gif` under the folders listed above.
2. Filter: resize NEAREST to 64x64; drop near-zero opaque frames (log drops).
3. Write `arcade_t2i/manifests/manifest_full.jsonl` with path, class_id, class_name, caption_ar, caption_en, source_folder.
4. Quantize with palette48 → `arcade_t2i/checkpoints/indices_full_u8.pt` shape `[N,64,64]`.
5. PR must report: **N total, per-folder counts, class histogram**.
6. **Reject work if N < 15,000 without written justification** (corrupt files only).

Caption from path, e.g. `nano arcked dat/data/warrior/...` → محارب / warrior.

### Phase B — Text conditioning on full captions

1. Rebuild tokens from manifest_full captions.
2. Keep Arabic normalization in `text_lang.py`.
3. Optional contrastive loss across classes.

### Phase C — Train on full indices (N≈16k)

**C1:** Text-CVAE on entire `indices_full_u8.pt` + captions; eval every 5 epochs.
**C2:** If still blobs → discrete masked diffusion + text on **same full N**.

Do not downsample for the final reported run (smoke tests must be labeled smoke).

### Phase D — Package

1. `generate_t2i.py --prompt "..."` (AR/EN)
2. CPU README
3. Large `.pt` via GitHub Release
4. Accepted eval grid under `samples/`

### Phase E — Animation (only after T2I gate passes)

Same-character sequences from `mm.trine` hashes → GIF.

---

## 4. Eval prompts (log all)

`فارس` `knight` `ساحر` `mage` `روبوت` `robot` `قطة` `cat` `تنين` `dragon` `محارب` `warrior`

---

## 5. Coordination

- Grok: plan + quality gate
- Jules: implement A→D, PR with N and eval grids
- Never present HQ skip-mix as text-to-image
- PR title: `[jules][t2i] phase-X — N=16330 — eval ok/fail`

---

## 6. Expected tree

```
arcade_t2i/
  JULES_T2I_PLAN.md
  model_t2i.py train_lean.py train_t2i.py build_manifest.py
  text_lang.py bilingual_vocab.json auto_classify.py
  manifests/manifest_full.jsonl
  checkpoints/indices_full_u8.pt  palette48*.pt  t2i_*.pt
  samples/
  TRAINING_LOG.md
```
