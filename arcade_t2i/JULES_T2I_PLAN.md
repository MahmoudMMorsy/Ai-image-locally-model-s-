# Jules Plan: Complete Neural Text-to-Image (64x64 Arcade)

**Assignee:** Google Jules (https://jules.google)
**Repo:** `MahmoudMMorsy/Ai-image-locally-model-s-`
**Branch (default):** `feature-pixel-art-sprite-engine-10984173554244238923`
**Work directory:** `arcade_t2i/`
**Labels:** `jules`, `t2i`, `text-to-image`

**Owner goal:** Real neural **text to image** (Arabic + English prompts to readable 64x64 arcade sprites). Not HQ skip-mix only. Not empty blobs.

---

## 0. Where to work

| Item | Path |
|------|------|
| Project root for this track | `arcade_t2i/` |
| Code (Grok handoff) | `arcade_t2i/*.py`, `arcade_t2i/bilingual_vocab.json` |
| Manifests | `arcade_t2i/manifests/` |
| Checkpoints | `arcade_t2i/checkpoints/` |
| Eval grids | `arcade_t2i/samples/` |
| Full image data (required) | `mm.trine/`, `nano arcked dat/`, `nano_arcade_data/` |

**Rule:** Production training must use **>=1000 images** (prefer full `mm.trine` ~5100 + classified sets). Never train only on 101-image smoke sets.

---

## 1. What Grok already finished (do not redo from zero)

| Phase | Status | Artifacts |
|-------|--------|-----------|
| Manifest skeleton + AR/EN captions | Done (weak captions on bank) | `manifests/manifest.jsonl` (~4211 rows) |
| Bilingual vocab + text_lang | Done | `bilingual_vocab.json`, `text_lang.py` |
| Compact Text-CVAE + lean train | Started | `model_t2i.py`, `train_lean.py`, `t2i_cvae_best.pt` |
| Fixed prompt eval | Started | `samples/T2I_EVAL_NOW.png` |
| Quality gate | **Failed so far** | Early samples = blobs, not clear characters |

**Current checkpoint is NOT shippable.** Continue training + fix data coupling.

---

## 2. Success criteria (hard gate)

Accept a checkpoint only if:

1. Prompt `فارس` / `knight` yields a readable character silhouette (not a solid blob).
2. Prompt `ساحر` / `mage` is visibly different from knight.
3. Prompt `robot` / `روبوت` is visibly different from both.
4. Same prompt, different seeds show variation.
5. Save grid under `arcade_t2i/samples/eval_accepted_YYYYMMDD.png` and log in `TRAINING_LOG.md`.

If failed, do not claim text-to-image works.

---

## 3. Remaining phases (execute in order)

### Phase A — Real captions on full data (blocking)

1. Index all PNGs under `mm.trine/` and class folders in `nano arcked dat/`.
2. Write `arcade_t2i/manifests/manifest_full.jsonl` with path, class_id, caption_ar, caption_en.
3. Quantize with palette48 to `arcade_t2i/checkpoints/indices_full_u8.pt` shape [N,64,64].
4. Report N and class histogram in the PR. **Stop if N < 1000.**

### Phase B — Stronger text conditioning

1. Rebuild token tables from actual captions in manifest_full.
2. Keep Arabic normalization in `text_lang.py`.
3. Optional contrastive loss: text closer to matching class latent than random.

### Phase C — Train generator

**C1:** Resume Text-CVAE (`train_lean.py`) on manifest_full; eval every 5 epochs.

**C2 (if C1 stays blob-like):** Discrete masked diffusion conditioned on text embedding + class_id. Prefer readable silhouettes over low CE alone.

### Phase D — Package

1. `generate_t2i.py --prompt "..."` (AR or EN).
2. CPU inference README.
3. Large `.pt` via GitHub Release, not giant git commits.
4. Accepted sample grid under `arcade_t2i/samples/`.

### Phase E — Animation (only after T2I gate passes)

Same-character pose sequences from `mm.trine` hashes to GIF.

---

## 4. Eval prompts (must log all)

`فارس` `knight` `ساحر` `mage` `روبوت` `robot` `قطة` `cat` `تنين` `dragon`

---

## 5. Coordination

- Grok: plan + handoff + quality gate.
- Jules: implement A to D, open PR, attach eval grids.
- Do not present HQ skip-mix as text-to-image.
- PR title format: `[jules][t2i] phase-X — N=... — eval ok/fail`

## 6. Expected tree

```
arcade_t2i/
  JULES_T2I_PLAN.md
  model_t2i.py train_lean.py train_t2i.py build_manifest.py
  text_lang.py bilingual_vocab.json auto_classify.py
  manifests/ checkpoints/ samples/
  TRAINING_LOG.md
```
