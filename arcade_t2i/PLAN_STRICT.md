# PLAN_STRICT — Full-N Text-to-Image

## Hard rules
1. Train on **all** `indices_full_u8.pt` (N=15755), not a 2k subset.
2. Bilingual AR/EN captions from `manifest_full.jsonl`.
3. Quality gate: generated knight/mage/robot/cat/dragon must be **recognizable characters**. Noise/blobs = FAIL.
4. Do not claim PASS without visual proof in `samples/`.

## Models
- **TextCVAE** (`model_t2i.py`, `train_strict.py`): reconstruction + KL; generate via class latent bank.
- **MaskDiff**: discrete masked prediction of palette indices.

## Acceptance
- [ ] recon shows clear structure (face/body)
- [ ] text prompts «فارس / knight / ساحر / robot» produce readable sprites
- [ ] samples committed under `arcade_t2i/samples/`

## Jules
Work only under `arcade_t2i/`. Use full N. Update STATUS.md after each run.
