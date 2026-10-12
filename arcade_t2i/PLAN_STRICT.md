# PLAN_STRICT — arcade_t2i quality recovery (2026-10-13)

## Problem (verified on repo)
- Jules Phase D claimed PASS with N=15,755.
- Actual `samples/eval_accepted_20261011.png` is **colored noise**, not readable sprites.
- Root causes:
  1. `train_t2i.py` used only a **random Subset of 2000** samples (not full N).
  2. Early stop at loss ~2.4 (weak for 48-way CE).
  3. Eval gate accepted noise (no structural checks).

## Goal
Readable 64x64 class-distinct sprites from AR/EN prompts on CPU, trained on **full** `indices_full_u8.pt` (N=15755).

## Steps
1. Train on **all 15,755** indices + bilingual captions (no 2k subset).
2. KL free-bits / low beta so reconstruction learns structure first.
3. Strict eval every N epochs (unique colors, variance, class separation).
4. Class-conditional z from encoded training latents (better than pure N(0,1)).
5. Push improved checkpoint + honest samples + TRAINING_LOG entry.

## Status (Grok 2026-10-13)
- Downloaded repo weights + indices N=15755.
- Jules recon CE ~2.65 (barely reconstructs).
- Strict full-N training started; epoch1 loss ~1.28 (down from 2.44).
- Batch=4 required (batch 16 OOM on this CPU sandbox).
- Do **not** mark PASS until samples are readable characters.

## Success gate
- knight / mage / robot visibly different silhouettes
- Not random static
- Document loss + metrics honestly
