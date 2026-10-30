# flux_student STATUS

Updated: 2026-10-30

## Done
- Branch `flux-student` on repo
- Architecture `StudentT2I` (base=48 ~10M / base=96 ~27M)
- Teacher pipeline: sd-turbo fallback (FLUX when resources allow)
- Generated **20** teacher images → resized 32/64/128/256
- Student trained **20 epochs** on 20 samples (CPU), loss ~0.98

## Honest quality
Still **data-starved**. Not usable for real characters yet. Need 500–5000+ teacher pairs before meaningful samples.

## Next on this branch
1. Scale teacher generation (sd-turbo / FLUX)
2. Train base=96 when n_data > 200
3. Optional: mix mm.trine arcade sprites for style phase
4. Export checkpoint + LoRA hook for user fine-tune

## Paths
- Code: `flux_student/`
- Local run (sandbox): `artifacts/flux_student/`
