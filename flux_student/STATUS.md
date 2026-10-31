# flux_student STATUS

Updated: 2026-10-30 (continuous run)

## Done
- Branch `flux-student`
- StudentT2I architecture (base=48 ~10M params)
- Teacher: 20 images via sd-turbo (FLUX pending resource)
- Training with augmentation (flip/jitter): loss dropped ~0.95 → improving
- Scripts on repo: model, train, generate, prompts, README

## In progress
- Longer CPU training (60 epochs + aug)
- Sample dumps under local `samples/`

## Blockers
- pip/HF intermittent 502 → cannot expand teacher set this session easily
- Quality still limited by n=20 teacher pairs

## Next (auto)
1. Finish current train job → save checkpoint
2. Retry teacher expansion when network allows
3. Raise base to 96 after n≥100
4. Optional mix of repo arcade sprites (mm.trine) for style phase
