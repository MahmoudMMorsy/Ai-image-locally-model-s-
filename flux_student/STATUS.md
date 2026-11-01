# STATUS — flux_student
Updated: 2026-11-02

## Disk-safe sync
- Local project ~11MB — no disk choke
- Branch `flux-student`: code + manifest.jsonl + STATUS
- PNGs stay local under data/resized + samples; base64 sidecar when needed
- continuous_loop keeps training

## Counts
- manifest: 26 teacher images (sd-turbo + prompt)
- Student ~10M params, loss ~0.37-0.39

## Policy
1. Train continuously locally
2. Push manifest after every +10 new teacher images
3. Do not clone full image trees into sandbox
