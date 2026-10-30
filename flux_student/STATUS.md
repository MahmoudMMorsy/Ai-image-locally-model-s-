# flux_student STATUS

- Branch: flux-student
- Student architecture: StudentT2I (UNet-like text-conditioned denoiser)
- First local run: 8 teacher images (sd-turbo), 30 epochs base=48, loss ~1.0 (data-starved)
- Next: scale teacher set to 100+, then base=96
- FLUX teacher: when disk/RAM allow --teacher flux
- Target sizes: 32 / 64 / 128 / 256 only
