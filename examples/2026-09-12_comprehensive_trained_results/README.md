# Comprehensive Model Fine-Tuning & Pixel Art Showcase (2026-09-12)

This directory contains showcase results and generated assets from fine-tuning all repository neural models on cleaned training datasets.

## 🌟 Model Specializations & Highlights

1. **Pixel Sprite Engine (`pixel_art_engine`)**:
   - Generator & Encoder fine-tuned for 64x64 character synthesis.
   - Ultra-fast CPU inference (<20ms per character frame).
   - Authentic retro palette quantization: **Signature 32-Color Palette**, **Game Boy (4 Green Shades)**, and **NES (16 Retro Colors)**.
   - 4-frame action sprite sheets (`_sheet.png`) and smooth looping GIFs (`_anim.gif`).

2. **Real Latent Diffusion ONNX (`models/real_diffusion_onnx`)**:
   - `RealLatentUNet` + `RealLatentDecoder` architecture exported to optimized ONNX (<50MB).
   - Instant iterative denoising in latent space (4, 32, 32) without GPU requirements.

3. **Bilingual Poster Engine (`poster_generator_256`)**:
   - `BilingualPosterEngine` & `ArabicPosterTextEmbedding` fine-tuned for 256x256 posters.
   - Perfect right-to-left Arabic text rendering (`arabic_reshaper` + `python-bidi`) and English typography layout.

## 🎨 Asset Summary
- **Character Sprites**: Knight, Wizard, Monster, Robot (PNG, Game Boy, NES).
- **Sprite Sheets & GIFs**: 4-frame animated running actions.
- **Bilingual Posters**: Cyberpunk, Knight Legend, Deep Space, Neon Samurai.
