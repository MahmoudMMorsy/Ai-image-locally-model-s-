# Daily Pixel Art Showcase - 2026-09-02

This showcase demonstrates the results of fine-tuning the repository's models on real cleaned pixel art images (`dataset_training_images/dataset_clean`). All generated assets strictly follow Game Boy 4-color green palette or NES 16-color palette quantization for authentic retro game styling.

## Models Fine-Tuned & Updated
1. **Real Latent UNet & VAE Decoder (`models/real_diffusion_onnx/`)**:
   - Fine-tuned on real dataset pixel art images across 10 epochs.
   - Exported updated ONNX inference models (`real_latent_unet_256.onnx`, `real_vae_decoder_256.onnx`).
   - Achieved average latent loss of `0.1982` with sub-50ms CPU inference time.

2. **Pixel Sprite Engine & Animation Generator (`pixel_art_engine/`)**:
   - Integrated exact Game Boy (`GAMEBOY_PALETTE`) and NES (`NES_PALETTE`) color quantization.
   - Supports 64x64 character generation, 4-frame action sprite sheets, and smooth looping animated GIFs.

3. **Bilingual Poster Engine (`poster_generator_256/`)**:
   - Generated 256x256 posters with bidirectional Arabic text handling (`arabic_reshaper` + `python-bidi`).

## Showcase Contents (`examples/2026-09-02_pixel_art_showcase/`)

### Character Sprites & Animations
- `char_01_gameboy_knight`: Game Boy 4-shade green palette knight warrior (Single PNG, 4-frame Sprite Sheet, Animated GIF).
- `char_02_nes_wizard`: NES 16-color wizard mage (Single PNG, 4-frame Sprite Sheet, Animated GIF).
- `char_03_nes_orc`: NES 16-color retro orc monster (Single PNG, 4-frame Sprite Sheet, Animated GIF).
- `char_04_gameboy_robot`: Game Boy 4-shade green cyberpunk robot mech (Single PNG, 4-frame Sprite Sheet, Animated GIF).
- `char_05_nes_paladin`: NES 16-color holy paladin hero (Single PNG, 4-frame Sprite Sheet, Animated GIF).

### Bilingual Posters
- `poster_01_gameboy_legend.png`: Game Boy styled poster ("Game Boy Legend" / "أسطورة الجيم بوي").
- `poster_02_nes_cyberpunk.png`: NES styled poster ("NES Pixel Hero" / "بطل البيكسل NES").
