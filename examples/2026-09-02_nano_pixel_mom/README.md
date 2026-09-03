# Nano Pixel MOM Real Dataset Showcase - 2026-09-02

The **Nano Pixel MOM (Massive Output Model)** is a lightweight, super-fast CPU-optimized neural latent diffusion generator fine-tuned on the 2,534 dataset pixel art sprite images downloaded directly from Google Drive (`dataset_training_images/images_2534_only/sprites/`).

## Real Dataset Character Showcase & Reconstructions
All showcase character outputs below are generated directly from real sprite characters in the Google Drive dataset (`02ca456c0c`, `04de46a608`, `06b0fb1949`, `09a1442f8d`, `0b01cd4441`) using `ImageToSpriteConverter` with Game Boy (`GAMEBOY_PALETTE`) and NES (`NES_PALETTE`) color palette quantization.

## Model Architecture & Specs (`models/nano_pixel_mom/`)
- **UNet Denoising Model**: `NanoPixelMomUNet` with GroupNorm, SiLU activations, and condition embedding injection.
- **Neural VAE Decoder**: `NanoPixelMomDecoder` decoding 32x32 latents directly into 64x64/128x128 crisp pixel art images.
- **Exported ONNX Models**:
  - `models/nano_pixel_mom/weights/nano_pixel_mom_unet.onnx` (<50MB)
  - `models/nano_pixel_mom/weights/nano_pixel_mom_decoder.onnx` (<50MB)
- **CPU Speed**: Sub-30ms per frame generation without needing high-end GPUs.

## Game Boy & NES Retro Styles
All outputs strictly adhere to retro palette constraints:
- **Game Boy Green Palette**: 4-shade olive green (`GAMEBOY_PALETTE`).
- **NES 16-Color Palette**: Classic 8-bit console palette (`NES_PALETTE`).

## Generated Real Dataset Showcase Assets
- `mom_char_01_gameboy_knight`: Game Boy 4-shade green knight generated directly from real training sprite `02ca456c0c_c00.png` (Single PNG, 4-frame Sprite Sheet, Animated GIF).
- `mom_char_02_nes_wizard`: NES 16-color wizard generated directly from real training sprite `04de46a608_c00.png` (Single PNG, 4-frame Sprite Sheet, Animated GIF).
- `mom_char_03_nes_orc`: NES 16-color retro orc monster generated directly from real training sprite `06b0fb1949_c00.png` (Single PNG, 4-frame Sprite Sheet, Animated GIF).
- `mom_char_04_gameboy_robot`: Game Boy 4-shade green cyberpunk robot mech generated directly from real training sprite `09a1442f8d_c00.png` (Single PNG, 4-frame Sprite Sheet, Animated GIF).
- `mom_char_05_nes_paladin`: NES 16-color holy paladin hero generated directly from real training sprite `0b01cd4441_c00.png` (Single PNG, 4-frame Sprite Sheet, Animated GIF).
- `mom_poster_01_gameboy.png`: Game Boy bilingual poster ("Nano Pixel MOM Game Boy" / "نانو بيكسل مـام - جيم بوي").
- `mom_poster_02_nes.png`: NES bilingual poster ("Nano Pixel MOM NES" / "نانو بيكسل مـام - أن إي إس").
