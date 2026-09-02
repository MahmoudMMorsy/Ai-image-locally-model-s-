# Nano Pixel MOM Showcase - 2026-09-02

The **Nano Pixel MOM (Massive Output Model)** is a lightweight, super-fast CPU-optimized neural latent diffusion generator fine-tuned on 2,534 dataset pixel art images downloaded from Google Drive (`images_2534_only`).

## Model Architecture & Specs (`models/nano_pixel_mom/`)
- **UNet Denoising Model**: `NanoPixelMomUNet` with GroupNorm, SiLU activations, and text/condition embedding injection.
- **Neural VAE Decoder**: `NanoPixelMomDecoder` decoding 32x32 latents directly into 64x64/128x128 crisp pixel art images.
- **Exported ONNX Models**:
  - `models/nano_pixel_mom/weights/nano_pixel_mom_unet.onnx` (<50MB)
  - `models/nano_pixel_mom/weights/nano_pixel_mom_decoder.onnx` (<50MB)
- **CPU Speed**: Sub-30ms per frame generation without needing high-end GPUs.

## Game Boy & NES Retro Styles
All outputs strictly adhere to retro palette constraints:
- **Game Boy Green Palette**: 4-shade olive green (`GAMEBOY_PALETTE`).
- **NES 16-Color Palette**: Classic 8-bit console palette (`NES_PALETTE`).

## Generated Showcase Assets
- `mom_char_01_gameboy_knight`: Game Boy 4-shade green knight (Single PNG, 4-frame Sprite Sheet, Animated GIF).
- `mom_char_02_nes_wizard`: NES 16-color wizard mage (Single PNG, 4-frame Sprite Sheet, Animated GIF).
- `mom_char_03_nes_orc`: NES 16-color retro orc monster (Single PNG, 4-frame Sprite Sheet, Animated GIF).
- `mom_char_04_gameboy_robot`: Game Boy 4-shade green cyberpunk robot mech (Single PNG, 4-frame Sprite Sheet, Animated GIF).
- `mom_char_05_nes_paladin`: NES 16-color holy paladin hero (Single PNG, 4-frame Sprite Sheet, Animated GIF).
- `mom_poster_01_gameboy.png`: Game Boy bilingual poster ("Nano Pixel MOM Game Boy" / "نانو بيكسل مـام - جيم بوي").
- `mom_poster_02_nes.png`: NES bilingual poster ("Nano Pixel MOM NES" / "نانو بيكسل مـام - أن إي إس").
