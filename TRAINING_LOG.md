
## Pipeline Execution - 2026-09-07 18:39:28
- **Models Fine-Tuned & Exported**:
  - `RealLatentUNet` & `RealLatentDecoder` -> ONNX (`models/real_diffusion_onnx/weights/real_latent_unet_256.onnx`, `models/real_diffusion_onnx/weights/real_vae_decoder_256.onnx`)
  - `NanoPixel-v1` -> ONNX (`models/nano_pixel_art_v1/weights/nanopixel_v1.onnx`, size: 5.25 MB)
  - `NanoPixel XL0_2` -> Checkpoint (`models/nano_pixel_XL0_2/weights/nanopixel_XL0_2.pt`)
  - `NanoPixel 3A XL` -> Checkpoint (`models/nano_pixel_3A_xl/weights/nanopixel_3A_xl.pt`)
  - `Poster Generator 256` -> ONNX (`poster_generator_256/weights/poster_generator_256.onnx`)
- **Dataset Images Processed**: 50 clean training sprite images
- **Game Boy / NES Quantization & Animations**: Generated 4 character archetypes (knight, wizard, monster, robot) with 4-frame animated GIFs and action sprite sheets.
- **Bilingual Poster Generated**: `examples/2026-09-02_pixel_art_showcase/bilingual_poster_256.png`
