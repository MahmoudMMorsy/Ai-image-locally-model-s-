
## Pipeline Execution - 2026-09-22_22-17
- Generated & Trained on 5,000 64x64 side-view arcade character sprite images.
- Quantized with Game Boy, NES, and Signature retro palettes.
- Exported ONNX models: `models/real_diffusion_onnx/weights/real_latent_unet_256.onnx` and `models/real_diffusion_onnx/weights/real_vae_decoder_256.onnx`.
- Showcase assets saved to `examples/2026-09-22_22-17_sideview_arcade_showcase`.

## Pipeline Execution - 2026-09-22_22-22
- Generated & Trained on 5,000 64x64 side-view arcade character sprite images loaded via PyTorch DataLoader.
- Computed real MSE image reconstruction & latent consistency loss during fine-tuning.
- Quantized with Game Boy, NES, and Signature retro palettes.
- Exported ONNX models: `models/real_diffusion_onnx/weights/real_latent_unet_256.onnx` and `models/real_diffusion_onnx/weights/real_vae_decoder_256.onnx`.
- Showcase assets saved to `examples/2026-09-22_22-22_sideview_arcade_showcase`.
