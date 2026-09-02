
## Pipeline Execution - 2026-08-29 18:24:06
- Fine-tuned Real Latent UNet + VAE Decoder
- Exported ONNX models: `models/real_diffusion_onnx/weights/real_latent_unet_256.onnx` and `models/real_diffusion_onnx/weights/real_vae_decoder_256.onnx`
- Verified Android ONNX Mobile Assets & Bilingual Tokenizer

## Pipeline Execution - 2026-09-02 13:48:43
- Fine-tuned Real Latent UNet + VAE Decoder on real cleaned pixel art dataset (10 epochs)
- Exported updated ONNX models: `models/real_diffusion_onnx/weights/real_latent_unet_256.onnx` and `models/real_diffusion_onnx/weights/real_vae_decoder_256.onnx`
- Saved updated PyTorch weights in `models/real_diffusion_onnx/weights/`
