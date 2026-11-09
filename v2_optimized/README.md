# Spriteo v2 Optimized ONNX Bundle

This directory contains the ONNX export of the spatial discrete autoencoder trained on the final verified single-sprite dataset.

- Input: RGBA tensor `[batch, 4, 64, 64]`, values normalized to `[0, 1]`.
- Encoder output: latent tensor `[batch, 64, 8, 8]`.
- Decoder inputs: latent tensor plus `class_id`.
- Decoder output: palette logits `[batch, 33, 64, 64]`.
- Palette: 32 colors plus transparent index `32`.
- ONNX opset: 18.
- Dataset: `spriteo_final_single_sprite_verified_2026-11-09`.

See `manifest.json` for class IDs, SHA-256 hashes, dimensions, and training metrics.
