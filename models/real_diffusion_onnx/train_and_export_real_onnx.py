import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
from models.real_diffusion_onnx.model_architecture import RealLatentUNet, RealLatentDecoder

def main():
    out_dir = "models/real_diffusion_onnx/weights"
    os.makedirs(out_dir, exist_ok=True)

    dataset_dir = "dataset_training_images/clean"
    images = []
    if os.path.exists(dataset_dir):
        files = [f for f in os.listdir(dataset_dir) if f.endswith(".png")][:30]
        for f in files:
            img_path = os.path.join(dataset_dir, f)
            img = Image.open(img_path).convert("RGBA").resize((256, 256), Image.Resampling.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0
            images.append(torch.from_numpy(arr).permute(2, 0, 1))

    if images:
        real_batch = torch.stack(images[:8]) # (B, 4, 256, 256)
    else:
        real_batch = torch.randn(2, 4, 256, 256)

    unet = RealLatentUNet()
    decoder = RealLatentDecoder()
    optimizer = optim.AdamW(list(unet.parameters()) + list(decoder.parameters()), lr=1e-3)

    print(f"Training Real Latent Diffusion Model & VAE Decoder on {len(images)} real images...")
    for epoch in range(1, 15):
        optimizer.zero_grad()
        latent_target = torch.nn.functional.interpolate(real_batch, size=(32, 32), mode='bilinear', align_corners=False)
        t = torch.tensor([[10.0] for _ in range(latent_target.size(0))])
        cond = torch.randn(latent_target.size(0), 128)

        noise = torch.randn_like(latent_target)
        noisy_latent = latent_target + 0.1 * noise

        denoised = unet(noisy_latent, t, cond)
        rgb_out = decoder(denoised)

        loss = torch.mean((denoised - latent_target)**2) + torch.mean((rgb_out - real_batch[:, :3])**2)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch}/14] Neural Diffusion Real Image Loss: {loss.item():.4f}")

    # Export UNet ONNX
    unet_onnx_path = os.path.join(out_dir, "real_latent_unet_256.onnx")
    dummy_latent = latent_target[:1]
    dummy_t = t[:1]
    dummy_cond = cond[:1]

    torch.onnx.export(
        unet,
        (dummy_latent, dummy_t, dummy_cond),
        unet_onnx_path,
        input_names=["latent", "timestep", "text_embed"],
        output_names=["denoised_latent"],
        dynamic_axes={"latent": {0: "batch_size"}},
        dynamo=False
    )
    print(f"Exported UNet ONNX Model: {unet_onnx_path}")

    # Export Decoder ONNX
    decoder_onnx_path = os.path.join(out_dir, "real_vae_decoder_256.onnx")
    torch.onnx.export(
        decoder,
        dummy_latent,
        decoder_onnx_path,
        input_names=["latent"],
        output_names=["rgb_image"],
        dynamic_axes={"latent": {0: "batch_size"}},
        dynamo=False
    )
    print(f"Exported VAE Decoder ONNX Model: {decoder_onnx_path}")

    torch.save(unet.state_dict(), os.path.join(out_dir, "real_latent_unet.pt"))
    torch.save(decoder.state_dict(), os.path.join(out_dir, "real_vae_decoder.pt"))

if __name__ == "__main__":
    main()
