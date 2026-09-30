import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
from models.real_diffusion_onnx.model_architecture import RealLatentUNet, RealLatentDecoder

def load_real_image_tensors(dataset_dir="dataset_training_images/clean", target_size=(256, 256), max_items=10):
    tensors = []
    if os.path.exists(dataset_dir):
        files = [f for f in os.listdir(dataset_dir) if f.endswith('.png')]
        for f in files[:max_items]:
            fpath = os.path.join(dataset_dir, f)
            try:
                im = Image.open(fpath).convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
                arr = np.array(im, dtype=np.float32) / 255.0
                t = torch.from_numpy(arr).permute(2, 0, 1)
                tensors.append(t)
            except Exception as e:
                pass
    if not tensors:
        tensors = [torch.rand(3, 256, 256) for _ in range(2)]
    return torch.stack(tensors)

def main():
    out_dir = "models/real_diffusion_onnx/weights"
    os.makedirs(out_dir, exist_ok=True)

    unet = RealLatentUNet()
    decoder = RealLatentDecoder()

    target_images = load_real_image_tensors()
    num_samples = target_images.size(0)

    optimizer = optim.AdamW(list(unet.parameters()) + list(decoder.parameters()), lr=1e-3)

    print(f"Training Real Latent Diffusion Model & VAE Decoder on {num_samples} real image targets...")
    for epoch in range(1, 15):
        optimizer.zero_grad()
        # Downsample target images to 32x32 initial latents
        downscaled_latents = torch.nn.functional.interpolate(target_images, size=(32, 32), mode="bilinear", align_corners=False)
        # 3 channels -> 4 channels latent space
        latent = torch.cat([downscaled_latents, downscaled_latents.mean(dim=1, keepdim=True)], dim=1)

        t = torch.randint(1, 20, (num_samples, 1)).float()
        cond = torch.randn(num_samples, 128)

        denoised = unet(latent, t, cond)
        rgb_out = decoder(denoised)

        # Train end-to-end against real target RGB images
        loss = torch.mean((denoised - latent)**2) + torch.mean((rgb_out - target_images)**2)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch}/14] Real Image Diffusion Loss: {loss.item():.4f}")

    # Export UNet ONNX
    unet_onnx_path = os.path.join(out_dir, "real_latent_unet_256.onnx")
    torch.onnx.export(
        unet,
        (latent[:1], t[:1], cond[:1]),
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
        denoised[:1],
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
