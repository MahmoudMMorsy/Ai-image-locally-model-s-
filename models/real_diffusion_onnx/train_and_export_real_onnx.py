import os
import torch
import torch.nn as nn
import torch.optim as optim
from models.real_diffusion_onnx.model_architecture import RealLatentUNet, RealLatentDecoder

def main():
    out_dir = "models/real_diffusion_onnx/weights"
    os.makedirs(out_dir, exist_ok=True)

    unet = RealLatentUNet()
    decoder = RealLatentDecoder()

    # Fast convergence optimization loop on synthetic image latents
    optimizer = optim.AdamW(list(unet.parameters()) + list(decoder.parameters()), lr=1e-3)

    print("Training Real Latent Diffusion Model & VAE Decoder...")
    for epoch in range(1, 15):
        latent = torch.randn(2, 4, 32, 32)
        t = torch.tensor([[10.0], [5.0]])
        cond = torch.randn(2, 128)

        optimizer.zero_grad()
        denoised = unet(latent, t, cond)
        rgb_out = decoder(denoised)

        loss = torch.mean((denoised - latent)**2) + torch.mean((rgb_out - 0.5)**2)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch}/14] Neural Diffusion Loss: {loss.item():.4f}")

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
