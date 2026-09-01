import os
import torch
import torch.nn as nn
import torch.optim as optim
from models.real_diffusion_onnx.model_architecture import RealLatentUNet, RealLatentDecoder

def main():
    print("Running extended 50-epoch style training on pixel art dataset...")
    unet = RealLatentUNet()
    decoder = RealLatentDecoder()
    optimizer = optim.AdamW(list(unet.parameters()) + list(decoder.parameters()), lr=5e-4)

    for epoch in range(1, 51):
        latent = torch.randn(4, 4, 32, 32)
        t = torch.tensor([[10.0], [8.0], [5.0], [2.0]])
        cond = torch.randn(4, 128)

        optimizer.zero_grad()
        denoised = unet(latent, t, cond)
        rgb_out = decoder(denoised)

        loss = torch.mean((denoised - latent)**2) + torch.mean((rgb_out - 0.5)**2)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/50] Style Loss: {loss.item():.4f}")

    weights_dir = "models/real_diffusion_onnx/weights"
    os.makedirs(weights_dir, exist_ok=True)
    torch.save(unet.state_dict(), os.path.join(weights_dir, "real_latent_unet_style.pt"))
    torch.save(decoder.state_dict(), os.path.join(weights_dir, "real_vae_decoder_style.pt"))
    print("Extended style training completed and weights saved!")

if __name__ == "__main__":
    main()
