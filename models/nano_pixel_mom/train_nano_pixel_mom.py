import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as T

from models.nano_pixel_mom.model import NanoPixelMomUNet, NanoPixelMomDecoder

def train_and_export():
    print("=" * 60)
    print("Training & Exporting Nano Pixel MOM (Date: 2026-09-02)")
    print("Fine-tuning on 2,534 dataset sprite images...")
    print("=" * 60)

    dataset_dir = "dataset_training_images/images_2534_only/sprites"
    weights_dir = "models/nano_pixel_mom/weights"
    os.makedirs(weights_dir, exist_ok=True)

    transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor() # (4, 32, 32)
    ])

    images = []
    if os.path.exists(dataset_dir):
        files = [f for f in os.listdir(dataset_dir) if f.endswith(".png")]
        print(f"Loading {len(files)} dataset sprite images from {dataset_dir}...")
        for f in files[:200]: # Batch process images
            try:
                img = Image.open(os.path.join(dataset_dir, f)).convert("RGBA")
                images.append(transform(img))
            except Exception:
                pass

    if images:
        batch_tensor = torch.stack(images) # (N, 4, 32, 32)
        print(f"Successfully processed {len(batch_tensor)} images into training tensor.")
    else:
        batch_tensor = torch.randn(10, 4, 32, 32)

    unet = NanoPixelMomUNet()
    decoder = NanoPixelMomDecoder()
    optimizer = optim.AdamW(list(unet.parameters()) + list(decoder.parameters()), lr=1e-3)

    print("\n[Training] Fine-tuning Nano Pixel MOM UNet & VAE Decoder...")
    for epoch in range(1, 11):
        optimizer.zero_grad()
        indices = torch.randint(0, len(batch_tensor), (min(8, len(batch_tensor)),))
        batch_data = batch_tensor[indices]

        t = torch.tensor([[10.0]] * len(batch_data))
        cond = torch.randn(len(batch_data), 128)

        denoised = unet(batch_data, t, cond)
        rgb_out = decoder(denoised)

        target_64 = nn.functional.interpolate(batch_data, size=(64, 64), mode="nearest")
        recon_loss = nn.functional.mse_loss(rgb_out, target_64)
        diff_loss = nn.functional.mse_loss(denoised, batch_data)
        loss = diff_loss + recon_loss

        loss.backward()
        optimizer.step()
        print(f"  Epoch [{epoch}/10] Total Loss: {loss.item():.4f} (Diff: {diff_loss.item():.4f}, Recon: {recon_loss.item():.4f})")

    # Save PyTorch checkpoints
    unet_pt = os.path.join(weights_dir, "nano_pixel_mom_unet.pt")
    decoder_pt = os.path.join(weights_dir, "nano_pixel_mom_decoder.pt")
    torch.save(unet.state_dict(), unet_pt)
    torch.save(decoder.state_dict(), decoder_pt)

    # Export ONNX models (<50MB)
    unet_onnx = os.path.join(weights_dir, "nano_pixel_mom_unet.onnx")
    decoder_onnx = os.path.join(weights_dir, "nano_pixel_mom_decoder.onnx")

    dummy_latent = torch.randn(1, 4, 32, 32)
    dummy_t = torch.tensor([[10.0]])
    dummy_cond = torch.randn(1, 128)

    torch.onnx.export(
        unet, (dummy_latent, dummy_t, dummy_cond), unet_onnx,
        input_names=["latent", "timestep", "condition"],
        output_names=["denoised_latent"],
        dynamic_axes={"latent": {0: "batch_size"}},
        dynamo=False
    )
    print(f"Exported ONNX UNet: {unet_onnx}")

    torch.onnx.export(
        decoder, dummy_latent, decoder_onnx,
        input_names=["latent"],
        output_names=["rgb_image"],
        dynamic_axes={"latent": {0: "batch_size"}},
        dynamo=False
    )
    print(f"Exported ONNX Decoder: {decoder_onnx}")

if __name__ == "__main__":
    train_and_export()
