import os
import glob
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.nano_pixel_mom.model import NanoPixelMomUNet, NanoPixelMomDecoder

class RealSpriteDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

def train():
    model_dir = "models/nano_pixel_mom"
    dataset_dir = "dataset_training_images/clean/images_2534_only/sprites"
    weights_dir = os.path.join(model_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    image_paths = glob.glob(os.path.join(dataset_dir, "*.png"))
    print(f"Loaded {len(image_paths)} real training images from {dataset_dir}")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = RealSpriteDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    unet = NanoPixelMomUNet()
    decoder = NanoPixelMomDecoder()

    # Simple image encoder to project real image to latent space
    encoder = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # 128 -> 64
        nn.GELU(),
        nn.Conv2d(32, 4, kernel_size=3, padding=1)
    )

    optimizer = optim.AdamW(list(unet.parameters()) + list(decoder.parameters()) + list(encoder.parameters()), lr=1e-3)
    criterion = nn.MSELoss()

    print("\n[nano_pixel_mom] Training UNet + VAE Decoder on real dataset...")
    num_epochs = 3
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        batches = 0
        for imgs in dataloader:
            batch_size = imgs.shape[0]

            # Encode real image to latent space (batch_size, 4, 64, 64)
            clean_latent = encoder(imgs)

            # Add diffusion noise
            timesteps = torch.rand(batch_size) * 10.0
            noise = torch.randn_like(clean_latent)
            noisy_latent = clean_latent + 0.1 * noise

            optimizer.zero_grad()
            denoised_latent = unet(noisy_latent, timesteps)
            rgb_out = decoder(denoised_latent)

            # Dual reconstruction and latent denoising loss
            loss = criterion(denoised_latent, clean_latent) + criterion(rgb_out, imgs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        avg_loss = running_loss / max(batches, 1)
        print(f"  Epoch [{epoch}/{num_epochs}] - Average Loss: {avg_loss:.6f}")

    # Save PyTorch checkpoints
    unet_pt = os.path.join(weights_dir, "nano_pixel_mom_unet.pt")
    decoder_pt = os.path.join(weights_dir, "nano_pixel_mom_decoder.pt")
    torch.save(unet.state_dict(), unet_pt)
    torch.save(decoder.state_dict(), decoder_pt)
    print(f"PyTorch weights saved to {unet_pt} and {decoder_pt}")

    # Export ONNX model
    onnx_path = os.path.join(weights_dir, "nano_pixel_mom.onnx")
    dummy_latent = torch.randn(1, 4, 64, 64)
    dummy_t = torch.tensor([5.0])

    class CombinedPipeline(nn.Module):
        def __init__(self, u, d):
            super().__init__()
            self.unet = u
            self.decoder = d
        def forward(self, latent, t):
            denoised = self.unet(latent, t)
            return self.decoder(denoised)

    combined_model = CombinedPipeline(unet, decoder)
    combined_model.eval()

    torch.onnx.export(
        combined_model,
        (dummy_latent, dummy_t),
        onnx_path,
        input_names=["latent", "timestep"],
        output_names=["generated_image"],
        dynamic_axes={"latent": {0: "batch_size"}, "generated_image": {0: "batch_size"}},
        dynamo=False
    )
    print(f"Exported ONNX model to {onnx_path}")

if __name__ == "__main__":
    train()
