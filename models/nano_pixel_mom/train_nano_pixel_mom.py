import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.nano_pixel_mom.model import NanoPixelMomUNet, NanoPixelMomDecoder
from models.nano_pixel_mom.dataset_processor import PixelMomDataset

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return torch.mean(loss)

class PaletteConsistencyLoss(nn.Module):
    def forward(self, pred_img):
        grad_x = torch.abs(pred_img[:, :, :, 1:] - pred_img[:, :, :, :-1])
        grad_y = torch.abs(pred_img[:, :, 1:, :] - pred_img[:, :, :-1, :])
        return torch.mean(grad_x) + torch.mean(grad_y)

def train_nano_pixel_mom():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training NanoPixelMom model on device: {device}")

    dataset = PixelMomDataset()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

    unet = NanoPixelMomUNet().to(device)
    decoder = NanoPixelMomDecoder().to(device)

    charbonnier = CharbonnierLoss()
    palette_loss = PaletteConsistencyLoss()

    optimizer = optim.AdamW(list(unet.parameters()) + list(decoder.parameters()), lr=1e-3, weight_decay=1e-4)

    epochs = 2
    print(f"Starting training over {len(dataset)} images for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        unet.train()
        decoder.train()
        total_loss = 0.0

        for batch_idx, (imgs, conds) in enumerate(dataloader):
            imgs = imgs.to(device)
            conds = conds.to(device)
            batch_size = imgs.shape[0]

            t = torch.randint(0, 20, (batch_size,), device=device).long()
            noise = torch.randn_like(imgs)

            noisy_imgs = imgs + 0.1 * noise
            pred_noise = unet(noisy_imgs, t, conds)
            decoded = decoder(pred_noise)

            loss = charbonnier(pred_noise, noise) + 0.05 * palette_loss(decoded)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size

        avg_loss = total_loss / len(dataset)
        print(f"Epoch [{epoch}/{epochs}] Average Loss: {avg_loss:.4f}")

    weights_dir = "models/nano_pixel_mom/weights"
    os.makedirs(weights_dir, exist_ok=True)

    unet_path = os.path.join(weights_dir, "nano_pixel_mom_unet.pt")
    decoder_path = os.path.join(weights_dir, "nano_pixel_mom_decoder.pt")

    torch.save(unet.state_dict(), unet_path)
    torch.save(decoder.state_dict(), decoder_path)

    print(f"Saved unet weights to {unet_path}")
    print(f"Saved decoder weights to {decoder_path}")

if __name__ == "__main__":
    train_nano_pixel_mom()
