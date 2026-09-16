"""
Neural Latent Diffusion UNet and VAE Decoder Architecture.
Provides real latent space denoising and VAE RGB decoding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RealLatentUNet(nn.Module):
    """
    Lightweight Latent UNet Denoising Network.
    Accepts latent tensor (B, 4, 32, 32), timestep (B, 1), and text embedding (B, 128).
    Outputs denoised latent tensor (B, 4, 32, 32).
    """
    def __init__(self, in_channels=4, cond_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )

        self.inc = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        ) # 16x16

        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ) # 32x32

        self.outc = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)

    def forward(self, x, timestep, cond):
        t_emb = self.time_embed(timestep).unsqueeze(-1).unsqueeze(-1) # (B, 64, 1, 1)
        c_emb = self.cond_embed(cond).unsqueeze(-1).unsqueeze(-1)     # (B, 64, 1, 1)

        x1 = self.inc(x) + t_emb + c_emb # (B, 64, 32, 32)
        x2 = self.down1(x1)              # (B, 128, 16, 16)
        x3 = self.bottleneck(x2)         # (B, 128, 16, 16)
        x4 = self.up1(x3) + x1           # (B, 64, 32, 32)

        return self.outc(x4)


class RealLatentDecoder(nn.Module):
    """
    Neural VAE Decoder mapping latent representation (B, 4, 32, 32) to RGB image (B, 3, 256, 256).
    """
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        # 32x32 -> 64x64 -> 128x128 -> 256x256
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        x = self.init_conv(z)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        out = torch.sigmoid(self.final_conv(x))
        return out
