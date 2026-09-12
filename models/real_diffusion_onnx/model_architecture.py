import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class RealLatentUNet(nn.Module):
    """
    Real Latent UNet for 256x256 image / sprite denoising in latent space (4, 32, 32).
    """
    def __init__(self, in_channels=4, out_channels=4, cond_dim=128):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, 64)
        self.time_proj = nn.Linear(1, 64)

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.res1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

        self.down = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # 32x32 -> 16x16
        self.res2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128)
        )

        self.up = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # 16x16 -> 32x32
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, latent, timestep, text_embed):
        t_emb = self.time_proj(timestep).unsqueeze(-1).unsqueeze(-1)
        c_emb = self.cond_proj(text_embed).unsqueeze(-1).unsqueeze(-1)

        x = F.relu(self.conv1(latent)) + t_emb + c_emb
        res = x
        x = F.relu(self.res1(x) + res)

        x = F.relu(self.down(x))
        x = F.relu(self.res2(x))

        x = F.relu(self.up(x))
        out = self.out_conv(x)
        return out


class RealLatentDecoder(nn.Module):
    """
    Neural VAE Decoder for converting latent features (4, 32, 32) into high-quality RGB images (3, 256, 256).
    """
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(in_channels, 64, kernel_size=4, stride=2, padding=1)  # 32x32 -> 64x64
        self.bn1 = nn.BatchNorm2d(64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)           # 64x64 -> 128x128
        self.bn2 = nn.BatchNorm2d(32)
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)           # 128x128 -> 256x256
        self.bn3 = nn.BatchNorm2d(16)
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        x = F.relu(self.bn1(self.up1(z)))
        x = F.relu(self.bn2(self.up2(x)))
        x = F.relu(self.bn3(self.up3(x)))
        out = torch.sigmoid(self.out_conv(x))
        return out
