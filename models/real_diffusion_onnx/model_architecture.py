"""
Real Latent UNet and VAE Decoder Model Architecture
Optimized PyTorch neural models for latent diffusion generation and 256x256 RGB image decoding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RealLatentUNet(nn.Module):
    """
    Lightweight 32x32 Latent UNet with text embedding and timestep conditioning.
    Processes latent feature maps (B, 4, 32, 32) and predicts denoised latents.
    """
    def __init__(self, in_channels=4, out_channels=4, cond_dim=128):
        super().__init__()
        self.time_proj = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 64)
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64)
        )

        # Down blocks
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.down1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # -> 16x16
        self.down2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # -> 8x8

        # Mid block
        self.mid_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.mid_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Up blocks
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # -> 16x16
        self.up1 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)  # -> 32x32

        # Output block
        self.conv_out = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t, cond):
        # x: (B, 4, 32, 32), t: (B, 1), cond: (B, 128)
        t_emb = self.time_proj(t).unsqueeze(-1).unsqueeze(-1)   # (B, 64, 1, 1)
        c_emb = self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1)  # (B, 64, 1, 1)
        emb = t_emb + c_emb

        h1 = F.gelu(self.conv_in(x) + emb)     # (B, 64, 32, 32)
        h2 = F.gelu(self.down1(h1))             # (B, 128, 16, 16)
        h3 = F.gelu(self.down2(h2))             # (B, 256, 8, 8)

        m = F.gelu(self.mid_conv1(h3))
        m = F.gelu(self.mid_conv2(m))            # (B, 256, 8, 8)

        u2 = F.gelu(self.up2(m))                # (B, 128, 16, 16)
        u2_cat = torch.cat([u2, h2], dim=1)     # (B, 256, 16, 16)

        u1 = F.gelu(self.up1(u2_cat))           # (B, 64, 32, 32)
        u1_cat = torch.cat([u1, h1], dim=1)     # (B, 128, 32, 32)

        out = self.conv_out(u1_cat)
        return out


class RealLatentDecoder(nn.Module):
    """
    Neural VAE Decoder mapping 32x32 latents (B, 4, 32, 32) to 256x256 RGB images (B, 3, 256, 256).
    """
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)

        # 32x32 -> 64x64
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        # 64x64 -> 128x128
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        # 128x128 -> 256x256
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)

        self.conv_out = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        h = F.gelu(self.conv_in(z))
        h = F.gelu(self.up1(h))
        h = F.gelu(self.up2(h))
        h = F.gelu(self.up3(h))
        rgb = torch.sigmoid(self.conv_out(h))
        return rgb
