"""
Real Latent UNet and VAE Decoder Architecture Module
Provides high-performance lightweight neural architectures for 256x256 image synthesis.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class RealLatentUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, cond_dim=128):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, 64)
        self.time_proj = nn.Linear(1, 64)

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, x, t, text_embed):
        # x: (B, 4, 32, 32), t: (B, 1), text_embed: (B, 128)
        cond_emb = self.cond_proj(text_embed).unsqueeze(-1).unsqueeze(-1) # (B, 64, 1, 1)
        time_emb = self.time_proj(t).unsqueeze(-1).unsqueeze(-1)         # (B, 64, 1, 1)

        h = self.act(self.conv1(x)) + cond_emb + time_emb
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        out = self.conv_out(h)
        return out


class RealLatentDecoder(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1) # -> 64x64
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1) # -> 128x128
        self.up3 = nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1) # -> 256x256
        self.conv_out = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, z):
        # z: (B, 4, 32, 32)
        h = self.act(self.conv_in(z))
        h = self.act(self.up1(h))
        h = self.act(self.up2(h))
        h = self.act(self.up3(h))
        out = torch.sigmoid(self.conv_out(h)) # (B, 3, 256, 256)
        return out
