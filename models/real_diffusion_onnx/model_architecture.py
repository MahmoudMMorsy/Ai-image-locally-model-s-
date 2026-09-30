"""
Real Latent UNet and VAE Decoder Neural Architectures
For 256x256 Neural Latent Diffusion Image Synthesis.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RealLatentUNet(nn.Module):
    """
    Conditioned Latent UNet Denoising Model operating on 32x32 latent representations.
    """
    def __init__(self, in_channels=4, out_channels=4, cond_dim=128, time_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, time_dim),
            nn.GELU()
        )

        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, latent, t, text_embed):
        t_emb = self.act(self.time_mlp(t)).unsqueeze(-1).unsqueeze(-1)
        c_emb = self.act(self.cond_mlp(text_embed)).unsqueeze(-1).unsqueeze(-1)

        h = self.act(self.conv_in(latent)) + t_emb + c_emb
        h = self.act(self.conv1(h))
        h = self.act(self.conv2(h))
        return self.conv_out(h)


class RealLatentDecoder(nn.Module):
    """
    Neural VAE Decoder mapping 32x32 latent feature maps to 256x256 RGB images.
    """
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        # 32x32 -> 64x64 -> 128x128 -> 256x256
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)

        self.conv_out = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, latent):
        h = self.act(self.conv_in(latent))
        h = self.act(self.up1(h))
        h = self.act(self.up2(h))
        h = self.act(self.up3(h))
        out = torch.sigmoid(self.conv_out(h))
        return out
