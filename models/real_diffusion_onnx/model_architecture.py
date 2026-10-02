import torch
import torch.nn as nn
import torch.nn.functional as F

class RealLatentUNet(nn.Module):
    """
    Real Latent UNet Architecture for Denoising Diffusion.
    Takes latent state (B, 4, 32, 32), timestep (B, 1), and text embedding (B, 128)
    and predicts denoised latent state (B, 4, 32, 32).
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
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.res1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, latent, timestep, text_embed):
        t_emb = self.time_mlp(timestep).unsqueeze(-1).unsqueeze(-1)
        c_emb = self.cond_mlp(text_embed).unsqueeze(-1).unsqueeze(-1)
        h = self.conv_in(latent) + t_emb + c_emb
        h = h + self.res1(h)
        return self.conv_out(h)

class RealLatentDecoder(nn.Module):
    """
    Neural VAE Decoder Architecture.
    Translates latent representations (B, 4, 32, 32) to full RGB resolution (B, 3, 256, 256).
    """
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 64, kernel_size=4, stride=2, padding=1), # -> 64x64
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # -> 128x128
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # -> 256x256
            nn.BatchNorm2d(16),
            nn.GELU()
        )
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, latent):
        h = self.up1(latent)
        h = self.up2(h)
        h = self.up3(h)
        return torch.sigmoid(self.out_conv(h))
