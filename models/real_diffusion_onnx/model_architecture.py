import torch
import torch.nn as nn
import torch.nn.functional as F

class RealLatentUNet(nn.Module):
    """
    Neural Latent UNet architecture operating on 4-channel latents (4, 32, 32).
    """
    def __init__(self, in_channels=4, cond_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        self.inc = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU()
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU()
        )
        self.outc = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)

    def forward(self, x, time, cond):
        t_emb = self.time_mlp(time).unsqueeze(-1).unsqueeze(-1)
        c_emb = self.cond_mlp(cond).unsqueeze(-1).unsqueeze(-1)
        h1 = self.inc(x) + t_emb + c_emb
        h2 = self.down1(h1)
        h3 = self.up1(h2) + h1
        out = self.outc(h3)
        return out

class RealLatentDecoder(nn.Module):
    """
    Neural VAE Decoder mapping 4-channel latents (4, 32, 32) to 3-channel RGB (3, 256, 256).
    """
    def __init__(self, latent_channels=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=4, stride=2, padding=1), # -> (128, 64, 64)
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 128, 128)
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> (32, 256, 256)
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),                        # -> (3, 256, 256)
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)
