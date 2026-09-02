import torch
import torch.nn as nn
import torch.nn.functional as F

class NanoPixelMomUNet(nn.Module):
    """
    Nano Pixel MOM Architecture (Massive Output Model)
    Super-fast CPU-optimized Latent UNet Generator for 64x64/128x128 Pixel Art Characters.
    """
    def __init__(self, in_channels=4, out_channels=4, cond_dim=128):
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

        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )

        self.mid = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, latent, timestep, condition):
        t_emb = self.time_mlp(timestep).unsqueeze(-1).unsqueeze(-1)
        c_emb = self.cond_mlp(condition).unsqueeze(-1).unsqueeze(-1)
        emb = t_emb + c_emb

        h = self.conv_in(latent) + emb
        h_down = self.down1(h)
        h_mid = self.mid(h_down)
        out = self.up1(h_mid)
        return out


class NanoPixelMomDecoder(nn.Module):
    """
    Nano Pixel MOM Neural VAE Decoder
    Decodes 32x32 latents directly into 64x64 or 128x128 crisp pixel art RGB.
    """
    def __init__(self, in_channels=4, out_channels=4):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 64x64
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, latent):
        h = self.conv_in(latent)
        return self.upsample(h)
