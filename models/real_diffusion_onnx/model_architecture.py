import torch
import torch.nn as nn
import torch.nn.functional as F

class RealLatentUNet(nn.Module):
    """
    Lightweight Neural Latent UNet for iterative denoising (32x32 latent space).
    CPU-optimized for real-time neural diffusion synthesis.
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

        self.in_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.down1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # -> 16x16
        self.mid_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # -> 32x32
        self.out_conv = nn.Conv2d(128, in_channels, kernel_size=3, padding=1)

    def forward(self, latent, timestep, text_embed):
        t_emb = self.time_mlp(timestep).unsqueeze(-1).unsqueeze(-1) # (B, 64, 1, 1)
        c_emb = self.cond_mlp(text_embed).unsqueeze(-1).unsqueeze(-1) # (B, 64, 1, 1)

        x1 = F.silu(self.in_conv(latent) + t_emb + c_emb) # (B, 64, 32, 32)
        x2 = F.silu(self.down1(x1)) # (B, 128, 16, 16)
        x_mid = F.silu(self.mid_conv(x2)) # (B, 128, 16, 16)
        x_up = F.silu(self.up1(x_mid)) # (B, 64, 32, 32)

        x_cat = torch.cat([x1, x_up], dim=1) # (B, 128, 32, 32)
        out = self.out_conv(x_cat)
        return out

class RealLatentDecoder(nn.Module):
    """
    Neural VAE Decoder mapping 32x32 latent features to 256x256 RGB image output.
    """
    def __init__(self, latent_channels=4, out_channels=3):
        super().__init__()
        self.in_conv = nn.Conv2d(latent_channels, 64, kernel_size=3, padding=1) # 32x32
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1) # -> 64x64
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1) # -> 128x128
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1) # -> 256x256
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, latent):
        x = F.relu(self.in_conv(latent))
        x = F.relu(self.up1(x))
        x = F.relu(self.up2(x))
        x = F.relu(self.up3(x))
        rgb = torch.sigmoid(self.out_conv(x))
        return rgb
