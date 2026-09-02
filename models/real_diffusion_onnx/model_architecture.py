import torch
import torch.nn as nn
import torch.nn.functional as F

class RealLatentUNet(nn.Module):
    """
    Lightweight Latent UNet for 256x256 image generation in 32x32 latent space (4 channels).
    Accepts latent (B, 4, 32, 32), timestep t (B, 1), and text_embed condition (B, 128).
    """
    def __init__(self, in_channels=4, out_channels=4, cond_dim=128):
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

        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        # Down block
        self.down_block = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )

        # Mid block
        self.mid_block = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU()
        )

        # Up block & output
        self.conv_out = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, latent, timestep, text_embed):
        t_emb = self.time_embed(timestep) # (B, 64)
        c_emb = self.cond_embed(text_embed) # (B, 64)
        emb = (t_emb + c_emb).unsqueeze(-1).unsqueeze(-1) # (B, 64, 1, 1)

        h = self.conv_in(latent) + emb
        h_down = self.down_block(h)
        h_mid = self.mid_block(torch.cat([h_down, h_down], dim=1))
        out = self.conv_out(h_mid)
        return out


class RealLatentDecoder(nn.Module):
    """
    Neural VAE Latent Decoder.
    Decodes latent features (B, 4, 32, 32) into high-resolution RGB image (B, 3, 256, 256).
    """
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)

        # Upsample 32x32 -> 64x64 -> 128x128 -> 256x256
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 64x64
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 256x256
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid() # RGB output in [0, 1]
        )

    def forward(self, latent):
        h = self.conv_in(latent)
        return self.upsample(h)
