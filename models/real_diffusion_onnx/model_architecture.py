import torch
import torch.nn as nn

class RealLatentUNet(nn.Module):
    """
    Lightweight Latent UNet for 256x256 diffusion generation.
    Operates on 4x32x32 latent representations.
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
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.conv_out = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)

    def forward(self, x, timestep, text_embed):
        t_emb = self.time_embed(timestep).unsqueeze(-1).unsqueeze(-1)
        c_emb = self.cond_embed(text_embed).unsqueeze(-1).unsqueeze(-1)
        h = self.conv_in(x) + t_emb + c_emb
        h = self.block1(h)
        out = self.conv_out(h)
        return out

class RealLatentDecoder(nn.Module):
    """
    Neural VAE Decoder mapping 4x32x32 latents to 3x256x256 RGB images.
    """
    def __init__(self, latent_channels=4, out_channels=3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 64x64
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # 128x128
            nn.SiLU(),
            nn.ConvTranspose2d(16, out_channels, kernel_size=4, stride=2, padding=1), # 256x256
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z)
