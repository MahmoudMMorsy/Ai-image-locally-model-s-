import torch
import torch.nn as nn
import torch.nn.functional as F

class RealLatentUNet(nn.Module):
    """
    Lightweight 256x256 Real Latent UNet Model for Diffusion Synthesis.
    Operates on 32x32x4 latent space representation.
    """
    def __init__(self, in_channels=4, out_channels=4, time_dim=64, cond_dim=128):
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
        self.conv_mid1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv_mid2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, x, t, cond):
        t_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        c_emb = self.cond_mlp(cond).unsqueeze(-1).unsqueeze(-1)

        h = self.act(self.conv_in(x)) + t_emb + c_emb
        h = self.act(self.conv_mid1(h))
        h = self.act(self.conv_mid2(h))
        out = self.conv_out(h)
        return out


class RealLatentDecoder(nn.Module):
    """
    Neural VAE Decoder for mapping 32x32x4 latents to 256x256x3 RGB images.
    """
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1)
        self.conv_out = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, z):
        h = self.act(self.conv_in(z))
        h = self.act(self.up1(h))
        h = self.act(self.up2(h))
        h = self.act(self.up3(h))
        out = torch.sigmoid(self.conv_out(h))
        return out
