"""
Neural Latent Diffusion UNet and VAE Decoder Model Architecture.
Supports latent space (4, 32, 32) denoising and high-resolution VAE decoding to (3, 256, 256).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class RealLatentUNet(nn.Module):
    """
    Latent UNet Model operating on (B, 4, 32, 32) feature maps.
    Conditioned on timestep t and 128-dim text condition vector.
    """
    def __init__(self, in_channels=4, out_channels=4, cond_dim=128):
        super().__init__()
        self.time_proj = nn.Linear(1, 64)
        self.cond_proj = nn.Linear(cond_dim, 64)

        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU()
        )
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x, t, cond):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.act(self.time_proj(t.float())).unsqueeze(-1).unsqueeze(-1)
        c_emb = self.act(self.cond_proj(cond)).unsqueeze(-1).unsqueeze(-1)

        h = self.conv_in(x) + t_emb + c_emb
        h = h + self.block1(h)
        out = self.conv_out(h)
        return out


class RealLatentDecoder(nn.Module):
    """
    Neural VAE Decoder mapping (B, 4, 32, 32) latent tensors up to (B, 3, 256, 256) RGB images.
    Upsampling stages: 32x32 -> 64x64 -> 128x128 -> 256x256.
    """
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        # 32x32 -> 64x64
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.GroupNorm(8, 64)

        # 64x64 -> 128x128
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.GroupNorm(8, 32)

        # 128x128 -> 256x256
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.GroupNorm(8, 16)

        self.conv_out = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        h = F.silu(self.conv_in(z))
        h = F.silu(self.bn1(self.up1(h)))
        h = F.silu(self.bn2(self.up2(h)))
        h = F.silu(self.bn3(self.up3(h)))
        out = torch.sigmoid(self.conv_out(h))
        return out
