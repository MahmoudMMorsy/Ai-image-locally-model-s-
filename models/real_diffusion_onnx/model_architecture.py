import torch
import torch.nn as nn
import torch.nn.functional as F

class RealLatentUNet(nn.Module):
    """
    Real Latent UNet for 32x32 latent space feature denoising.
    Takes latent (B, 4, 32, 32), timestep (B, 1), and text embedding condition (B, 128).
    Outputs denoised latent feature map (B, 4, 32, 32).
    """
    def __init__(self, in_channels=4, out_channels=4, cond_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 64)
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64)
        )

        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1)
        )
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t, cond):
        t_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        c_emb = self.cond_mlp(cond).unsqueeze(-1).unsqueeze(-1)

        h = F.gelu(self.conv_in(x)) + t_emb + c_emb
        h = h + self.block1(h)
        out = self.conv_out(h)
        return out


class RealLatentDecoder(nn.Module):
    """
    Neural VAE Decoder for converting latent features (B, 4, 32, 32)
    to RGB poster/sprite image (B, 3, 256, 256).
    """
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        # 32x32 -> 64x64 -> 128x128 -> 256x256
        self.up1 = nn.ConvTranspose2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.gn1 = nn.GroupNorm(8, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.gn2 = nn.GroupNorm(4, 32)

        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.gn3 = nn.GroupNorm(4, 16)

        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        x = F.gelu(self.gn1(self.up1(z)))
        x = F.gelu(self.gn2(self.up2(x)))
        x = F.gelu(self.gn3(self.up3(x)))
        out = torch.sigmoid(self.out_conv(x))
        return out
