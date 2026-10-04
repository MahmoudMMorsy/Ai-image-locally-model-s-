import torch
import torch.nn as nn
import torch.nn.functional as F

class RealLatentUNet(nn.Module):
    """
    Real Latent UNet Model for Diffusion Synthesis.
    Processes latent maps (B, 4, 32, 32) with timestep and text condition embeddings.
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
        self.block1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.block2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, x, t, cond):
        t_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1) # (B, 64, 1, 1)
        c_emb = self.cond_mlp(cond).unsqueeze(-1).unsqueeze(-1) # (B, 64, 1, 1)

        h = self.act(self.conv_in(x)) + t_emb + c_emb
        h = self.act(self.block1(h))
        h = self.act(self.block2(h))
        return self.conv_out(h)


class RealLatentDecoder(nn.Module):
    """
    Neural VAE Decoder. Maps latent representation (B, 4, 32, 32) to RGB image (B, 3, 256, 256).
    """
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        # Upsample 32x32 -> 64x64 -> 128x128 -> 256x256
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)

        self.conv_out = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, z):
        h = self.act(self.conv_in(z))
        h = self.act(self.up1(h))
        h = self.act(self.up2(h))
        h = self.act(self.up3(h))
        return torch.sigmoid(self.conv_out(h))
