import torch
import torch.nn as nn

class NanoPixelMomUNet(nn.Module):
    """
    Ultra-lightweight high-speed neural diffusion UNet for nano_pixel_mom.
    Optimized for fast CPU inference and low memory footprint (<50MB).
    """
    def __init__(self, in_channels=4, out_channels=4, time_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv_mid1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv_mid2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.relu = nn.GELU()

    def forward(self, x, t):
        t_emb = self.time_mlp(t.unsqueeze(-1)).unsqueeze(-1).unsqueeze(-1)
        h = self.relu(self.conv_in(x)) + t_emb
        h = self.relu(self.conv_mid1(h))
        h = self.relu(self.conv_mid2(h))
        return self.conv_out(h)

class NanoPixelMomDecoder(nn.Module):
    """
    Decoder mapping latent representation (4 channels, 64x64) to RGB image (3 channels, 128x128).
    """
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 32, kernel_size=4, stride=2, padding=1), # 64 -> 128
            nn.GELU(),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.upsample(z)
