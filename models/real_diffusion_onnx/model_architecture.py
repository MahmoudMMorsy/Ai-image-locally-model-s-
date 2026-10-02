import torch
import torch.nn as nn
import torch.nn.functional as F

class RealLatentUNet(nn.Module):
    def __init__(self, in_channels=4, cond_dim=128):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.time_emb = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        self.cond_emb = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1)
        )
        self.conv_out = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)

    def forward(self, x, timestep, condition):
        h = self.conv_in(x)
        t_emb = self.time_emb(timestep).unsqueeze(-1).unsqueeze(-1)
        c_emb = self.cond_emb(condition).unsqueeze(-1).unsqueeze(-1)
        h = h + t_emb + c_emb
        h = h + self.block1(h)
        return self.conv_out(h)


class RealLatentDecoder(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"), # 32x32 -> 64x64
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"), # 64x64 -> 128x128
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"), # 128x128 -> 256x256
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z)
