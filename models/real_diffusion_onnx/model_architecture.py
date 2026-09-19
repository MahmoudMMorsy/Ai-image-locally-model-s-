import torch
import torch.nn as nn
import torch.nn.functional as F

class RealLatentUNet(nn.Module):
    def __init__(self, in_channels=4, cond_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, cond_dim)
        )
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.res1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.res2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t, cond):
        t_emb = self.time_embed(t)
        c = cond + t_emb
        h = F.silu(self.conv_in(x))
        h = F.silu(self.res1(h) + c.unsqueeze(-1).unsqueeze(-1))
        h = F.silu(self.res2(h))
        return self.conv_out(h)

class RealLatentDecoder(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(in_channels, 64, kernel_size=4, stride=2, padding=1) # 32x32 -> 64x64
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)           # 64x64 -> 128x128
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)           # 128x128 -> 256x256
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        h = F.relu(self.up1(z))
        h = F.relu(self.up2(h))
        h = F.relu(self.up3(h))
        return torch.sigmoid(self.out_conv(h))
