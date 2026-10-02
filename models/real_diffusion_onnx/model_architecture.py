import torch
import torch.nn as nn
import torch.nn.functional as F

class RealLatentUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, cond_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        self.cond_proj = nn.Linear(cond_dim, 64)

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, latent, timestep, text_embed):
        t_emb = self.time_mlp(timestep).unsqueeze(-1).unsqueeze(-1)
        c_emb = self.cond_proj(text_embed).unsqueeze(-1).unsqueeze(-1)

        h = F.silu(self.conv1(latent)) + t_emb + c_emb
        h = F.silu(self.conv2(h))
        h = F.silu(self.conv3(h))
        return self.conv_out(h)

class RealLatentDecoder(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, latent):
        h = F.silu(self.up1(latent))
        h = F.silu(self.up2(h))
        h = F.silu(self.up3(h))
        return torch.sigmoid(self.out_conv(h))
