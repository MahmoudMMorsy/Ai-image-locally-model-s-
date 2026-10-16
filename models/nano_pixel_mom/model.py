import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class MomBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim=32, cond_dim=64):
        super().__init__()
        self.mlp = nn.Linear(t_dim + cond_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(8, out_ch)

    def forward(self, x, t_cond):
        h = F.silu(self.gn(self.conv1(x)))
        emb = self.mlp(t_cond)[..., None, None]
        h = h + emb
        return F.silu(self.conv2(h))

class NanoPixelMomUNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=4, cond_dim=64):
        super().__init__()
        self.t_emb = nn.Sequential(
            SinusoidalEmbeddings(32),
            nn.Linear(32, 32),
            nn.SiLU()
        )
        self.cond_proj = nn.Linear(cond_dim, cond_dim)

        self.down1 = MomBlock(in_ch, 32, 32, cond_dim)
        self.pool1 = nn.Conv2d(32, 64, 4, stride=2, padding=1)

        self.down2 = MomBlock(64, 128, 32, cond_dim)
        self.pool2 = nn.Conv2d(128, 128, 4, stride=2, padding=1)

        self.mid = MomBlock(128, 128, 32, cond_dim)

        self.up2 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
        self.b_up2 = MomBlock(256, 64, 32, cond_dim)

        self.up1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.b_up1 = MomBlock(64, 32, 32, cond_dim)

        self.out = nn.Conv2d(32, out_ch, 1)

    def forward(self, x, t, cond):
        t_emb = self.t_emb(t)
        c_emb = self.cond_proj(cond)
        t_cond = torch.cat([t_emb, c_emb], dim=1)

        x1 = self.down1(x, t_cond)
        p1 = self.pool1(x1)

        x2 = self.down2(p1, t_cond)
        p2 = self.pool2(x2)

        mid = self.mid(p2, t_cond)

        u2 = self.up2(mid)
        u2 = torch.cat([u2, x2], dim=1)
        b2 = self.b_up2(u2, t_cond)

        u1 = self.up1(b2)
        u1 = torch.cat([u1, x1], dim=1)
        b1 = self.b_up1(u1, t_cond)

        return self.out(b1)

class NanoPixelMomDecoder(nn.Module):
    def __init__(self, in_ch=4, out_ch=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, out_ch, 1)
        )

    def forward(self, z):
        return self.net(z)
