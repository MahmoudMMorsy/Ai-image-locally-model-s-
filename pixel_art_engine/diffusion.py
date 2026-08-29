"""
Lightweight Denoising Diffusion Probabilistic Model (DDPM / Unet)
For 64x64 Pixel Art Character Sprite Generation trained on user dataset.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, cond_dim=32):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim + cond_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.transform = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x, t_cond):
        h = F.relu(self.bn1(self.conv1(x)))
        emb = F.relu(self.time_mlp(t_cond))
        emb = emb[(..., ) + (None, ) * 2]
        h = h + emb
        h = F.relu(self.bn2(self.transform(h)))
        return h


class PixelUNet(nn.Module):
    """
    Lightweight 64x64 Conditioned UNet Denoising Model for Pixel Art Generation.
    Supports text/class condition vectors.
    """
    def __init__(self, in_channels=4, out_channels=4, time_dim=32, cond_dim=32):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        # Downsampling
        self.down1 = Block(in_channels, 32, time_dim, cond_dim)   # 64x64 -> 64x64
        self.pool1 = nn.Conv2d(32, 32, 4, stride=2, padding=1)    # 64x64 -> 32x32

        self.down2 = Block(32, 64, time_dim, cond_dim)            # 32x32 -> 32x32
        self.pool2 = nn.Conv2d(64, 64, 4, stride=2, padding=1)    # 32x32 -> 16x16

        # Bottleneck
        self.bot = Block(64, 128, time_dim, cond_dim)

        # Upsampling
        self.up2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1) # 16x16 -> 32x32
        self.b_up2 = Block(128, 64, time_dim, cond_dim)

        self.up1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)  # 32x32 -> 64x64
        self.b_up1 = Block(64, 32, time_dim, cond_dim)

        self.out = nn.Conv2d(32, out_channels, 1)

    def forward(self, x, timestep, condition=None):
        t = self.time_mlp(timestep)
        if condition is None:
            condition = torch.zeros(x.size(0), 32, device=x.device)

        t_cond = torch.cat([t, condition], dim=1)

        x1 = self.down1(x, t_cond)
        p1 = self.pool1(x1)

        x2 = self.down2(p1, t_cond)
        p2 = self.pool2(x2)

        bot = self.bot(p2, t_cond)

        u2 = self.up2(bot)
        u2 = torch.cat([u2, x2], dim=1)
        b2 = self.b_up2(u2, t_cond)

        u1 = self.up1(b2)
        u1 = torch.cat([u1, x1], dim=1)
        b1 = self.b_up1(u1, t_cond)

        return self.out(b1)


class PixelDiffusion:
    def __init__(self, timesteps=100, beta_start=0.0001, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.device = torch.device(device)
        self.model = PixelUNet(in_channels=4, out_channels=4).to(self.device)

        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alpha = torch.sqrt(self.alphas_cumprod[t])[(..., ) + (None, ) * 3]
        sqrt_one_minus_alpha = torch.sqrt(1.0 - self.alphas_cumprod[t])[(..., ) + (None, ) * 3]
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise, noise

    @torch.no_grad()
    def sample(self, shape=(1, 4, 64, 64), condition=None, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.model.eval()

        if condition is None:
            condition = torch.zeros(shape[0], 32, device=self.device)

        img = torch.randn(shape, device=self.device)
        for i in reversed(range(self.timesteps)):
            t = torch.tensor([i], device=self.device, dtype=torch.long)
            predicted_noise = self.model(img, t, condition)

            beta = self.betas[i]
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]

            if i > 0:
                noise = torch.randn_like(img)
            else:
                noise = 0

            img = (1 / torch.sqrt(alpha)) * (img - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise) + torch.sqrt(beta) * noise

        return img.clamp(-1.0, 1.0)
