import torch
import torch.nn as nn
import torch.nn.functional as F

class RealLatentUNet(nn.Module):
    """
<<<<<<< HEAD
    Lightweight CPU-friendly Latent UNet for 32x32 latents (decoding to 256x256 images)
    or 8x8 latents (decoding to 64x64 pixel sprites).
    """
    def __init__(self, in_channels=4, out_channels=4, cond_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.SiLU(),
=======
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
>>>>>>> origin/feature-pixel-art-sprite-engine-10984173554244238923
            nn.Linear(64, 64)
        )

        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
<<<<<<< HEAD

        self.down1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.mid = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)

        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x, timestep, cond):
        t_emb = self.time_embed(timestep).unsqueeze(-1).unsqueeze(-1)
        c_emb = self.cond_embed(cond).unsqueeze(-1).unsqueeze(-1)

        h = F.relu(self.conv_in(x)) + t_emb + c_emb
        h_down = F.relu(self.down1(h))
        h_mid = F.relu(self.mid(h_down))
        h_up = F.relu(self.up1(h_mid))
        out = self.conv_out(h_up + h)
=======
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
>>>>>>> origin/feature-pixel-art-sprite-engine-10984173554244238923
        return out


class RealLatentDecoder(nn.Module):
    """
<<<<<<< HEAD
    Lightweight VAE Neural Decoder mapping 32x32 latents to 256x256 RGB images
    or 8x8 latents to 64x64 pixel art images.
    """
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # -> 64x64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # -> 128x128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # -> 256x256
            nn.ReLU(inplace=True),
            nn.Conv2d(8, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z)
=======
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
>>>>>>> origin/feature-pixel-art-sprite-engine-10984173554244238923
