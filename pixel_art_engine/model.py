"""
Lightweight Neural Network Architecture for 64x64 Pixel Art Sprite Generation
CPU-optimized PyTorch model for text-to-sprite and latent sprite generation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Simple lightweight residual block for preserving fine pixel details."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + res)


class PixelSpriteGenerator(nn.Module):
    """
    Lightweight Generator for 64x64 RGBA Pixel Art Sprites.
    Extremely fast on CPU (<50ms per character).
    """
    def __init__(self, latent_dim=64, condition_dim=32, out_channels=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        # Initial dense projection: maps (latent + condition) to 8x8x128
        self.fc = nn.Linear(latent_dim + condition_dim, 128 * 8 * 8)

        # Upsampling backbone: 8x8 -> 16x16 -> 32x32 -> 64x64
        self.res1 = ResidualBlock(128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # -> 16x16
        self.bn1 = nn.BatchNorm2d(64)

        self.res2 = ResidualBlock(64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # -> 32x32
        self.bn2 = nn.BatchNorm2d(32)

        self.res3 = ResidualBlock(32)
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)  # -> 64x64
        self.bn3 = nn.BatchNorm2d(16)

        # Output convolution to 4 channels (RGBA)
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, z, condition=None):
        if condition is None:
            condition = torch.zeros(z.size(0), self.condition_dim, device=z.device)

        x = torch.cat([z, condition], dim=1)
        x = self.fc(x)
        x = x.view(-1, 128, 8, 8)

        x = self.res1(x)
        x = F.relu(self.bn1(self.up1(x)))

        x = self.res2(x)
        x = F.relu(self.bn2(self.up2(x)))

        x = self.res3(x)
        x = F.relu(self.bn3(self.up3(x)))

        out = torch.sigmoid(self.out_conv(x)) # [0, 1] RGBA values
        return out


class PixelSpriteEncoder(nn.Module):
    """
    Lightweight Encoder for mapping 64x64 RGBA images into character identity latent vector.
    """
    def __init__(self, in_channels=4, latent_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=4, stride=2, padding=1) # -> 32x32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)          # -> 16x16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)          # -> 8x8
        self.conv4 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)         # -> 4x4
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = x.view(x.size(0), -1)
        latent = self.fc(x)
        return latent
