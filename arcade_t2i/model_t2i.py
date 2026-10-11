"""Text-conditioned CVAE for 64x64 pixel art sprite generation."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextEncoder(nn.Module):
    def __init__(self, n_tokens: int = 100, dim: int = 64, out: int = 128):
        super().__init__()
        self.emb = nn.Embedding(n_tokens, dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(dim, out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(out, out),
            nn.LayerNorm(out)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        e = self.emb(token_ids.clamp(0, self.emb.num_embeddings - 1))
        mask = (token_ids > 0).float().unsqueeze(-1)
        pooled = (e * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.fc(pooled)

class ResBlock2d(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, channels)
        self.cond_proj = nn.Linear(cond_dim, channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        c = self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1)
        h = self.conv1(x + c)
        h = self.gn1(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.gn2(h)
        return self.act(x + h)

class TextCVAE(nn.Module):
    def __init__(self, n_colors: int = 48, latent: int = 64, base: int = 24, cond_dim: int = 128, n_tokens: int = 100):
        super().__init__()
        self.latent = latent
        self.cond_dim = cond_dim
        self.n_colors = n_colors
        self.base = base

        self.color_emb = nn.Embedding(n_colors, 16)
        self.text_enc = TextEncoder(n_tokens=n_tokens, dim=64, out=cond_dim)
        self.class_emb = nn.Embedding(32, cond_dim)

        self.enc1 = nn.Conv2d(16, base, 4, 2, 1) # 32x32
        self.enc2 = nn.Conv2d(base, base * 2, 4, 2, 1) # 16x16
        self.enc3 = nn.Conv2d(base * 2, base * 4, 4, 2, 1) # 8x8
        self.enc4 = nn.Conv2d(base * 4, base * 4, 4, 2, 1) # 4x4

        flat_dim = base * 4 * 4 * 4
        self.fc_mu = nn.Linear(flat_dim + cond_dim, latent)
        self.fc_lv = nn.Linear(flat_dim + cond_dim, latent)

        self.fc_dec = nn.Linear(latent + cond_dim, flat_dim)

        self.res4 = ResBlock2d(base * 4, cond_dim)
        self.up3 = nn.ConvTranspose2d(base * 4, base * 4, 4, 2, 1)
        self.res3 = ResBlock2d(base * 4, cond_dim)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1)
        self.res2 = ResBlock2d(base * 2, cond_dim)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, 2, 1)
        self.res1 = ResBlock2d(base, cond_dim)

        self.up0 = nn.ConvTranspose2d(base, base, 4, 2, 1)
        self.res0 = ResBlock2d(base, cond_dim)

        self.out_conv = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base, n_colors, 1)
        )

    def cond(self, tokens: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
        t_feat = self.text_enc(tokens)
        c_feat = self.class_emb(class_ids.clamp(0, 31))
        return t_feat + c_feat

    def encode(self, idx: torch.Tensor, cond: torch.Tensor):
        x = self.color_emb(idx.long()).permute(0, 3, 1, 2)
        h1 = F.silu(self.enc1(x))
        h2 = F.silu(self.enc2(h1))
        h3 = F.silu(self.enc3(h2))
        h4 = F.silu(self.enc4(h3))

        flat = h4.flatten(1)
        flat_cond = torch.cat([flat, cond], dim=1)
        return self.fc_mu(flat_cond), self.fc_lv(flat_cond)

    def reparam(self, mu: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:
        std = (0.5 * lv).exp()
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h_dec = F.silu(self.fc_dec(torch.cat([z, cond], dim=1)))
        h_dec = h_dec.view(-1, self.base * 4, 4, 4)

        h = self.res4(h_dec, cond)
        h = self.res3(self.up3(h), cond)
        h = self.res2(self.up2(h), cond)
        h = self.res1(self.up1(h), cond)
        h = self.res0(self.up0(h), cond)

        logits = self.out_conv(h)
        return logits

    def forward(self, idx: torch.Tensor, tokens: torch.Tensor, class_ids: torch.Tensor):
        c = self.cond(tokens, class_ids)
        mu, lv = self.encode(idx, c)
        z = self.reparam(mu, lv)
        logits = self.decode(z, c)
        return logits, mu, lv

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, class_ids: torch.Tensor, n: int = 1, z: torch.Tensor | None = None, temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        device = tokens.device
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if class_ids.dim() == 0:
            class_ids = class_ids.unsqueeze(0)

        c = self.cond(tokens[:1].expand(n, -1), class_ids[:1].expand(n))
        if z is None:
            z = torch.randn(n, self.latent, device=device)

        logits = self.decode(z, c)
        if temperature > 0 and temperature != 1.0:
            probs = F.softmax(logits / temperature, dim=1)
            B, C, H, W = probs.shape
            flat_probs = probs.permute(0, 2, 3, 1).reshape(-1, C)
            sampled = torch.multinomial(flat_probs, 1).view(B, H, W)
            return sampled
        else:
            return logits.argmax(dim=1)

if __name__ == "__main__":
    model = TextCVAE()
    dummy_idx = torch.randint(0, 48, (2, 64, 64))
    dummy_tokens = torch.randint(0, 50, (2, 16))
    dummy_class = torch.tensor([0, 1])
    logits, mu, lv = model(dummy_idx, dummy_tokens, dummy_class)
    print("Forward pass successful! Logits shape:", logits.shape)
