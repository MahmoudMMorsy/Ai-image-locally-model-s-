"""Text-conditioned CVAE — compact for CPU RAM."""
from __future__ import annotations
import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, n_tokens: int = 512, dim: int = 64, out: int = 64):
        super().__init__()
        self.emb = nn.Embedding(n_tokens, dim, padding_idx=0)
        self.fc = nn.Sequential(nn.Linear(dim, out), nn.ReLU(True), nn.Linear(out, out))
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        e = self.emb(token_ids.clamp(0, self.emb.num_embeddings - 1))
        mask = (token_ids > 0).float().unsqueeze(-1)
        pooled = (e * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.fc(pooled)

class TextCVAE(nn.Module):
    def __init__(self, n_colors: int = 48, latent: int = 64, base: int = 24, cond_dim: int = 64):
        super().__init__()
        self.latent = latent
        self.cond_dim = cond_dim
        self.n_colors = n_colors
        self.color_emb = nn.Embedding(n_colors, 8)
        self.text = TextEncoder(512, 64, cond_dim)
        self.class_emb = nn.Embedding(128, cond_dim)
        self.enc = nn.Sequential(
            nn.Conv2d(8 + 1, base, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(base, base * 2, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1), nn.ReLU(True),
            nn.Flatten(),
        )
        flat = base * 4 * 8 * 8
        self.fc_mu = nn.Linear(flat, latent)
        self.fc_lv = nn.Linear(flat, latent)
        self.fc_dec = nn.Linear(latent + cond_dim, flat)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(base, base, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(base, n_colors, 3, 1, 1),
        )
        self._base = base

    def cond(self, tokens, class_ids):
        return self.text(tokens) + self.class_emb(class_ids.clamp(0, 127))

    def encode(self, idx, c):
        x = self.color_emb(idx).permute(0, 3, 1, 2)
        cmap = c.mean(1, keepdim=True).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 64, 64)
        h = self.enc(torch.cat([x, cmap], 1))
        return self.fc_mu(h), self.fc_lv(h)

    def reparam(self, mu, lv):
        std = (0.5 * lv).exp()
        return mu + std * torch.randn_like(std)

    def decode(self, z, c):
        b = self._base
        h = self.fc_dec(torch.cat([z, c], 1)).view(-1, b * 4, 8, 8)
        return self.dec(h)

    def forward(self, idx, tokens, class_ids):
        c = self.cond(tokens, class_ids)
        mu, lv = self.encode(idx, c)
        z = self.reparam(mu, lv)
        return self.decode(z, c), mu, lv

    @torch.no_grad()
    def generate(self, tokens, class_ids, n=1, z=None):
        c = self.cond(tokens[:1].expand(n, -1), class_ids[:1].expand(n))
        if z is None:
            z = torch.randn(n, self.latent, device=c.device)
        return self.decode(z, c).argmax(1)
