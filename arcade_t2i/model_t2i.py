"""Text-conditioned CVAE for 64x64 pixel art sprite generation."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextEncoder(nn.Module):
    def __init__(self, n_tokens=100, emb=32, out=32):
        super().__init__()
        self.emb = nn.Embedding(n_tokens, emb, padding_idx=0)
        self.proj = nn.Sequential(nn.Linear(emb, out), nn.SiLU(), nn.Linear(out, out))
    def forward(self, tokens):
        e = self.emb(tokens.clamp(0, self.emb.num_embeddings - 1))
        m = (tokens > 0).float().unsqueeze(-1)
        pooled = (e * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.proj(pooled)

class TextCVAE(nn.Module):
    def __init__(self, n_colors=48, n_tokens=100, n_classes=32, latent=64, base=24):
        super().__init__()
        self.latent = latent
        self.color_emb = nn.Embedding(n_colors, 12)
        self.text = TextEncoder(n_tokens, 32, 32)
        self.class_emb = nn.Embedding(n_classes, 32)
        self.enc = nn.Sequential(
            nn.Conv2d(12, base, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(base, base * 2, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(base * 4, base * 4, 4, 2, 1), nn.SiLU(),
        )
        self.fc_mu = nn.Linear(base * 4 * 8 * 8 + 64, latent)
        self.fc_lv = nn.Linear(base * 4 * 8 * 8 + 64, latent)
        self.dec_fc = nn.Linear(latent + 64, base * 4 * 8 * 8)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(base, base, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(base, n_colors, 3, 1, 1),
        )
        self.cond_dim = 64

    def cond(self, tokens, class_ids):
        return torch.cat([self.text(tokens), self.class_emb(class_ids.clamp(0, 31))], -1)

    def encode(self, x, c):
        h = self.color_emb(x.clamp(0, 47)).permute(0, 3, 1, 2).contiguous()
        h = self.enc(h).flatten(1)
        h = torch.cat([h, c], -1)
        return self.fc_mu(h), self.fc_lv(h)

    def reparam(self, mu, lv):
        return mu + torch.randn_like(mu) * (0.5 * lv).exp()

    def decode(self, z, c):
        h = self.dec_fc(torch.cat([z, c], -1)).view(z.size(0), -1, 8, 8)
        return self.dec(h)

    def forward(self, x, tokens, class_ids):
        c = self.cond(tokens, class_ids)
        mu, lv = self.encode(x, c)
        z = self.reparam(mu, lv)
        logits = self.decode(z, c)
        return logits, mu, lv

    @torch.no_grad()
    def generate(self, tokens, class_ids, n=1, z=None, temperature=1.0):
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if class_ids.dim() == 0:
            class_ids = class_ids.unsqueeze(0)
        tokens = tokens.expand(n, -1)
        class_ids = class_ids.expand(n)
        c = self.cond(tokens, class_ids)
        if z is None:
            z = torch.randn(n, self.latent, device=tokens.device)
        logits = self.decode(z, c)
        if temperature and temperature > 0:
            logits = logits / max(temperature, 1e-6)
        return logits.argmax(1)
