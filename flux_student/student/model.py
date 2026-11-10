"""Project student model: compact text-to-image denoiser (32/64/128/256)."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, c: int, tdim: int):
        super().__init__()
        self.n1 = nn.GroupNorm(8, c)
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.n2 = nn.GroupNorm(8, c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.t = nn.Linear(tdim, c * 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.t(t).chunk(2, dim=1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return x + h


class StudentT2I(nn.Module):
    """Compact text-conditioned epsilon predictor. base=96 ~27M params."""

    def __init__(self, base: int = 96, tdim: int = 384, cdim: int = 384, vocab: int = 16384):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, 192, padding_idx=0)
        self.txt = nn.Sequential(nn.Linear(192, cdim), nn.SiLU(), nn.Linear(cdim, cdim))
        self.time = nn.Sequential(nn.Linear(128, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.cond = nn.Linear(cdim, tdim)
        self.in_c = nn.Conv2d(3, base, 3, padding=1)
        self.d1 = nn.ModuleList([ResBlock(base, tdim) for _ in range(2)])
        self.ds1 = nn.Conv2d(base, base * 2, 4, 2, 1)
        self.d2 = nn.ModuleList([ResBlock(base * 2, tdim) for _ in range(2)])
        self.ds2 = nn.Conv2d(base * 2, base * 3, 4, 2, 1)
        self.d3 = nn.ModuleList([ResBlock(base * 3, tdim) for _ in range(2)])
        self.ds3 = nn.Conv2d(base * 3, base * 4, 4, 2, 1)
        self.mid = nn.ModuleList([ResBlock(base * 4, tdim) for _ in range(2)])
        self.us3 = nn.ConvTranspose2d(base * 4, base * 3, 4, 2, 1)
        self.u3 = nn.ModuleList([ResBlock(base * 3, tdim) for _ in range(2)])
        self.us2 = nn.ConvTranspose2d(base * 3, base * 2, 4, 2, 1)
        self.u2 = nn.ModuleList([ResBlock(base * 2, tdim) for _ in range(2)])
        self.us1 = nn.ConvTranspose2d(base * 2, base, 4, 2, 1)
        self.u1 = nn.ModuleList([ResBlock(base, tdim) for _ in range(2)])
        self.out = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, 3, 3, padding=1))

    def temb(self, t: torch.Tensor, dim: int = 128) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device) * (math.log(10000.0) / half))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def encode_text(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.emb(ids.clamp(0, self.vocab - 1))
        m = (ids > 0).float().unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.txt(pooled)

    def forward(self, x: torch.Tensor, t: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        te = self.time(self.temb(t)) + self.cond(self.encode_text(ids))
        h = self.in_c(x)
        for b in self.d1:
            h = b(h, te)
        h = self.ds1(h)
        for b in self.d2:
            h = b(h, te)
        h = self.ds2(h)
        for b in self.d3:
            h = b(h, te)
        h = self.ds3(h)
        for b in self.mid:
            h = b(h, te)
        h = self.us3(h)
        for b in self.u3:
            h = b(h, te)
        h = self.us2(h)
        for b in self.u2:
            h = b(h, te)
        h = self.us1(h)
        for b in self.u1:
            h = b(h, te)
        return self.out(h)

    @torch.no_grad()
    def sample(self, ids: torch.Tensor, size: int = 64, steps: int = 20, device=None) -> torch.Tensor:
        device = device or ids.device
        x = torch.randn(ids.size(0), 3, size, size, device=device)
        for i, ti in enumerate(reversed(range(0, 1000, max(1, 1000 // steps)))):
            t = torch.full((ids.size(0),), ti, device=device, dtype=torch.long)
            eps = self.forward(x, t, ids)
            alpha = 1.0 - (ti / 1000.0) * 0.9
            x = (x - (1 - alpha) * eps) / max(alpha, 1e-3) ** 0.5
            if i < steps - 1:
                x = x + torch.randn_like(x) * (ti / 1000.0) * 0.2
        return x.clamp(-1, 1)


class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper for nn.Linear."""
    def __init__(self, original_linear: nn.Linear, r: int = 4, lora_alpha: float = 1.0):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.scaling = lora_alpha / r
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * (1.0 / math.sqrt(in_features)))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original_linear(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# Add inject_lora helper to StudentT2I
def inject_lora(model: StudentT2I, r: int = 4, scale: float = 1.0):
    """Lightweight hook allowing low-rank adapter fine-tuning of the text conditioning projection."""
    if isinstance(model.cond, nn.Linear):
        model.cond = LoRALinear(model.cond, r=r, lora_alpha=scale)
    return model


StudentT2I.inject_lora = inject_lora


if __name__ == "__main__":
    m = StudentT2I()
    n = count_params(m)
    print(f"params={n/1e6:.2f}M  fp16~{n*2/1e6:.0f}MB  fp32~{n*4/1e6:.0f}MB")
