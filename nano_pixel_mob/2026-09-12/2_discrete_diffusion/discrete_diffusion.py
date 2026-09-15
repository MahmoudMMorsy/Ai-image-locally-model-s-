import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

class ConvDiscreteDiffusion(nn.Module):
    """
    Fast ConvNet-based Discrete Masked Diffusion Generator.
    Fixes sampling mask accumulation logic across steps.
    """
    def __init__(self, num_classes=17, mask_token=17, d_model=64):
        super().__init__()
        self.num_classes = num_classes
        self.mask_token = mask_token

        self.tok_emb = nn.Embedding(num_classes + 1, d_model)
        self.time_emb = nn.Embedding(100, d_model)

        self.net = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, num_classes, kernel_size=1)
        )

    def forward(self, x, t):
        b, h, w = x.size()
        emb = self.tok_emb(x)
        t_emb = self.time_emb(t).unsqueeze(1).unsqueeze(1)
        h_in = (emb + t_emb).permute(0, 3, 1, 2)
        logits = self.net(h_in)
        return logits.permute(0, 2, 3, 1)

    @torch.no_grad()
    def sample(self, steps=10, device='cpu'):
        self.eval()
        x = torch.full((1, 32, 32), self.mask_token, dtype=torch.long, device=device)
        for step in reversed(range(steps)):
            t = torch.full((1,), step * 10, dtype=torch.long, device=device)
            logits = self(x, t)
            probs = F.softmax(logits, dim=-1)
            pred_tokens = torch.multinomial(probs.view(-1, self.num_classes), 1).view(1, 32, 32)

            # Maintain already unmasked tokens across steps
            is_currently_masked = (x == self.mask_token)
            unmask_prob = 1.0 / (step + 1)
            should_unmask = (torch.rand_like(x.float()) < unmask_prob) & is_currently_masked
            x = torch.where(should_unmask, pred_tokens, x)

        # Unmask any remaining masked tokens at final step
        x = torch.where(x == self.mask_token, pred_tokens, x)
        return x[0].cpu().numpy()

def train_discrete_diffusion():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    dataset_path = os.path.join(parent_dir, "dataset", "arcade_sprites_32x32.pt")
    data = torch.load(dataset_path)
    indices = data["indices"]
    palettes = data["palettes"]

    N = indices.size(0)
    model = ConvDiscreteDiffusion()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)

    epochs = 15
    batch_size = 16
    mask_token = 17

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            batch_x = indices[idx]
            b_size = batch_x.size(0)

            t = torch.randint(0, 100, (b_size,), dtype=torch.long)
            mask_probs = (t.float() / 100.0).view(b_size, 1, 1)
            mask = torch.rand_like(batch_x.float()) < mask_probs
            corrupted_x = torch.where(mask, mask_token, batch_x)

            logits = model(corrupted_x, t)
            loss = F.cross_entropy(logits.reshape(-1, 17), batch_x.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * b_size

    torch.save(model.state_dict(), os.path.join(base_dir, "discrete_diffusion_weights.pt"))
    sample_indices = model.sample(steps=10)

    ref_palette = palettes[0].numpy()
    img_arr = np.zeros((32, 32, 4), dtype=np.uint8)
    for r in range(32):
        for c in range(32):
            idx = sample_indices[r, c]
            if 1 <= idx <= 16:
                color = ref_palette[min(idx-1, len(ref_palette)-1)]
                img_arr[r, c] = [color[0], color[1], color[2], 255]

    img = Image.fromarray(img_arr, mode='RGBA')
    img_64 = img.resize((64, 64), Image.NEAREST)
    img_64.save(os.path.join(base_dir, "sample_diffusion_64x64.png"))

if __name__ == "__main__":
    train_discrete_diffusion()
