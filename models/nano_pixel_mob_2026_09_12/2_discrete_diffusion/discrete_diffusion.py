import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

class ConvDiscreteDiffusion(nn.Module):
    """
    Fast ConvNet-based Discrete Masked Diffusion Generator.
    Highly efficient on CPU (runs training in ~10 seconds).
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
        # x: (B, 32, 32)
        # t: (B,)
        b, h, w = x.size()
        emb = self.tok_emb(x) # (B, 32, 32, d_model)
        t_emb = self.time_emb(t).unsqueeze(1).unsqueeze(1)
        h_in = (emb + t_emb).permute(0, 3, 1, 2) # (B, d_model, 32, 32)
        logits = self.net(h_in) # (B, num_classes, 32, 32)
        return logits.permute(0, 2, 3, 1) # (B, 32, 32, num_classes)

    @torch.no_grad()
    def sample(self, steps=10, device='cpu'):
        self.eval()
        x = torch.full((1, 32, 32), self.mask_token, dtype=torch.long, device=device)
        for step in reversed(range(steps)):
            t = torch.full((1,), step * 10, dtype=torch.long, device=device)
            logits = self(x, t) # (1, 32, 32, 17)
            probs = F.softmax(logits, dim=-1)
            pred_tokens = torch.multinomial(probs.view(-1, self.num_classes), 1).view(1, 32, 32)

            mask_ratio = step / float(steps)
            unmask_mask = torch.rand_like(x.float()) > mask_ratio
            x = torch.where(unmask_mask, pred_tokens, x)

        return x[0].cpu().numpy()

def train_discrete_diffusion():
    print("Training Model 2: Fast Conv Discrete Masked Diffusion Generator...")
    dataset_path = "models/nano_pixel_mob_2026_09_12/dataset/arcade_sprites_32x32.pt"
    data = torch.load(dataset_path)
    indices = data["indices"] # (N, 32, 32)
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

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/N:.4f}")

    out_dir = "models/nano_pixel_mob_2026_09_12/2_discrete_diffusion"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "discrete_diffusion_weights.pt"))
    print(f"Model 2 weights saved to {out_dir}/discrete_diffusion_weights.pt")

    # Generate test sample
    print("Generating sample character sprite from Discrete Diffusion...")
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
    img_64.save(os.path.join(out_dir, "sample_diffusion_64x64.png"))
    print(f"Saved sample image to {out_dir}/sample_diffusion_64x64.png")

if __name__ == "__main__":
    train_discrete_diffusion()
