import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

class PixelLLM(nn.Module):
    """
    Fast Autoregressive Palette Token Transformer (~200K params)
    Optimized for fast CPU execution and fast sampling.
    """
    def __init__(self, vocab_size=18, max_seq_len=1024, d_model=96, nhead=4, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size # 0: BG, 1-16: Palette colors, 17: BOS token
        self.max_seq_len = max_seq_len
        self.bos_token = 17

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len + 1, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        b, t = x.size()
        tok_emb = self.token_emb(x)
        p_emb = self.pos_emb[:, :t, :]
        h = tok_emb + p_emb

        mask = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
        out = self.transformer(h, mask=mask)
        logits = self.head(out)
        return logits

    @torch.no_grad()
    def generate(self, device='cpu', temperature=0.85, top_k=8):
        self.eval()
        x = torch.full((1, 1), self.bos_token, dtype=torch.long, device=device)
        for i in range(self.max_seq_len):
            logits = self(x)[:, -1, :]
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)

        indices = x[0, 1:].view(32, 32).cpu().numpy()
        return indices

def train_pixel_llm():
    print("Training Model 1: PixelLLM (Fast Autoregressive Palette Transformer)...")
    dataset_path = "models/nano_pixel_mob_2026_09_12/dataset/arcade_sprites_32x32.pt"
    data = torch.load(dataset_path)
    indices = data["indices"] # (N, 32, 32)
    palettes = data["palettes"] # (N, 16, 3)

    N = indices.size(0)
    flat_indices = indices.view(N, 1024)
    bos = torch.full((N, 1), 17, dtype=torch.long)
    seqs = torch.cat([bos, flat_indices], dim=1) # (N, 1025)

    model = PixelLLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)

    epochs = 12
    batch_size = 16
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            batch_seqs = seqs[idx]

            inputs = batch_seqs[:, :-1]
            targets = batch_seqs[:, 1:]

            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, 18), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/N:.4f}")

    out_dir = "models/nano_pixel_mob_2026_09_12/1_pixel_llm"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "pixel_llm_weights.pt"))
    print(f"Model 1 weights saved to {out_dir}/pixel_llm_weights.pt")

    # Generate test sample
    print("Generating sample character sprite from PixelLLM...")
    sample_indices = model.generate(temperature=0.85, top_k=8)

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
    img_64.save(os.path.join(out_dir, "sample_pixelllm_64x64.png"))
    print(f"Saved sample image to {out_dir}/sample_pixelllm_64x64.png")

if __name__ == "__main__":
    train_pixel_llm()
