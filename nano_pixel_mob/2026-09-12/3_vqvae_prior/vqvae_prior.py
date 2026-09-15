import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

class VQVAEEncoder(nn.Module):
    def __init__(self, in_channels=17, hidden_dim=64, embedding_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, embedding_dim, kernel_size=3, padding=1)
        )
    def forward(self, x):
        return self.conv(x)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=32, embedding_dim=32, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z):
        z_permuted = z.permute(0, 2, 3, 1).contiguous()
        flat_z = z_permuted.view(-1, self.embedding_dim)

        distances = torch.sum(flat_z**2, dim=1, keepdim=True) + \
                    torch.sum(self.embeddings.weight**2, dim=1) - \
                    2 * torch.matmul(flat_z, self.embeddings.weight.t())

        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embeddings(encoding_indices).view(z_permuted.shape)

        q_loss = F.mse_loss(quantized.detach(), z_permuted)
        c_loss = F.mse_loss(quantized, z_permuted.detach())
        loss = q_loss + self.commitment_cost * c_loss

        quantized = z_permuted + (quantized - z_permuted).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        indices = encoding_indices.view(z.size(0), z.size(2), z.size(3))
        return quantized, loss, indices

class VQVAEDecoder(nn.Module):
    def __init__(self, embedding_dim=32, hidden_dim=64, out_channels=17):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
        )
    def forward(self, z):
        return self.deconv(z)

class VQVAEModel(nn.Module):
    def __init__(self, num_classes=17, num_embeddings=32, embedding_dim=32):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = VQVAEEncoder(num_classes, 64, embedding_dim)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = VQVAEDecoder(embedding_dim, 64, num_classes)

    def forward(self, x_indices):
        x_onehot = F.one_hot(x_indices, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        z = self.encoder(x_onehot)
        quantized, vq_loss, indices = self.vq(z)
        logits = self.decoder(quantized)
        return logits, vq_loss, indices

    @torch.no_grad()
    def decode_indices(self, indices):
        quantized = self.vq.embeddings(indices).permute(0, 3, 1, 2).contiguous()
        logits = self.decoder(quantized)
        preds = torch.argmax(logits, dim=1)
        return preds

class LatentPriorTransformer(nn.Module):
    def __init__(self, num_codes=32, d_model=64, num_layers=2):
        super().__init__()
        self.bos_token = num_codes
        self.emb = nn.Embedding(num_codes + 1, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 65, d_model))
        nn.init.normal_(self.pos, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model*2,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_codes)

    def forward(self, x):
        b, t = x.size()
        h = self.emb(x) + self.pos[:, :t, :]
        mask = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
        out = self.transformer(h, mask=mask)
        return self.head(out)

    @torch.no_grad()
    def generate(self, device='cpu', temperature=0.85):
        self.eval()
        x = torch.full((1, 1), self.bos_token, dtype=torch.long, device=device)
        for _ in range(64):
            logits = self(x)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            x = torch.cat((x, next_token), dim=1)
        return x[0, 1:].view(1, 8, 8)

def train_vqvae_prior():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    dataset_path = os.path.join(parent_dir, "dataset", "arcade_sprites_32x32.pt")
    data = torch.load(dataset_path)
    indices = data["indices"]
    palettes = data["palettes"]

    N = indices.size(0)

    vqvae = VQVAEModel()
    optimizer_vq = torch.optim.AdamW(vqvae.parameters(), lr=3e-3, weight_decay=0.01)

    epochs_vq = 15
    batch_size = 16

    for epoch in range(epochs_vq):
        vqvae.train()
        total_loss = 0
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            batch_x = indices[idx]
            b_size = batch_x.size(0)

            logits, vq_loss, _ = vqvae(batch_x)
            recon_loss = F.cross_entropy(logits, batch_x)
            loss = recon_loss + vq_loss

            optimizer_vq.zero_grad()
            loss.backward()
            optimizer_vq.step()
            total_loss += loss.item() * b_size

    vqvae.eval()
    with torch.no_grad():
        _, _, code_indices = vqvae(indices)

    code_flat = code_indices.view(N, 64)
    bos = torch.full((N, 1), 32, dtype=torch.long)
    seqs = torch.cat([bos, code_flat], dim=1)

    prior = LatentPriorTransformer()
    optimizer_prior = torch.optim.AdamW(prior.parameters(), lr=3e-3, weight_decay=0.01)

    epochs_prior = 15
    for epoch in range(epochs_prior):
        prior.train()
        total_loss = 0
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            batch_seqs = seqs[idx]

            inputs = batch_seqs[:, :-1]
            targets = batch_seqs[:, 1:]

            logits = prior(inputs)
            loss = F.cross_entropy(logits.reshape(-1, 32), targets.reshape(-1))

            optimizer_prior.zero_grad()
            loss.backward()
            optimizer_prior.step()
            total_loss += loss.item() * len(idx)

    torch.save(vqvae.state_dict(), os.path.join(base_dir, "vqvae_weights.pt"))
    torch.save(prior.state_dict(), os.path.join(base_dir, "prior_weights.pt"))

    sampled_codes = prior.generate()
    sample_indices = vqvae.decode_indices(sampled_codes)[0].cpu().numpy()

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
    img_64.save(os.path.join(base_dir, "sample_vqvae_prior_64x64.png"))

if __name__ == "__main__":
    train_vqvae_prior()
