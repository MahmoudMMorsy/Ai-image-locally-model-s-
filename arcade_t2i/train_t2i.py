"""Train TextCVAE generator on full arcade dataset - fast CPU execution."""
import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from model_t2i import TextCVAE
from text_lang import TextProcessor

torch.set_num_threads(os.cpu_count() or 4)

class ArcadeDataset(Dataset):
    def __init__(self, indices_path="arcade_t2i/checkpoints/indices_full_u8.pt", manifest_path="arcade_t2i/manifests/manifest_full.jsonl"):
        self.indices = torch.load(indices_path)
        self.text_proc = TextProcessor()

        self.rows = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))

        assert len(self.rows) == self.indices.shape[0], f"Mismatch manifest ({len(self.rows)}) vs indices ({self.indices.shape[0]})"

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img_idx = self.indices[idx]

        if idx % 2 == 0:
            caption = row["caption_ar"]
        else:
            caption = row["caption_en"]

        tokens = self.text_proc.encode(caption)
        class_id = torch.tensor(row["class_id"], dtype=torch.long)

        return img_idx, tokens, class_id

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("arcade_t2i/checkpoints", exist_ok=True)
    os.makedirs("arcade_t2i/samples", exist_ok=True)

    full_dataset = ArcadeDataset()
    print(f"Loaded full dataset with N = {len(full_dataset)} samples.")

    sample_indices = torch.randperm(len(full_dataset))[:2000].tolist()
    dataset = Subset(full_dataset, sample_indices)

    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

    model = TextCVAE(n_colors=48, latent=64, base=24, cond_dim=128, n_tokens=100).to(device)
    ckpt_path = "arcade_t2i/checkpoints/t2i_cvae_best.pt"

    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                print("Resumed weights from existing checkpoint!")
        except Exception as e:
            print("Starting fresh training:", e)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    color_weights = torch.ones(48, device=device)
    color_weights[0] = 0.2
    color_weights[1:] = 1.5

    best_loss = float("inf")

    print("\n--- Starting Fast CPU Training Loop ---")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_batches = 0

        beta = min(1.0, epoch / 6.0) * 0.005

        for idx, tokens, class_ids in loader:
            idx = idx.to(device)
            tokens = tokens.to(device)
            class_ids = class_ids.to(device)

            optimizer.zero_grad()
            logits, mu, lv = model(idx, tokens, class_ids)

            recon_loss = F.cross_entropy(logits, idx.long(), weight=color_weights)
            kl_loss = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / idx.size(0)

            loss = recon_loss + beta * kl_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_batches += 1

        scheduler.step()

        avg_recon = total_recon_loss / total_batches
        avg_kl = total_kl_loss / total_batches
        avg_total = avg_recon + beta * avg_kl

        print(f"Epoch {epoch:02d}/{epochs:02d} | Recon Loss: {avg_recon:.4f} | KL: {avg_kl:.4f} | Total: {avg_total:.4f}")

        if avg_total < best_loss:
            best_loss = avg_total
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_total,
            }, ckpt_path)

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.2f} seconds!")
    print(f"Best checkpoint saved to: {ckpt_path}")

if __name__ == "__main__":
    train()
