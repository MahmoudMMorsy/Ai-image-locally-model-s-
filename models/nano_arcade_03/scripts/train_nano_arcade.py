"""
Nano Arcade 03 Full Pixel-Space Diffusion Trainer & ONNX Exporter
Trains on all samples in 'mm.trine' across subcategories with noise scheduling, text embedding conditioning, and ONNX export.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np

class NanoArcade03UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, cond_dim=64):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 32)
        )
        self.cond_proj = nn.Linear(cond_dim, 32)

        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1)
        )
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t, cond):
        if t.dim() == 1:
            t = t.float().unsqueeze(1)
        t_emb = self.time_embed(t.float())
        c_emb = self.cond_proj(cond)
        emb = t_emb + c_emb # (B, 32)

        h = self.conv_in(x)
        emb_spatial = emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h.shape[2], h.shape[3])
        h = h + torch.cat([emb_spatial, emb_spatial], dim=1) # (B, 64, 64, 64)
        h = self.block1(h)
        return torch.sigmoid(self.conv_out(h))

def text_to_embedding(text: str, dim: int = 64) -> torch.Tensor:
    vec = torch.zeros(dim)
    words = text.lower().replace(',', '').split()
    for idx, word in enumerate(words):
        h = sum(ord(c) for c in word)
        vec[h % dim] += 1.0 / (idx + 1)
    norm = torch.norm(vec)
    return vec / (norm + 1e-6)

def train_and_export():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Nano Arcade 03] Initializing training pipeline on 'mm.trine' on {device}...")

    model = NanoArcade03UNet().to(device)
    dataset_dir = "mm.trine"

    dataset = []
    if os.path.exists(dataset_dir):
        files = [f for f in os.listdir(dataset_dir) if f.endswith(".png")]
        print(f"[Nano Arcade 03] Found {len(files)} dataset images in {dataset_dir}.")
        for f in files:
            img_path = os.path.join(dataset_dir, f)
            txt_path = os.path.join(dataset_dir, f.replace(".png", ".txt"))
            caption = "pixel art 64x64 character sprite"
            if os.path.exists(txt_path):
                with open(txt_path, "r") as tf:
                    caption = tf.read().strip()

            img = Image.open(img_path).convert("RGBA").resize((64, 64))
            arr = np.array(img, dtype=np.float32) / 255.0
            t_img = torch.from_numpy(arr).permute(2, 0, 1)
            t_cond = text_to_embedding(caption, dim=64)
            dataset.append((t_img, t_cond))

    if len(dataset) > 0:
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.MSELoss()
        model.train()
        print(f"[Nano Arcade 03] Training DDPM diffusion model on ALL {len(dataset)} dataset samples...")
        num_epochs = 20
        for epoch in range(1, num_epochs + 1):
            total_loss = 0.0
            for t_img, t_cond in dataset:
                x0 = t_img.unsqueeze(0).to(device)
                c = t_cond.unsqueeze(0).to(device)
                t = torch.randint(0, 20, (1,), device=device).float()

                noise = torch.randn_like(x0)
                alpha_t = 1.0 - (t / 20.0 + 0.01)[..., None, None, None]
                xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(1.0 - alpha_t) * noise

                pred = model(xt, t, c)
                loss = criterion(pred, x0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 2 == 0 or epoch == num_epochs:
                print(f"Epoch [{epoch}/{num_epochs}] Diffusion Training Loss: {total_loss / len(dataset):.4f}")

    weights_dir = "models/nano_arcade_03/weights"
    os.makedirs(weights_dir, exist_ok=True)
    pt_path = os.path.join(weights_dir, "nano_arcade_03.pt")
    onnx_path = os.path.join(weights_dir, "nano_arcade_03.onnx")

    model_cpu = model.to("cpu").eval()
    torch.save(model_cpu.state_dict(), pt_path)
    print(f"[Nano Arcade 03] Saved PyTorch weights to {pt_path}")

    dummy_x = torch.randn(1, 4, 64, 64)
    dummy_t = torch.tensor([5.0])
    dummy_c = torch.randn(1, 64)

    torch.onnx.export(
        model_cpu,
        (dummy_x, dummy_t, dummy_c),
        onnx_path,
        input_names=["input_image", "timestep", "text_embed"],
        output_names=["output_sprite"],
        opset_version=18
    )
    print(f"[Nano Arcade 03] Successfully exported ONNX model to {onnx_path}")

if __name__ == "__main__":
    train_and_export()
