import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
from models.nano_pixel_3A_xl.scripts.export_onnx import main as export_onnx_main

class NanoPixel3AXLUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, time_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv_mid = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.relu = nn.GELU()

    def forward(self, x, t):
        t_emb = self.time_mlp(t.unsqueeze(-1)).unsqueeze(-1).unsqueeze(-1)
        h = self.relu(self.conv_in(x)) + t_emb
        h = self.relu(self.conv_mid(h))
        return self.conv_out(h)

def main():
    model_dir = "models/nano_pixel_3A_xl"
    dataset_dir = "dataset_training_images/clean"
    weights_dir = os.path.join(model_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    images = []
    if os.path.exists(dataset_dir):
        files = [f for f in os.listdir(dataset_dir) if f.endswith(".png")][:30]
        for f in files:
            img_path = os.path.join(dataset_dir, f)
            img = Image.open(img_path).convert("RGBA").resize((64, 64), Image.Resampling.NEAREST)
            arr = np.array(img, dtype=np.float32) / 255.0
            images.append(torch.from_numpy(arr).permute(2, 0, 1))

    if images:
        real_batch = torch.stack(images[:8])
    else:
        real_batch = torch.randn(2, 4, 64, 64)

    model = NanoPixel3AXLUNet()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    print(f"Training NanoPixel 3A XL model on {len(images)} real images...")
    for epoch in range(1, 11):
        t = torch.tensor([5.0] * real_batch.size(0))
        noise = torch.randn_like(real_batch)
        noisy_x = real_batch + 0.1 * noise
        optimizer.zero_grad()
        pred = model(noisy_x, t)
        loss = torch.mean((pred - real_batch)**2)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch}/10] Loss: {loss.item():.4f}")

    weight_path = os.path.join(weights_dir, "nanopixel_3A_xl.pt")
    torch.save(model.state_dict(), weight_path)
    print(f"NanoPixel 3A XL model weights saved to: {weight_path}")

    export_onnx_main()

if __name__ == "__main__":
    main()
