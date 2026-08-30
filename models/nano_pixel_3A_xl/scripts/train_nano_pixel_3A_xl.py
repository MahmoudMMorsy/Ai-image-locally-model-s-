import os
import torch
import torch.nn as nn
import torch.optim as optim

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
    dataset_dir = os.path.join(model_dir, "dataset")
    weights_dir = os.path.join(model_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    model = NanoPixel3AXLUNet()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    print("Training NanoPixel 3A XL model...")
    for epoch in range(1, 11):
        x = torch.randn(2, 4, 64, 64)
        t = torch.tensor([5.0, 10.0])
        optimizer.zero_grad()
        pred = model(x, t)
        loss = torch.mean((pred - x)**2)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch}/10] Loss: {loss.item():.4f}")

    weight_path = os.path.join(weights_dir, "nanopixel_3A_xl.pt")
    torch.save(model.state_dict(), weight_path)
    print(f"NanoPixel 3A XL model weights saved successfully to: {weight_path}")

if __name__ == "__main__":
    main()
