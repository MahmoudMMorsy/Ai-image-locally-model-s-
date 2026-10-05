"""
Nano Arcade 01 Model Suite and Pipeline Runner
----------------------------------------------
Provides inference, palette extraction, fine-tuning, ONNX export, and showcase generation
for the Arcade Pixel Art UNet + Latent Autoencoder model suite (`models/nano_arcade_01`).
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np


class NanoArcadeUNet(nn.Module):
    """64x64 Arcade Pixel UNet Autoencoder and Latent Generator."""
    def __init__(self, n_colors: int = 48, latent: int = 128, base: int = 48):
        super().__init__()
        self.n_colors = n_colors
        self.latent = latent
        self.emb = nn.Embedding(n_colors, 16)

        self.e1 = nn.Sequential(nn.Conv2d(16, base, 3, 1, 1), nn.ReLU(True))
        self.e2 = nn.Sequential(nn.Conv2d(base, base * 2, 4, 2, 1), nn.ReLU(True))
        self.e3 = nn.Sequential(nn.Conv2d(base * 2, base * 4, 4, 2, 1), nn.ReLU(True))
        self.e4 = nn.Sequential(nn.Conv2d(base * 4, base * 4, 4, 2, 1), nn.ReLU(True))
        self.fc = nn.Linear(base * 4 * 8 * 8, latent)

        self.fc_up = nn.Linear(latent, base * 4 * 8 * 8)
        self.d4 = nn.Sequential(nn.ConvTranspose2d(base * 4, base * 4, 4, 2, 1), nn.ReLU(True))
        self.d3 = nn.Sequential(nn.ConvTranspose2d(base * 8, base * 2, 4, 2, 1), nn.ReLU(True))
        self.d2 = nn.Sequential(nn.ConvTranspose2d(base * 4, base, 4, 2, 1), nn.ReLU(True))
        self.out = nn.Conv2d(base * 2, n_colors, 3, 1, 1)
        self._base = base

    def encode(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.emb(idx).permute(0, 3, 1, 2)
        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        return self.fc(x4.flatten(1))

    def decode_logits(self, z: torch.Tensor, skips=None) -> torch.Tensor:
        b = self._base
        h = self.fc_up(z).view(-1, b * 4, 8, 8)
        h = self.d4(h)
        if skips is not None:
            h = torch.cat([h, skips["x3"]], 1)
        else:
            h = torch.cat([h, h], 1)
        h = self.d3(h)
        if skips is not None:
            h = torch.cat([h, skips["x2"]], 1)
        else:
            h = torch.cat([h, h], 1)
        h = self.d2(h)
        if skips is not None:
            h = torch.cat([h, skips["x1"]], 1)
        else:
            h = torch.cat([h, h], 1)
        return self.out(h)

    def forward(self, idx: torch.Tensor):
        x = self.emb(idx).permute(0, 3, 1, 2)
        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        z = self.fc(x4.flatten(1))
        skips = {"x1": x1, "x2": x2, "x3": x3}
        return self.decode_logits(z, skips), z


class LatentDecoder(nn.Module):
    """Decode z -> palette logits for neural image generation without skips."""
    def __init__(self, n_colors: int = 48, latent: int = 128, base: int = 48):
        super().__init__()
        self.fc = nn.Linear(latent, base * 4 * 8 * 8)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base * 4, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(base, n_colors, 3, 1, 1),
        )
        self._base = base

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, self._base * 4, 8, 8)
        return self.net(h)


class NanoArcadeGenerator:
    """High-level wrapper for training, ONNX export, and generating arcade pixel sprites."""
    def __init__(self, model_dir: str = "models/nano_arcade_01"):
        self.model_dir = Path(model_dir)
        self.ckpt_dir = self.model_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.unet = NanoArcadeUNet(n_colors=48, latent=128, base=48)
        self.decoder = LatentDecoder(n_colors=48, latent=128, base=48)

        self.palette = None
        self.load_weights()

    def load_weights(self):
        pal_path = self.ckpt_dir / "palette48_full.pt"
        if pal_path.exists():
            self.palette = torch.load(pal_path, map_location="cpu", weights_only=False)
        else:
            # Fallback default 48 RGBA palette
            self.palette = torch.zeros((48, 4), dtype=torch.float32)

        unet_ckpt = self.ckpt_dir / "palette_full_best.pt"
        if unet_ckpt.exists():
            ckpt = torch.load(unet_ckpt, map_location="cpu", weights_only=False)
            if "model" in ckpt:
                self.unet.load_state_dict(ckpt["model"], strict=False)
            if "decoder" in ckpt:
                self.decoder.load_state_dict(ckpt["decoder"], strict=False)

    def fine_tune(self, image_folder: str, epochs: int = 2, lr: float = 1e-3):
        """Quick fine-tuning pass on real dataset images."""
        img_paths = list(Path(image_folder).glob("*.png"))[:300]
        if not img_paths:
            return

        images = []
        for p in img_paths:
            img = Image.open(p).convert("RGBA").resize((64, 64), Image.NEAREST)
            images.append(np.array(img, dtype=np.float32) / 255.0)

        imgs_tensor = torch.tensor(np.stack(images)).permute(0, 3, 1, 2) # (B, 4, 64, 64)

        # Convert images to palette indices
        flat = imgs_tensor.permute(0, 2, 3, 1).reshape(-1, 4) * 255.0
        pal = self.palette.to(imgs_tensor.device)
        dist = ((flat[:, None, :] - pal[None, :, :]) ** 2).sum(-1)
        indices = dist.argmin(1).view(len(img_paths), 64, 64)

        opt = torch.optim.Adam(list(self.unet.parameters()) + list(self.decoder.parameters()), lr=lr)

        self.unet.train()
        self.decoder.train()

        dataset = torch.utils.data.TensorDataset(indices)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

        for ep in range(epochs):
            for (batch_idx,) in loader:
                logits, z = self.unet(batch_idx)
                loss_unet = F.cross_entropy(logits, batch_idx)

                logits_dec = self.decoder(z.detach())
                loss_dec = F.cross_entropy(logits_dec, batch_idx)

                loss = loss_unet + loss_dec
                opt.zero_grad()
                loss.backward()
                opt.step()

        # Save updated checkpoint
        torch.save({
            "model": self.unet.state_dict(),
            "decoder": self.decoder.state_dict(),
            "n_colors": 48,
            "latent": 128,
            "base": 48,
            "arch": "UNet+LatentDecoder"
        }, self.ckpt_dir / "palette_full_best.pt")

    def export_onnx(self, output_path: str = None) -> str:
        """Exports the latent generator decoder to ONNX format."""
        if output_path is None:
            weights_dir = self.model_dir / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(weights_dir / "nano_arcade_generator.onnx")

        self.decoder.eval()
        dummy_latent = torch.randn(1, 128)

        torch.onnx.export(
            self.decoder,
            dummy_latent,
            output_path,
            input_names=["latent"],
            output_names=["logits"],
            dynamic_axes={"latent": {0: "batch_size"}, "logits": {0: "batch_size"}},
            opset_version=14,
            dynamo=False,
        )
        return output_path

    def generate_sprite(self, z: torch.Tensor = None) -> Image.Image:
        """Generate a sharp 64x64 pixel sprite image."""
        self.decoder.eval()
        if z is None:
            z = torch.randn(1, 128)
        with torch.no_grad():
            logits = self.decoder(z)
            pred_idx = logits.argmax(1) # (1, 64, 64)
            pal = self.palette / 255.0
            rgba = F.embedding(pred_idx, pal).permute(0, 3, 1, 2)[0] # (4, 64, 64)
            arr = (rgba.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(arr, "RGBA")


if __name__ == "__main__":
    gen = NanoArcadeGenerator()
    onnx_file = gen.export_onnx()
    print(f"Exported ONNX model to: {onnx_file}")
    img = gen.generate_sprite()
    img.save("test_nano_arcade.png")
    print("Saved test image to test_nano_arcade.png")
