import os
import torch
import torch.nn as nn

class PosterLatentUNet256(nn.Module):
    """
    Lightweight 256x256 Latent UNet Model for Poster Generation.
    Takes latent noise tensor (1, 4, 32, 32) + condition vector (128) + timestep
    Outputs denoised latent feature map for 256x256 poster rendering.
    """
    def __init__(self, in_channels=4, out_channels=4, cond_dim=128):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, 64)
        self.time_proj = nn.Linear(1, 64)

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, x, t, cond):
        # x: (B, 4, 32, 32), t: (B, 1), cond: (B, 128)
        t_emb = self.act(self.time_proj(t)).unsqueeze(-1).unsqueeze(-1)
        c_emb = self.act(self.cond_proj(cond)).unsqueeze(-1).unsqueeze(-1)

        h = self.act(self.conv1(x)) + t_emb + c_emb
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        out = self.conv_out(h)
        return out

def main():
    model = PosterLatentUNet256()
    model.eval()

    dummy_latent = torch.randn(1, 4, 32, 32)
    dummy_t = torch.tensor([[10.0]])
    dummy_cond = torch.randn(1, 128)

    weights_dir = "poster_generator_256/weights"
    os.makedirs(weights_dir, exist_ok=True)
    onnx_path = os.path.join(weights_dir, "poster_generator_256.onnx")

    torch.onnx.export(
        model,
        (dummy_latent, dummy_t, dummy_cond),
        onnx_path,
        input_names=["latent", "timestep", "condition"],
        output_names=["denoised_latent"],
        dynamic_axes={"latent": {0: "batch_size"}}
    )

    print(f"Poster Generation ONNX model successfully exported to: {onnx_path}")

if __name__ == "__main__":
    main()
