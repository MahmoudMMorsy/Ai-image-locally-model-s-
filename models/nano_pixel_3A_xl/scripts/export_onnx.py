import os
import torch
import torch.nn as nn

class NanoPixel3AXLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.relu = nn.GELU()
        self.conv2 = nn.Conv2d(32, 4, kernel_size=3, padding=1)

    def forward(self, x, t):
        return self.conv2(self.relu(self.conv1(x)))

def main():
    model = NanoPixel3AXLModel()
    model.eval()

    dummy_latent = torch.randn(1, 4, 128, 128)
    dummy_t = torch.tensor([10.0])

    weights_dir = "models/nano_pixel_3A_xl/weights"
    os.makedirs(weights_dir, exist_ok=True)
    onnx_path = os.path.join(weights_dir, "nanopixel_3A_xl.onnx")

    torch.onnx.export(
        model,
        (dummy_latent, dummy_t),
        onnx_path,
        input_names=["latent", "timestep"],
        output_names=["noise_pred"],
        dynamic_axes={"latent": {0: "batch_size"}}
    )

    print(f"ONNX Model successfully exported to: {onnx_path}")

if __name__ == "__main__":
    main()
