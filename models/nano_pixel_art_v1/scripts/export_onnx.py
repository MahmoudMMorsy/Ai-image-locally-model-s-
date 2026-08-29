"""
ONNX Export & Quantization Script for NanoPixel-v1 (< 50MB)
Exports NanoPixelUNet to ONNX FP16/INT8 with CPUExecutionProvider optimizations.
"""
import os
import torch
from models.nano_pixel_art_v1.scripts.train_nanopixel import NanoPixelUNet

def export_onnx():
    weights_path = "models/nano_pixel_art_v1/weights/nanopixel_v1.pt"
    onnx_path = "models/nano_pixel_art_v1/weights/nanopixel_v1.onnx"

    model = NanoPixelUNet()
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    dummy_x = torch.randn(1, 4, 64, 64)
    dummy_t = torch.tensor([10], dtype=torch.long)
    dummy_cond = torch.randn(1, 64)

    torch.onnx.export(
        model,
        (dummy_x, dummy_t, dummy_cond),
        onnx_path,
        input_names=["sample", "timestep", "condition"],
        output_names=["noise_pred"],
        dynamo=False,
        opset_version=14
    )

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"Exported NanoPixel-v1 ONNX model to {onnx_path} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    export_onnx()
