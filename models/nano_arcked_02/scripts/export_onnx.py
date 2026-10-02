import os
import torch
from models.nano_arcked_02.scripts.train_nano_arcked_02 import NanoArcKed02UNet

def export_onnx():
    weights_dir = "models/nano_arcked_02/weights"
    os.makedirs(weights_dir, exist_ok=True)

    pt_path = os.path.join(weights_dir, "nano_arcked_02.pt")
    onnx_path = os.path.join(weights_dir, "nano_arcked_02.onnx")

    model = NanoArcKed02UNet()
    if os.path.exists(pt_path):
        model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model.eval()

    dummy_x = torch.randn(1, 4, 64, 64)
    dummy_t = torch.tensor([10], dtype=torch.long)
    dummy_cond = torch.randn(1, 64)

    torch.onnx.export(
        model,
        (dummy_x, dummy_t, dummy_cond),
        onnx_path,
        input_names=["latent", "timestep", "text_embed"],
        output_names=["denoised_latent"],
        dynamic_axes={"latent": {0: "batch_size"}},
        dynamo=False
    )
    print(f"[NanoArcKed-02] ONNX model exported successfully to: {onnx_path}")

if __name__ == "__main__":
    export_onnx()
