import os
import torch
from models.nano_pixel_mom.model import NanoPixelMomUNet, NanoPixelMomDecoder

def export_onnx():
    weights_dir = "models/nano_pixel_mom/weights"
    os.makedirs(weights_dir, exist_ok=True)

    unet = NanoPixelMomUNet()
    decoder = NanoPixelMomDecoder()

    unet_weights = os.path.join(weights_dir, "nano_pixel_mom_unet.pt")
    decoder_weights = os.path.join(weights_dir, "nano_pixel_mom_decoder.pt")

    if os.path.exists(unet_weights):
        unet.load_state_dict(torch.load(unet_weights, map_location="cpu"))
    if os.path.exists(decoder_weights):
        decoder.load_state_dict(torch.load(decoder_weights, map_location="cpu"))

    unet.eval()
    decoder.eval()

    dummy_latent = torch.randn(1, 4, 64, 64)
    dummy_t = torch.tensor([10]).long()
    dummy_cond = torch.randn(1, 64)

    unet_onnx_path = os.path.join(weights_dir, "nano_pixel_mom_unet.onnx")
    decoder_onnx_path = os.path.join(weights_dir, "nano_pixel_mom_decoder.onnx")

    torch.onnx.export(
        unet,
        (dummy_latent, dummy_t, dummy_cond),
        unet_onnx_path,
        input_names=["latent", "timestep", "condition"],
        output_names=["noise_pred"],
        dynamic_axes={"latent": {0: "batch_size"}},
        dynamo=False
    )

    torch.onnx.export(
        decoder,
        dummy_latent,
        decoder_onnx_path,
        input_names=["latent"],
        output_names=["image"],
        dynamic_axes={"latent": {0: "batch_size"}},
        dynamo=False
    )

    print(f"Exported UNet ONNX to {unet_onnx_path} ({os.path.getsize(unet_onnx_path)} bytes)")
    print(f"Exported Decoder ONNX to {decoder_onnx_path} ({os.path.getsize(decoder_onnx_path)} bytes)")

if __name__ == "__main__":
    export_onnx()
