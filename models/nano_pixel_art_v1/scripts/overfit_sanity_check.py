"""
Overfitting Sanity Check Script for NanoPixel-v1
Trains on 10 synthetic pixel art images for 500 steps to verify that the Pixel-Space UNet
can perfectly overfit and reconstruct images given their exact prompt.
"""
import torch
import torch.optim as optim
import numpy as np
from PIL import Image
from models.nano_pixel_art_v1.scripts.train_nanopixel import NanoPixelUNet, CharbonnierLoss
from models.nano_pixel_art_v1.scripts.train_nano_pixel import caption_to_prompt_embedding

def run_overfitting_sanity_check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Running Overfitting Sanity Check (10 images, 500 steps) on {device} ===")

    # 1. Create 10 distinct synthetic 64x64 pixel art images and captions
    images = []
    conds = []

    for i in range(10):
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        arr[16:48, 16:48, 0] = (i * 25) % 255 # red pattern
        arr[16:48, 16:48, 1] = (255 - i * 25) % 255 # green pattern
        arr[16:48, 16:48, 2] = 200 # blue pattern
        arr[16:48, 16:48, 3] = 255 # opaque

        t_img = torch.from_numpy(arr.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        caption = f"pixel art, 16-bit, gba style, character_{i}, idle movement"
        t_cond = caption_to_prompt_embedding(caption)

        images.append(t_img)
        conds.append(t_cond)

    x0_batch = torch.stack(images).to(device) # (10, 4, 64, 64)
    c_batch = torch.stack(conds).to(device)   # (10, 64)

    model = NanoPixelUNet().to(device)
    criterion = CharbonnierLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3) # higher LR for fast 500-step overfit test

    model.train()
    print("Training 500 steps...")
    for step in range(1, 501):
        t = torch.randint(0, 10, (10,), device=device).long()
        noise = torch.randn_like(x0_batch)

        alpha_t = 1.0 - (t.float() / 20.0 + 0.01)[..., None, None, None]
        xt = torch.sqrt(alpha_t) * x0_batch + torch.sqrt(1.0 - alpha_t) * noise

        pred_noise = model(xt, t, c_batch)
        loss = criterion(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step [{step}/500] Loss: {loss.item():.5f}")

    # Verify reconstruction
    model.eval()
    with torch.no_grad():
        t0 = torch.zeros(10, device=device).long()
        rec_noise = model(x0_batch, t0, c_batch)
        rec_error = criterion(rec_noise, torch.zeros_like(rec_noise)).item()

    print(f"Sanity Check Final Reconstruction Loss: {rec_error:.5f}")
    if rec_error < 0.2:
        print("✅ OVERFITTING SANITY CHECK PASSED: Model successfully overfits and reconstructs pixel space targets!")
    else:
        print("⚠️ SANITY CHECK WARNING: High reconstruction error.")

if __name__ == "__main__":
    run_overfitting_sanity_check()
