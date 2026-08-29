"""
Verification and Character Matrix Benchmark Script for NanoPixel-v1
Generates a character matrix across 5 diverse character prompts:
(Mage, Cyberpunk Rogue, Retro Orc, Mecha, Space Marine) at 64x64 and 128x128.
"""
import os
import time
from PIL import Image
from models.nano_pixel_art_v1.inference.nanopixel_engine import NanoPixelEngine

def verify_and_benchmark():
    engine = NanoPixelEngine()
    prompts = [
        "Mage",
        "Cyberpunk Rogue",
        "Retro Orc",
        "Mecha",
        "Space Marine"
    ]

    os.makedirs("models/nano_pixel_art_v1/inference/matrix", exist_ok=True)
    images_64 = []
    images_128 = []

    print("=== Benchmarking NanoPixel-v1 Inference ===")
    for prompt in prompts:
        t0 = time.time()
        img64 = engine.generate(prompt, size=64, seed=123)
        t64 = time.time() - t0

        t0 = time.time()
        img128 = engine.generate(prompt, size=128, seed=123)
        t128 = time.time() - t0

        img64.save(f"models/nano_pixel_art_v1/inference/matrix/{prompt.lower().replace(' ', '_')}_64.png")
        img128.save(f"models/nano_pixel_art_v1/inference/matrix/{prompt.lower().replace(' ', '_')}_128.png")

        images_64.append(img64)
        images_128.append(img128)
        print(f"[{prompt}] 64x64: {t64*1000:.1f}ms | 128x128: {t128*1000:.1f}ms")

    # Combine into matrix sheet
    sheet_64 = Image.new("RGBA", (64 * len(prompts), 64), (0, 0, 0, 0))
    for idx, img in enumerate(images_64):
        sheet_64.paste(img, (idx * 64, 0))
    sheet_64.save("models/nano_pixel_art_v1/inference/matrix/character_matrix_64.png")

    sheet_128 = Image.new("RGBA", (128 * len(prompts), 128), (0, 0, 0, 0))
    for idx, img in enumerate(images_128):
        sheet_128.paste(img, (idx * 128, 0))
    sheet_128.save("models/nano_pixel_art_v1/inference/matrix/character_matrix_128.png")

    print("\nCharacter matrices saved to models/nano_pixel_art_v1/inference/matrix/")

if __name__ == "__main__":
    verify_and_benchmark()
