"""
nano_pixel_3A_xl Dataset Preprocessing Routine
Processes raw sprite images into 64x64 and 128x128 multi-scale character samples.
"""
import os
from PIL import Image
import numpy as np

RAW_DIR = "models/nano_pixel_3A_xl/dataset/raw_sprites"
PROC_64_DIR = "models/nano_pixel_3A_xl/dataset/processed_64x64"
PROC_128_DIR = "models/nano_pixel_3A_xl/dataset/processed_128x128"

class NanoPixel3AXLDatasetProcessor:
    def __init__(self, raw_dir=RAW_DIR, out_64=PROC_64_DIR, out_128=PROC_128_DIR):
        self.raw_dir = raw_dir
        self.out_64 = out_64
        self.out_128 = out_128

    def process(self):
        os.makedirs(self.out_64, exist_ok=True)
        os.makedirs(self.out_128, exist_ok=True)
        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir, exist_ok=True)

        files = [f for f in os.listdir(self.raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        print(f"[nano_pixel_3A_xl] Processing {len(files)} raw sprite images from {self.raw_dir}...")

        exported = 0
        for f in files:
            file_path = os.path.join(self.raw_dir, f)
            try:
                im = Image.open(file_path).convert("RGBA")
                crop_64 = im.resize((64, 64), Image.Resampling.NEAREST)
                crop_128 = im.resize((128, 128), Image.Resampling.NEAREST)

                base = os.path.splitext(f)[0]
                crop_64.save(os.path.join(self.out_64, f"{base}.png"))
                crop_128.save(os.path.join(self.out_128, f"{base}.png"))
                exported += 1
            except Exception as e:
                print(f"Error in 3A_xl processor: {e}")

        print(f"[nano_pixel_3A_xl] Exported {exported} multi-scale character images.")

if __name__ == "__main__":
    processor = NanoPixel3AXLDatasetProcessor()
    processor.process()
