import os

class NanoPixel3AXLDatasetProcessor:
    def __init__(self, raw_dir="models/nano_pixel_3A_xl/dataset/raw_sprites"):
        self.raw_dir = raw_dir

    def process(self):
        print(f"Processing sprites in: {self.raw_dir}")
