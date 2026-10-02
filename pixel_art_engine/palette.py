"""
Pixel Art Palette & Quantization Module
Ensures crisp, indexed color palette quantization for pixel art sprites and retro showcases.
"""
import numpy as np
from PIL import Image

# Game Boy 4-color palette (classic green shades)
GAMEBOY_PALETTE = np.array([
    [0, 0, 0, 0],         # 0: Transparent
    [15, 56, 15, 255],    # 1: Darkest Green
    [48, 98, 48, 255],    # 2: Dark Green
    [139, 172, 15, 255],  # 3: Light Green
    [155, 188, 15, 255],  # 4: Lightest Green
], dtype=np.uint8)

# NES 16-color retro palette
NES_PALETTE = np.array([
    [0, 0, 0, 0],         # 0: Transparent
    [0, 0, 0, 255],       # 1: Black
    [255, 255, 255, 255], # 2: White
    [124, 124, 124, 255], # 3: Medium Gray
    [188, 188, 188, 255], # 4: Light Gray
    [252, 0, 0, 255],     # 5: Red
    [248, 56, 0, 255],    # 6: Orange-Red
    [252, 160, 68, 255],  # 7: Peach / Orange
    [248, 184, 0, 255],   # 8: Yellow
    [0, 168, 0, 255],     # 9: Green
    [0, 120, 248, 255],   # 10: Bright Blue
    [0, 0, 188, 255],     # 11: Dark Blue
    [104, 68, 252, 255],  # 12: Purple
    [216, 0, 204, 255],   # 13: Magenta
    [112, 40, 0, 255],    # 14: Dark Brown
    [248, 120, 88, 255],  # 15: Light Salmon / Skin
], dtype=np.uint8)

# Signature pixel art palette (Retro 32-color palette + Alpha/Transparency)
SIGNATURE_PALETTE = np.array([
    [0, 0, 0, 0],         # 0: Transparent
    [20, 16, 19, 255],     # 1: Dark Black / Outline
    [54, 45, 50, 255],     # 2: Dark Charcoal
    [100, 90, 95, 255],    # 3: Medium Gray
    [180, 175, 182, 255],  # 4: Light Gray
    [240, 240, 245, 255],  # 5: White / Highlight
    [140, 25, 40, 255],    # 6: Dark Red
    [225, 45, 65, 255],    # 7: Bright Red
    [245, 125, 45, 255],   # 8: Orange
    [250, 200, 50, 255],   # 9: Yellow
    [40, 110, 55, 255],    # 10: Dark Green
    [75, 190, 80, 255],    # 11: Vibrant Green
    [160, 230, 90, 255],   # 12: Light Lime Green
    [25, 50, 120, 255],    # 13: Dark Blue
    [45, 115, 215, 255],   # 14: Royal Blue
    [95, 195, 245, 255],   # 15: Sky Blue / Cyan
    [90, 40, 110, 255],    # 16: Dark Purple
    [175, 70, 190, 255],   # 17: Bright Purple
    [235, 130, 205, 255],  # 18: Pink
    [105, 60, 40, 255],    # 19: Dark Brown
    [170, 105, 60, 255],   # 20: Medium Brown / Hair
    [230, 165, 110, 255],  # 21: Light Brown / Tan
    [250, 205, 165, 255],  # 22: Skin Light Tone
    [210, 150, 110, 255],  # 23: Skin Medium Shadow
    [155, 95, 65, 255],    # 24: Skin Dark Shadow
    [70, 140, 130, 255],   # 25: Teal / Armor
    [140, 210, 195, 255],  # 26: Light Teal
    [220, 235, 170, 255],  # 27: Pastel Yellow-Green
    [255, 235, 205, 255],  # 28: Cream Highlight
    [60, 80, 100, 255],    # 29: Steel Blue Dark
    [110, 140, 165, 255],  # 30: Steel Blue Light
    [200, 70, 120, 255],   # 31: Magenta
], dtype=np.uint8)


def quantize_palette(image: Image.Image, palette: np.ndarray = SIGNATURE_PALETTE) -> Image.Image:
    """
    Quantizes a PIL Image to a specific palette array.
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    arr = np.array(image, dtype=np.float32) # (H, W, 4)
    alpha = arr[:, :, 3]
    transparent_mask = alpha < 128

    rgb = arr[:, :, :3]
    pal_rgb = palette[:, :3].astype(np.float32)

    diff = rgb[:, :, np.newaxis, :] - pal_rgb[np.newaxis, np.newaxis, :, :]
    dist = np.sum(diff ** 2, axis=-1)

    # If palette has transparent index 0
    if palette.shape[0] > 1 and palette[0, 3] == 0:
        dist[:, :, 0] = 1e9

    nearest_idx = np.argmin(dist, axis=-1)
    quantized_arr = palette[nearest_idx].copy()

    if palette.shape[0] > 1 and palette[0, 3] == 0:
        quantized_arr[transparent_mask] = palette[0]

    return Image.fromarray(quantized_arr, mode="RGBA")

def quantize_to_pixel_art(image: Image.Image, size=(64, 64), palette=SIGNATURE_PALETTE) -> Image.Image:
    """
    Resizes and quantizes any PIL Image to a crisp pixel art sprite
    using nearest neighbor sampling and exact distance-based color palette mapping.
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    resized = image.resize(size, Image.Resampling.NEAREST)
    return quantize_palette(resized, palette)
