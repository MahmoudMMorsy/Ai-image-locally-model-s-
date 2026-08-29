"""
Procedural Pixel Art Sprite Generator & Neural Latent Fine-Tuner
Generates structured 64x64 pixel art character bodies (heads, torsos, armor, limbs, outlines)
and pre-trains the neural weights for consistent character generation.
"""
import torch
import torch.optim as optim
import numpy as np
from PIL import Image, ImageDraw
import os
from .model import PixelSpriteGenerator, PixelSpriteEncoder
from .palette import quantize_to_pixel_art, SIGNATURE_PALETTE


def generate_procedural_sprite(archetype: str, color_name: str = "blue", seed: int = 42) -> Image.Image:
    """
    Procedurally synthesizes a structured 64x64 pixel art sprite character with
    distinct head, torso, armor, limbs, and outline.
    """
    np.random.seed(seed)
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Color definitions
    color_map = {
        "blue": (45, 115, 215, 255),
        "red": (225, 45, 65, 255),
        "green": (75, 190, 80, 255),
        "gold": (250, 200, 50, 255),
        "purple": (175, 70, 190, 255),
        "dark": (54, 45, 50, 255)
    }
    armor_color = color_map.get(color_name.lower(), (45, 115, 215, 255))
    skin_color = (250, 205, 165, 255)
    outline_color = (20, 16, 19, 255)

    # 1. Outline / Body Base (Centered 64x64)
    # Head
    draw.rectangle([24, 10, 39, 24], fill=skin_color, outline=outline_color)
    # Eyes
    draw.rectangle([27, 16, 29, 18], fill=outline_color)
    draw.rectangle([34, 16, 36, 18], fill=outline_color)

    # Helmet/Hat based on archetype
    if archetype in ["knight", "warrior", "hero"]:
        draw.rectangle([23, 8, 40, 14], fill=armor_color, outline=outline_color)
    elif archetype in ["wizard", "mage"]:
        draw.polygon([(32, 2), (20, 12), (43, 12)], fill=(175, 70, 190, 255), outline=outline_color)

    # Torso / Armor
    draw.rectangle([20, 25, 43, 44], fill=armor_color, outline=outline_color)
    # Chestplate highlight
    draw.rectangle([26, 28, 37, 36], fill=(240, 240, 245, 255))

    # Arms & Legs
    # Left Arm & Right Arm
    draw.rectangle([14, 26, 19, 40], fill=skin_color, outline=outline_color)
    draw.rectangle([44, 26, 49, 40], fill=skin_color, outline=outline_color)

    # Legs
    draw.rectangle([22, 45, 29, 58], fill=(100, 90, 95, 255), outline=outline_color)
    draw.rectangle([34, 45, 41, 58], fill=(100, 90, 95, 255), outline=outline_color)

    # Boots
    draw.rectangle([20, 54, 29, 60], fill=outline_color)
    draw.rectangle([34, 54, 43, 60], fill=outline_color)

    return quantize_to_pixel_art(img, size=(64, 64), palette=SIGNATURE_PALETTE)


def pretrain_engine_weights(generator: PixelSpriteGenerator, encoder: PixelSpriteEncoder, device="cpu", epochs=20):
    """
    Trains generator and encoder on procedurally synthesized character sprites
    so the neural weights produce structured pixel art characters.
    """
    optimizer_g = optim.Adam(generator.parameters(), lr=0.005)
    optimizer_e = optim.Adam(encoder.parameters(), lr=0.005)

    generator.train()
    encoder.train()

    archetypes = ["knight", "warrior", "wizard", "hero", "rogue"]
    colors = ["blue", "red", "green", "gold", "purple"]

    for epoch in range(epochs):
        for arch in archetypes:
            for col in colors:
                sprite = generate_procedural_sprite(arch, col, seed=epoch + 10)
                arr = np.array(sprite, dtype=np.float32) / 255.0
                target_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

                optimizer_g.zero_grad()
                optimizer_e.zero_grad()

                z_latent = encoder(target_tensor)
                cond = torch.zeros(1, 32, device=device)
                rec_tensor = generator(z_latent, cond)

                loss = torch.mean((rec_tensor - target_tensor) ** 2)
                loss.backward()

                optimizer_g.step()
                optimizer_e.step()

    generator.eval()
    encoder.eval()
