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


def generate_procedural_sprite(archetype: str = "knight", color_name: str = "gold", seed: int = 42) -> Image.Image:
    """
    Synthesizes detailed 16-bit RPG Pixel Art Sprites with layered armor, capes,
    weapons, pauldrons, specular gems, hair, and expressive eyes.
    """
    np.random.seed(seed)
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    palettes = {
        "red": {
            "armor_hi": (255, 120, 130, 255), "armor_mid": (225, 45, 65, 255),
            "armor_dark": (140, 25, 40, 255), "accent": (250, 200, 50, 255)
        },
        "blue": {
            "armor_hi": (130, 185, 255, 255), "armor_mid": (45, 115, 215, 255),
            "armor_dark": (25, 50, 120, 255), "accent": (240, 240, 245, 255)
        },
        "green": {
            "armor_hi": (160, 230, 90, 255), "armor_mid": (75, 190, 80, 255),
            "armor_dark": (40, 110, 55, 255), "accent": (250, 200, 50, 255)
        },
        "gold": {
            "armor_hi": (255, 235, 120, 255), "armor_mid": (250, 200, 50, 255),
            "armor_dark": (170, 120, 20, 255), "accent": (225, 45, 65, 255)
        },
        "purple": {
            "armor_hi": (235, 130, 205, 255), "armor_mid": (175, 70, 190, 255),
            "armor_dark": (90, 40, 110, 255), "accent": (95, 195, 245, 255)
        }
    }

    pal = palettes.get(color_name.lower(), palettes["gold"])
    skin_hi = (255, 225, 195, 255)
    skin_mid = (250, 205, 165, 255)
    skin_dark = (210, 150, 110, 255)
    outline = (20, 16, 19, 255)
    hair_mid = (170, 105, 60, 255) if archetype != "wizard" else (220, 220, 230, 255)
    hair_hi = (230, 165, 110, 255) if archetype != "wizard" else (255, 255, 255, 255)
    hair_dark = (105, 60, 40, 255) if archetype != "wizard" else (160, 160, 180, 255)

    # 1. Cape
    if archetype in ["knight", "wizard", "hero"]:
        draw.polygon([(20, 26), (44, 26), (48, 58), (16, 58)], fill=pal["armor_dark"])
        draw.polygon([(22, 28), (42, 28), (45, 56), (19, 56)], fill=pal["armor_mid"])

    # 2. Legs & Boots
    draw.rectangle([25, 42, 31, 54], fill=(60, 80, 100, 255))
    draw.rectangle([33, 42, 39, 54], fill=(60, 80, 100, 255))
    draw.polygon([(24, 52), (32, 52), (32, 60), (22, 60)], fill=pal["armor_dark"], outline=outline)
    draw.polygon([(32, 52), (40, 52), (42, 60), (32, 60)], fill=pal["armor_dark"], outline=outline)
    draw.rectangle([24, 53, 30, 55], fill=pal["armor_mid"])
    draw.rectangle([34, 53, 40, 55], fill=pal["armor_mid"])

    # 3. Torso & Armor
    draw.polygon([(23, 24), (41, 24), (39, 44), (25, 44)], fill=pal["armor_mid"], outline=outline)
    draw.polygon([(24, 25), (31, 25), (30, 43), (25, 43)], fill=pal["armor_dark"])
    draw.polygon([(32, 25), (40, 25), (38, 43), (31, 43)], fill=pal["armor_hi"])
    draw.rectangle([29, 28, 35, 36], fill=pal["accent"], outline=outline)
    draw.rectangle([31, 30, 33, 34], fill=(255, 255, 255, 255))
    draw.rectangle([24, 40, 40, 44], fill=(105, 60, 40, 255), outline=outline)
    draw.rectangle([30, 39, 34, 45], fill=pal["accent"], outline=outline)

    # Shoulder Pads
    draw.polygon([(17, 23), (25, 22), (25, 30), (18, 31)], fill=pal["armor_mid"], outline=outline)
    draw.polygon([(39, 22), (47, 23), (46, 31), (39, 30)], fill=pal["armor_hi"], outline=outline)

    # 4. Arms & Weapon
    draw.rectangle([17, 30, 22, 42], fill=pal["armor_mid"], outline=outline)
    draw.rectangle([17, 39, 22, 43], fill=skin_mid, outline=outline)
    draw.rectangle([42, 30, 47, 42], fill=pal["armor_hi"], outline=outline)
    draw.rectangle([42, 39, 47, 43], fill=skin_mid, outline=outline)

    if archetype in ["knight", "warrior", "hero"]:
        draw.rectangle([48, 12, 51, 48], fill=(220, 230, 240, 255), outline=outline)
        draw.rectangle([49, 14, 50, 46], fill=(255, 255, 255, 255))
        draw.rectangle([45, 36, 54, 38], fill=pal["accent"], outline=outline)
        draw.rectangle([48, 39, 51, 44], fill=(105, 60, 40, 255), outline=outline)
    elif archetype in ["wizard", "mage"]:
        draw.rectangle([48, 6, 51, 56], fill=(105, 60, 40, 255), outline=outline)
        draw.ellipse([46, 2, 53, 9], fill=pal["accent"], outline=outline)
        draw.ellipse([48, 4, 51, 7], fill=(255, 255, 255, 255))

    # 5. Head & Face
    draw.rectangle([25, 11, 39, 23], fill=skin_mid, outline=outline)
    draw.rectangle([26, 12, 32, 22], fill=skin_dark)
    draw.rectangle([32, 12, 38, 22], fill=skin_hi)
    draw.rectangle([28, 16, 30, 19], fill=(255, 255, 255, 255), outline=outline)
    draw.rectangle([34, 16, 36, 19], fill=(255, 255, 255, 255), outline=outline)
    draw.rectangle([29, 17, 30, 19], fill=outline)
    draw.rectangle([35, 17, 36, 19], fill=outline)

    if archetype in ["knight", "warrior"]:
        draw.polygon([(24, 7), (40, 7), (41, 15), (23, 15)], fill=pal["armor_mid"], outline=outline)
        draw.rectangle([26, 14, 38, 17], fill=outline)
        draw.polygon([(30, 2), (34, 2), (36, 7), (28, 7)], fill=pal["accent"], outline=outline)
    elif archetype in ["wizard", "mage"]:
        draw.polygon([(32, 0), (20, 11), (44, 11)], fill=pal["armor_mid"], outline=outline)
        draw.rectangle([20, 10, 44, 13], fill=pal["accent"], outline=outline)
    else:
        draw.polygon([(23, 7), (41, 7), (41, 14), (23, 14)], fill=hair_mid, outline=outline)
        draw.rectangle([27, 8, 37, 11], fill=hair_hi)
        draw.polygon([(23, 11), (27, 17), (26, 11)], fill=hair_dark)
        draw.polygon([(37, 11), (41, 17), (40, 11)], fill=hair_dark)

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
