"""
Arcade & NES/GBA Pixel Art Character Template and Painter Engine.
Renders clean 64x64 retro arcade game characters with outlines, shading, weapons, and accessories.
"""
import numpy as np
from PIL import Image

# 16-color NES/GBA Arcade Palette
GBA_PALETTES = {
    "knight": {
        "outline": (15, 15, 25, 255),
        "armor_dark": (70, 80, 105, 255),
        "armor_mid": (120, 135, 170, 255),
        "armor_light": (190, 210, 240, 255),
        "skin_dark": (180, 100, 60, 255),
        "skin_mid": (230, 150, 100, 255),
        "skin_light": (255, 200, 150, 255),
        "cape_dark": (120, 10, 20, 255),
        "cape_mid": (190, 30, 45, 255),
        "cape_light": (240, 70, 80, 255),
        "gold_dark": (160, 110, 10, 255),
        "gold_mid": (220, 170, 30, 255),
        "gold_light": (255, 220, 90, 255),
        "steel_dark": (60, 60, 70, 255),
        "steel_light": (200, 210, 225, 255),
        "eye": (255, 255, 255, 255),
    },
    "wizard": {
        "outline": (15, 10, 30, 255),
        "robe_dark": (40, 20, 90, 255),
        "robe_mid": (80, 50, 160, 255),
        "robe_light": (140, 100, 230, 255),
        "trim_dark": (160, 110, 10, 255),
        "trim_mid": (230, 180, 40, 255),
        "trim_light": (255, 230, 110, 255),
        "skin_dark": (180, 100, 60, 255),
        "skin_mid": (230, 150, 100, 255),
        "skin_light": (255, 200, 150, 255),
        "beard_dark": (140, 140, 150, 255),
        "beard_light": (230, 230, 240, 255),
        "magic_dark": (10, 160, 200, 255),
        "magic_light": (120, 230, 255, 255),
        "staff_wood": (100, 60, 30, 255),
        "eye": (100, 230, 255, 255),
    },
    "monster": {
        "outline": (10, 25, 10, 255),
        "skin_dark": (20, 80, 30, 255),
        "skin_mid": (45, 145, 55, 255),
        "skin_light": (95, 205, 80, 255),
        "belly_dark": (140, 130, 70, 255),
        "belly_light": (210, 200, 130, 255),
        "horn_dark": (80, 60, 40, 255),
        "horn_light": (190, 170, 130, 255),
        "eye_glow": (255, 40, 20, 255),
        "teeth": (240, 240, 220, 255),
        "claw": (50, 50, 50, 255),
        "blood": (160, 10, 20, 255),
    },
    "robot": {
        "outline": (20, 20, 25, 255),
        "metal_dark": (50, 60, 75, 255),
        "metal_mid": (100, 115, 135, 255),
        "metal_light": (170, 185, 205, 255),
        "joint": (30, 30, 35, 255),
        "visor_dark": (180, 20, 20, 255),
        "visor_light": (255, 80, 80, 255),
        "energy_core": (30, 200, 255, 255),
        "gold_trim": (220, 170, 30, 255),
    }
}

def draw_rect(arr, x1, y1, x2, y2, color):
    """Draws a filled rectangle in uint8 RGBA array (64, 64, 4)."""
    x1, x2 = max(0, min(63, x1)), max(0, min(63, x2))
    y1, y2 = max(0, min(63, y1)), max(0, min(63, y2))
    for y in range(y1, y2 + 1):
        for x in range(x1, x2 + 1):
            arr[y, x] = color

def draw_outline(arr, color):
    """Adds a 1-pixel dark outline around all non-transparent pixels."""
    h, w, c = arr.shape
    mask = arr[:, :, 3] > 0
    outline_mask = np.zeros((h, w), dtype=bool)

    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                # check 4 neighbors
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
                        outline_mask[y, x] = True
                        break

    arr[outline_mask] = color

def generate_arcade_sprite(archetype: str = "knight", color_theme: str = "blue", pose: str = "idle", frame: int = 0) -> Image.Image:
    """
    Generates a high quality 64x64 NES/GBA/Arcade style pixel art character.
    """
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    arch = archetype.lower()

    # Select base palette
    if "wiz" in arch or "mage" in arch:
        pal = GBA_PALETTES["wizard"]
        arch_type = "wizard"
    elif "monst" in arch or "orc" in arch or "gob" in arch or "beast" in arch:
        pal = GBA_PALETTES["monster"]
        arch_type = "monster"
    elif "rob" in arch or "cyb" in arch or "mech" in arch:
        pal = GBA_PALETTES["robot"]
        arch_type = "robot"
    else:
        pal = GBA_PALETTES["knight"]
        arch_type = "knight"

    # Adjust color theme if specified
    if "red" in color_theme:
        if arch_type == "knight":
            pal["armor_dark"] = (100, 20, 20, 255)
            pal["armor_mid"] = (180, 40, 40, 255)
            pal["armor_light"] = (240, 100, 100, 255)
    elif "green" in color_theme:
        if arch_type == "knight":
            pal["armor_dark"] = (20, 80, 30, 255)
            pal["armor_mid"] = (40, 150, 60, 255)
            pal["armor_light"] = (90, 220, 110, 255)
    elif "gold" in color_theme or "yellow" in color_theme:
        if arch_type == "knight":
            pal["armor_dark"] = (140, 90, 10, 255)
            pal["armor_mid"] = (210, 150, 20, 255)
            pal["armor_light"] = (255, 210, 70, 255)

    # Pose offsets for animation
    y_bounce = 0
    leg_shift = 0
    arm_shift = 0
    if pose == "walk" or pose == "run":
        y_bounce = (frame % 2)
        leg_shift = 2 if (frame % 2 == 1) else -2
        arm_shift = -3 if (frame % 2 == 1) else 3
    elif pose == "attack":
        arm_shift = 6 if frame > 1 else 3
    elif pose == "jump":
        y_bounce = -4

    # --- Draw Character Components ---
    if arch_type == "knight":
        # Cape
        draw_rect(arr, 20, 22 + y_bounce, 43, 52 + y_bounce, pal["cape_dark"])
        draw_rect(arr, 23, 24 + y_bounce, 40, 50 + y_bounce, pal["cape_mid"])

        # Legs / Boots
        draw_rect(arr, 24 - leg_shift, 44 + y_bounce, 29 - leg_shift, 56 + y_bounce, pal["armor_dark"])
        draw_rect(arr, 34 + leg_shift, 44 + y_bounce, 39 + leg_shift, 56 + y_bounce, pal["armor_dark"])
        draw_rect(arr, 23 - leg_shift, 52 + y_bounce, 29 - leg_shift, 57 + y_bounce, pal["steel_dark"])
        draw_rect(arr, 34 + leg_shift, 52 + y_bounce, 40 + leg_shift, 57 + y_bounce, pal["steel_dark"])

        # Torso / Breastplate
        draw_rect(arr, 22, 24 + y_bounce, 41, 44 + y_bounce, pal["armor_dark"])
        draw_rect(arr, 24, 25 + y_bounce, 39, 42 + y_bounce, pal["armor_mid"])
        draw_rect(arr, 26, 26 + y_bounce, 33, 38 + y_bounce, pal["armor_light"]) # highlight
        draw_rect(arr, 28, 41 + y_bounce, 35, 43 + y_bounce, pal["gold_mid"]) # belt

        # Head / Helmet
        draw_rect(arr, 22, 8 + y_bounce, 41, 24 + y_bounce, pal["armor_dark"])
        draw_rect(arr, 24, 9 + y_bounce, 39, 23 + y_bounce, pal["armor_mid"])
        draw_rect(arr, 25, 10 + y_bounce, 31, 20 + y_bounce, pal["armor_light"])
        # Visor slit
        draw_rect(arr, 26, 16 + y_bounce, 38, 18 + y_bounce, (10, 10, 15, 255))
        draw_rect(arr, 30, 16 + y_bounce, 32, 18 + y_bounce, pal["eye"]) # glowing eye
        # Plume on top
        draw_rect(arr, 30, 2 + y_bounce, 33, 9 + y_bounce, pal["cape_mid"])
        draw_rect(arr, 28, 4 + y_bounce, 35, 8 + y_bounce, pal["cape_light"])

        # Shield (Left Arm)
        draw_rect(arr, 15 - arm_shift, 25 + y_bounce, 23 - arm_shift, 42 + y_bounce, pal["gold_dark"])
        draw_rect(arr, 17 - arm_shift, 27 + y_bounce, 21 - arm_shift, 40 + y_bounce, pal["gold_mid"])
        draw_rect(arr, 18 - arm_shift, 29 + y_bounce, 20 - arm_shift, 38 + y_bounce, pal["cape_mid"])

        # Sword (Right Arm)
        if pose == "attack" and frame >= 1:
            # Thrusting Sword forward
            draw_rect(arr, 40, 26 + y_bounce, 58, 29 + y_bounce, pal["steel_light"])
            draw_rect(arr, 38, 24 + y_bounce, 40, 31 + y_bounce, pal["gold_mid"])
        else:
            # Holding sword vertical
            draw_rect(arr, 41 + arm_shift, 28 + y_bounce, 45 + arm_shift, 34 + y_bounce, pal["skin_mid"])
            draw_rect(arr, 42 + arm_shift, 12 + y_bounce, 44 + arm_shift, 28 + y_bounce, pal["steel_light"])
            draw_rect(arr, 40 + arm_shift, 28 + y_bounce, 46 + arm_shift, 30 + y_bounce, pal["gold_mid"])

    elif arch_type == "wizard":
        # Robe / Body
        draw_rect(arr, 20, 20 + y_bounce, 43, 56 + y_bounce, pal["robe_dark"])
        draw_rect(arr, 23, 22 + y_bounce, 40, 54 + y_bounce, pal["robe_mid"])
        draw_rect(arr, 25, 24 + y_bounce, 32, 52 + y_bounce, pal["robe_light"])
        draw_rect(arr, 30, 20 + y_bounce, 33, 56 + y_bounce, pal["trim_mid"]) # trim

        # Head / Beard / Hat
        draw_rect(arr, 25, 14 + y_bounce, 38, 22 + y_bounce, pal["skin_mid"])
        draw_rect(arr, 23, 20 + y_bounce, 40, 34 + y_bounce, pal["beard_light"]) # long beard
        draw_rect(arr, 26, 22 + y_bounce, 37, 30 + y_bounce, pal["beard_dark"])

        # Pointy Wizard Hat
        draw_rect(arr, 18, 13 + y_bounce, 45, 16 + y_bounce, pal["trim_dark"]) # brim
        draw_rect(arr, 22, 8 + y_bounce, 41, 13 + y_bounce, pal["robe_dark"])
        draw_rect(arr, 26, 2 + y_bounce, 37, 8 + y_bounce, pal["robe_mid"])
        draw_rect(arr, 29, 0 + y_bounce, 34, 3 + y_bounce, pal["trim_light"])

        # Staff with Glowing Orb
        draw_rect(arr, 44 + arm_shift, 14 + y_bounce, 46 + arm_shift, 56 + y_bounce, pal["staff_wood"])
        draw_rect(arr, 42 + arm_shift, 8 + y_bounce, 48 + arm_shift, 14 + y_bounce, pal["magic_dark"])
        draw_rect(arr, 43 + arm_shift, 9 + y_bounce, 47 + arm_shift, 13 + y_bounce, pal["magic_light"])

    elif arch_type == "monster":
        # Big bulky body
        draw_rect(arr, 16, 20 + y_bounce, 47, 52 + y_bounce, pal["skin_dark"])
        draw_rect(arr, 19, 22 + y_bounce, 44, 49 + y_bounce, pal["skin_mid"])
        draw_rect(arr, 24, 26 + y_bounce, 39, 46 + y_bounce, pal["belly_light"])

        # Legs / Claws
        draw_rect(arr, 18 - leg_shift, 48 + y_bounce, 26 - leg_shift, 58 + y_bounce, pal["skin_dark"])
        draw_rect(arr, 37 + leg_shift, 48 + y_bounce, 45 + leg_shift, 58 + y_bounce, pal["skin_dark"])

        # Head / Horns / Red glowing eyes
        draw_rect(arr, 18, 8 + y_bounce, 45, 22 + y_bounce, pal["skin_mid"])
        draw_rect(arr, 22, 14 + y_bounce, 26, 17 + y_bounce, pal["eye_glow"])
        draw_rect(arr, 37, 14 + y_bounce, 41, 17 + y_bounce, pal["eye_glow"])
        # Horns
        draw_rect(arr, 14, 2 + y_bounce, 20, 10 + y_bounce, pal["horn_light"])
        draw_rect(arr, 43, 2 + y_bounce, 49, 10 + y_bounce, pal["horn_light"])
        # Mouth / Fangs
        draw_rect(arr, 24, 19 + y_bounce, 39, 23 + y_bounce, (10, 0, 0, 255))
        draw_rect(arr, 26, 19 + y_bounce, 28, 22 + y_bounce, pal["teeth"])
        draw_rect(arr, 35, 19 + y_bounce, 37, 22 + y_bounce, pal["teeth"])

    elif arch_type == "robot":
        # Metallic torso / joints
        draw_rect(arr, 20, 22 + y_bounce, 43, 46 + y_bounce, pal["metal_dark"])
        draw_rect(arr, 23, 24 + y_bounce, 40, 44 + y_bounce, pal["metal_mid"])
        # Glowing Energy Core
        draw_rect(arr, 28, 28 + y_bounce, 35, 35 + y_bounce, pal["energy_core"])

        # Metallic Legs
        draw_rect(arr, 22 - leg_shift, 46 + y_bounce, 28 - leg_shift, 58 + y_bounce, pal["metal_dark"])
        draw_rect(arr, 35 + leg_shift, 46 + y_bounce, 41 + leg_shift, 58 + y_bounce, pal["metal_dark"])

        # Square Metallic Head with Visor
        draw_rect(arr, 21, 8 + y_bounce, 42, 22 + y_bounce, pal["metal_mid"])
        draw_rect(arr, 23, 10 + y_bounce, 40, 20 + y_bounce, pal["metal_light"])
        draw_rect(arr, 24, 13 + y_bounce, 39, 17 + y_bounce, pal["visor_dark"])
        draw_rect(arr, 28, 14 + y_bounce, 35, 16 + y_bounce, pal["visor_light"])

    # Apply 1px dark arcade outline
    draw_outline(arr, pal["outline"])

    return Image.fromarray(arr, mode="RGBA")
