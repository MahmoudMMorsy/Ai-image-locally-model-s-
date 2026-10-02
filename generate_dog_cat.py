import os
import imageio
import numpy as np
from PIL import Image, ImageDraw

def draw_rect(draw, bbox, fill=None, outline=None):
    """Utility helper to draw rectangle on PIL ImageDraw."""
    draw.rectangle(bbox, fill=fill, outline=outline)

# -------------------- CAT CHARACTERS -------------------- #

def create_ginger_cat(frame_idx=0):
    """Cat 1: Orange/Ginger Tabby Cat."""
    img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    orange = (235, 130, 40, 255)
    dark_orange = (190, 90, 20, 255)
    white = (245, 245, 245, 255)
    pink = (240, 140, 160, 255)
    green = (40, 200, 80, 255)
    black = (20, 20, 25, 255)

    offset = (frame_idx % 2) * 2
    tail_y = 44 + ((frame_idx % 4) - 2) * 2

    # Tail
    draw.line([(42, 46), (52, tail_y), (56, tail_y - 4)], fill=orange, width=3)

    # Body & Belly
    draw_rect(draw, [22, 28, 42, 52], fill=orange, outline=dark_orange)
    draw_rect(draw, [27, 34, 37, 50], fill=white)

    # Head
    draw_rect(draw, [20, 16, 44, 32], fill=orange, outline=dark_orange)

    # Ears
    draw.polygon([(20, 16), (25, 6), (29, 16)], fill=orange, outline=dark_orange)
    draw.polygon([(22, 16), (25, 9), (27, 16)], fill=pink)
    draw.polygon([(35, 16), (39, 6), (44, 16)], fill=orange, outline=dark_orange)
    draw.polygon([(37, 16), (39, 9), (42, 16)], fill=pink)

    # Eyes & Pupils
    draw_rect(draw, [25, 21, 28, 25], fill=green)
    draw_rect(draw, [26, 22, 27, 24], fill=black)
    draw_rect(draw, [36, 21, 39, 25], fill=green)
    draw_rect(draw, [37, 22, 38, 24], fill=black)

    # Nose & Mouth
    draw_rect(draw, [31, 26, 33, 27], fill=pink)
    draw.line([(32, 27), (32, 29)], fill=dark_orange, width=1)

    # Whiskers
    draw.line([(14, 25), (20, 26)], fill=white, width=1)
    draw.line([(14, 28), (20, 28)], fill=white, width=1)
    draw.line([(44, 26), (50, 25)], fill=white, width=1)
    draw.line([(44, 28), (50, 28)], fill=white, width=1)

    # Paws
    draw_rect(draw, [22 + offset, 52, 27 + offset, 58], fill=white, outline=dark_orange)
    draw_rect(draw, [37 - offset, 52, 42 - offset, 58], fill=white, outline=dark_orange)

    return img

def create_black_cat(frame_idx=0):
    """Cat 2: Midnight Black Cat with Glowing Yellow Eyes & Red Collar."""
    img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    dark_grey = (35, 35, 45, 255)
    deep_black = (15, 15, 22, 255)
    yellow = (255, 220, 40, 255)
    black = (10, 10, 10, 255)
    red_collar = (220, 30, 40, 255)
    gold_bell = (255, 215, 0, 255)
    pink = (230, 130, 150, 255)

    offset = (frame_idx % 2) * 2
    tail_y = 42 + ((frame_idx % 4) - 2) * 3

    # Tail
    draw.line([(42, 45), (54, tail_y), (58, tail_y - 6)], fill=dark_grey, width=3)

    # Body
    draw_rect(draw, [22, 28, 42, 52], fill=dark_grey, outline=deep_black)

    # Red Collar with Gold Bell
    draw_rect(draw, [21, 30, 43, 33], fill=red_collar)
    draw_rect(draw, [30, 32, 34, 35], fill=gold_bell)

    # Head
    draw_rect(draw, [20, 16, 44, 31], fill=dark_grey, outline=deep_black)

    # Ears
    draw.polygon([(20, 16), (24, 5), (29, 16)], fill=dark_grey, outline=deep_black)
    draw.polygon([(22, 16), (24, 8), (27, 16)], fill=pink)
    draw.polygon([(35, 16), (40, 5), (44, 16)], fill=dark_grey, outline=deep_black)
    draw.polygon([(37, 16), (40, 8), (42, 16)], fill=pink)

    # Glowing Yellow Eyes
    draw_rect(draw, [25, 21, 29, 25], fill=yellow)
    draw_rect(draw, [27, 21, 28, 25], fill=black) # Slit pupil
    draw_rect(draw, [35, 21, 39, 25], fill=yellow)
    draw_rect(draw, [37, 21, 38, 25], fill=black)

    # Nose
    draw_rect(draw, [31, 26, 33, 27], fill=pink)

    # Whiskers
    draw.line([(14, 25), (20, 26)], fill=(200, 200, 210, 255), width=1)
    draw.line([(44, 26), (50, 25)], fill=(200, 200, 210, 255), width=1)

    # Paws
    draw_rect(draw, [22 + offset, 52, 27 + offset, 58], fill=deep_black, outline=dark_grey)
    draw_rect(draw, [37 - offset, 52, 42 - offset, 58], fill=deep_black, outline=dark_grey)

    return img

def create_siamese_cat(frame_idx=0):
    """Cat 3: Elegant Siamese Cat with Cream Body & Dark Points."""
    img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cream = (245, 235, 215, 255)
    dark_brown = (70, 45, 35, 255)
    deep_brown = (40, 25, 20, 255)
    blue_eye = (40, 160, 240, 255)
    black = (10, 10, 15, 255)
    pink = (240, 150, 160, 255)

    offset = (frame_idx % 2) * 2
    tail_y = 42 + ((frame_idx % 4) - 2) * 2

    # Tail (Dark Brown)
    draw.line([(42, 46), (53, tail_y), (57, tail_y - 5)], fill=dark_brown, width=3)

    # Body (Cream)
    draw_rect(draw, [22, 28, 42, 52], fill=cream, outline=dark_brown)

    # Head (Cream base, Dark Brown mask)
    draw_rect(draw, [20, 16, 44, 32], fill=cream, outline=dark_brown)
    draw_rect(draw, [26, 21, 38, 31], fill=dark_brown) # Dark face mask

    # Ears (Dark)
    draw.polygon([(20, 16), (24, 6), (29, 16)], fill=dark_brown, outline=deep_brown)
    draw.polygon([(35, 16), (40, 6), (44, 16)], fill=dark_brown, outline=deep_brown)

    # Bright Sapphire Blue Eyes
    draw_rect(draw, [25, 21, 28, 25], fill=blue_eye)
    draw_rect(draw, [26, 22, 27, 24], fill=black)
    draw_rect(draw, [36, 21, 39, 25], fill=blue_eye)
    draw_rect(draw, [37, 22, 38, 24], fill=black)

    # Nose
    draw_rect(draw, [31, 26, 33, 27], fill=pink)

    # Paws (Dark points)
    draw_rect(draw, [22 + offset, 52, 27 + offset, 58], fill=dark_brown, outline=deep_brown)
    draw_rect(draw, [37 - offset, 52, 42 - offset, 58], fill=dark_brown, outline=deep_brown)

    return img

def create_calico_cat(frame_idx=0):
    """Cat 4: Tri-color Calico Cat (White, Orange, & Black patches)."""
    img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    white = (250, 250, 250, 255)
    orange = (230, 120, 35, 255)
    black_patch = (40, 40, 45, 255)
    outline = (100, 90, 80, 255)
    green_eye = (60, 210, 110, 255)
    pink = (245, 150, 170, 255)

    offset = (frame_idx % 2) * 2
    tail_y = 44 + ((frame_idx % 4) - 2) * 2

    # Tail (Calico pattern)
    draw.line([(42, 46), (48, tail_y)], fill=orange, width=3)
    draw.line([(48, tail_y), (56, tail_y - 4)], fill=black_patch, width=3)

    # Body (White base with Orange & Black patches)
    draw_rect(draw, [22, 28, 42, 52], fill=white, outline=outline)
    draw_rect(draw, [22, 28, 30, 40], fill=orange)
    draw_rect(draw, [34, 38, 42, 50], fill=black_patch)

    # Head
    draw_rect(draw, [20, 16, 44, 32], fill=white, outline=outline)
    draw_rect(draw, [20, 16, 29, 24], fill=orange) # Left head patch
    draw_rect(draw, [35, 16, 44, 24], fill=black_patch) # Right head patch

    # Ears
    draw.polygon([(20, 16), (24, 6), (28, 16)], fill=orange, outline=outline)
    draw.polygon([(36, 16), (40, 6), (44, 16)], fill=black_patch, outline=outline)

    # Green Eyes
    draw_rect(draw, [25, 21, 28, 25], fill=green_eye)
    draw_rect(draw, [26, 22, 27, 24], fill=(10, 10, 10, 255))
    draw_rect(draw, [36, 21, 39, 25], fill=green_eye)
    draw_rect(draw, [37, 22, 38, 24], fill=(10, 10, 10, 255))

    # Nose
    draw_rect(draw, [31, 26, 33, 27], fill=pink)

    # Paws
    draw_rect(draw, [22 + offset, 52, 27 + offset, 58], fill=white, outline=outline)
    draw_rect(draw, [37 - offset, 52, 42 - offset, 58], fill=orange, outline=outline)

    return img


# -------------------- DOG CHARACTERS -------------------- #

def create_shiba_dog(frame_idx=0):
    """Dog 1: Golden/Brown Shiba Inu / Husky Mix."""
    img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    brown = (180, 110, 50, 255)
    dark_brown = (120, 65, 20, 255)
    cream = (245, 225, 190, 255)
    black = (25, 25, 30, 255)
    red_collar = (220, 40, 40, 255)
    tongue_pink = (245, 110, 140, 255)

    offset = (frame_idx % 2) * 2
    wag = ((frame_idx % 4) - 2) * 3

    # Tail (wagging)
    draw.line([(44, 38), (54 + wag, 30)], fill=brown, width=4)

    # Body & Chest
    draw_rect(draw, [20, 26, 44, 50], fill=brown, outline=dark_brown)
    draw_rect(draw, [26, 32, 38, 48], fill=cream)
    draw_rect(draw, [20, 26, 44, 29], fill=red_collar)

    # Head
    draw_rect(draw, [18, 12, 46, 28], fill=brown, outline=dark_brown)
    draw_rect(draw, [26, 20, 38, 28], fill=cream, outline=dark_brown) # Snout
    draw_rect(draw, [30, 20, 34, 23], fill=black) # Nose

    # Ears
    draw_rect(draw, [14, 12, 20, 24], fill=dark_brown)
    draw_rect(draw, [44, 12, 50, 24], fill=dark_brown)

    # Eyes
    draw_rect(draw, [23, 16, 27, 20], fill=black)
    draw_rect(draw, [24, 17, 25, 18], fill=(255, 255, 255, 255))
    draw_rect(draw, [37, 16, 41, 20], fill=black)
    draw_rect(draw, [38, 17, 39, 18], fill=(255, 255, 255, 255))

    # Tongue on odd frames
    if frame_idx % 2 == 1:
        draw_rect(draw, [30, 25, 34, 29], fill=tongue_pink)

    # Paws
    draw_rect(draw, [20 + offset, 50, 26 + offset, 58], fill=cream, outline=dark_brown)
    draw_rect(draw, [38 - offset, 50, 44 - offset, 58], fill=cream, outline=dark_brown)

    return img

def create_husky_dog(frame_idx=0):
    """Dog 2: Arctic Siberian Husky with Blue Eyes & Grey Coat."""
    img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    grey = (110, 120, 135, 255)
    dark_grey = (60, 70, 85, 255)
    white = (245, 245, 250, 255)
    ice_blue = (60, 180, 255, 255)
    black = (20, 20, 25, 255)
    blue_collar = (30, 80, 200, 255)

    offset = (frame_idx % 2) * 2
    wag = ((frame_idx % 4) - 2) * 3

    # Fluffy Tail
    draw.line([(44, 38), (56 + wag, 28)], fill=grey, width=5)
    draw.line([(48, 34), (58 + wag, 26)], fill=white, width=3)

    # Body
    draw_rect(draw, [20, 26, 44, 50], fill=grey, outline=dark_grey)
    draw_rect(draw, [25, 30, 39, 48], fill=white)
    draw_rect(draw, [20, 26, 44, 29], fill=blue_collar)

    # Head (Grey top, white face)
    draw_rect(draw, [18, 12, 46, 28], fill=grey, outline=dark_grey)
    draw_rect(draw, [22, 18, 42, 28], fill=white) # White face mask
    draw_rect(draw, [27, 12, 37, 18], fill=dark_grey) # Distinctive husky forehead marking

    # Perky Pointed Ears
    draw.polygon([(16, 12), (21, 3), (25, 12)], fill=dark_grey)
    draw.polygon([(39, 12), (43, 3), (48, 12)], fill=dark_grey)

    # Ice Blue Eyes
    draw_rect(draw, [23, 16, 27, 20], fill=ice_blue)
    draw_rect(draw, [25, 17, 26, 19], fill=black)
    draw_rect(draw, [37, 16, 41, 20], fill=ice_blue)
    draw_rect(draw, [38, 17, 39, 19], fill=black)

    # Black Nose
    draw_rect(draw, [30, 21, 34, 24], fill=black)

    # Paws
    draw_rect(draw, [20 + offset, 50, 26 + offset, 58], fill=white, outline=dark_grey)
    draw_rect(draw, [38 - offset, 50, 44 - offset, 58], fill=white, outline=dark_grey)

    return img

def create_dalmatian_dog(frame_idx=0):
    """Dog 3: Classic White Dalmatian with Black Spots & Green Collar."""
    img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    white = (250, 250, 250, 255)
    black = (20, 20, 25, 255)
    outline = (90, 90, 95, 255)
    green_collar = (30, 180, 70, 255)
    pink = (240, 140, 160, 255)

    offset = (frame_idx % 2) * 2
    wag = ((frame_idx % 4) - 2) * 3

    # Tail
    draw.line([(44, 38), (54 + wag, 32)], fill=white, width=3)
    draw_rect(draw, [48 + wag//2, 34, 50 + wag//2, 36], fill=black)

    # Body
    draw_rect(draw, [20, 26, 44, 50], fill=white, outline=outline)
    # Spots on body
    draw_rect(draw, [24, 30, 27, 33], fill=black)
    draw_rect(draw, [35, 34, 39, 37], fill=black)
    draw_rect(draw, [28, 42, 31, 45], fill=black)

    # Green Collar
    draw_rect(draw, [20, 26, 44, 29], fill=green_collar)

    # Head
    draw_rect(draw, [18, 12, 46, 28], fill=white, outline=outline)
    draw_rect(draw, [20, 14, 23, 17], fill=black) # Spot on head
    draw_rect(draw, [30, 20, 34, 23], fill=black) # Nose

    # Floppy Black Ears
    draw_rect(draw, [14, 12, 19, 25], fill=black)
    draw_rect(draw, [45, 12, 50, 25], fill=black)

    # Brown Eyes
    draw_rect(draw, [23, 16, 27, 20], fill=(110, 60, 30, 255))
    draw_rect(draw, [24, 17, 25, 18], fill=black)
    draw_rect(draw, [37, 16, 41, 20], fill=(110, 60, 30, 255))
    draw_rect(draw, [38, 17, 39, 18], fill=black)

    # Tongue on odd frames
    if frame_idx % 2 == 1:
        draw_rect(draw, [30, 25, 34, 29], fill=pink)

    # Paws
    draw_rect(draw, [20 + offset, 50, 26 + offset, 58], fill=white, outline=outline)
    draw_rect(draw, [38 - offset, 50, 44 - offset, 58], fill=white, outline=outline)

    return img

def create_cyber_dog(frame_idx=0):
    """Dog 4: Futuristic Sci-Fi Cyber Dog with Visor & Neon Glow."""
    img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    dark_metal = (40, 45, 55, 255)
    light_metal = (110, 120, 140, 255)
    cyan_glow = (0, 230, 255, 255)
    red_visor = (255, 40, 60, 255)
    black = (15, 15, 20, 255)

    offset = (frame_idx % 2) * 2
    wag = ((frame_idx % 4) - 2) * 3

    # Antenna / Tail
    draw.line([(44, 38), (54 + wag, 26)], fill=light_metal, width=3)
    draw_rect(draw, [52 + wag, 24, 56 + wag, 28], fill=cyan_glow)

    # Cyber Body
    draw_rect(draw, [20, 26, 44, 50], fill=dark_metal, outline=black)
    draw_rect(draw, [26, 32, 38, 44], fill=light_metal)
    draw_rect(draw, [30, 36, 34, 40], fill=cyan_glow) # Energy core

    # Head
    draw_rect(draw, [18, 12, 46, 28], fill=dark_metal, outline=black)
    draw_rect(draw, [22, 14, 42, 21], fill=red_visor) # Glowing Red Visor
    draw_rect(draw, [26, 16, 38, 18], fill=(255, 200, 210, 255)) # Visor highlight

    # Cyber Ears / Sensors
    draw_rect(draw, [15, 8, 20, 20], fill=light_metal, outline=black)
    draw_rect(draw, [44, 8, 49, 20], fill=light_metal, outline=black)

    # Snout
    draw_rect(draw, [27, 22, 37, 27], fill=light_metal)
    draw_rect(draw, [30, 22, 34, 24], fill=cyan_glow)

    # Metallic Paws
    draw_rect(draw, [20 + offset, 50, 26 + offset, 58], fill=light_metal, outline=black)
    draw_rect(draw, [38 - offset, 50, 44 - offset, 58], fill=light_metal, outline=black)

    return img


# -------------------- MAIN GENERATION LOGIC -------------------- #

def main():
    out_dir = "examples/dog_and_cat_showcase"
    os.makedirs(out_dir, exist_ok=True)

    cat_generators = {
        "ginger_cat": create_ginger_cat,
        "black_cat": create_black_cat,
        "siamese_cat": create_siamese_cat,
        "calico_cat": create_calico_cat,
    }

    dog_generators = {
        "shiba_dog": create_shiba_dog,
        "husky_dog": create_husky_dog,
        "dalmatian_dog": create_dalmatian_dog,
        "cyber_dog": create_cyber_dog,
    }

    print("Generating 4 Cat Characters...")
    for name, gen_fn in cat_generators.items():
        # Single Image
        single = gen_fn(0)
        single.save(f"{out_dir}/{name}_single.png")

        # 4-frame Sprite Sheet (256x64)
        frames = [gen_fn(i) for i in range(4)]
        sheet = Image.new('RGBA', (256, 64), (0, 0, 0, 0))
        for i, frame in enumerate(frames):
            sheet.paste(frame, (i * 64, 0))
        sheet.save(f"{out_dir}/{name}_spritesheet.png")

        # Animated GIF
        frames_np = [np.array(f) for f in frames]
        imageio.mimsave(f"{out_dir}/{name}_animation.gif", frames_np, fps=4, loop=0)
        print(f" Saved: {name} (PNG, Spritesheet, GIF)")

    print("\nGenerating 4 Dog Characters...")
    for name, gen_fn in dog_generators.items():
        # Single Image
        single = gen_fn(0)
        single.save(f"{out_dir}/{name}_single.png")

        # 4-frame Sprite Sheet (256x64)
        frames = [gen_fn(i) for i in range(4)]
        sheet = Image.new('RGBA', (256, 64), (0, 0, 0, 0))
        for i, frame in enumerate(frames):
            sheet.paste(frame, (i * 64, 0))
        sheet.save(f"{out_dir}/{name}_spritesheet.png")

        # Animated GIF
        frames_np = [np.array(f) for f in frames]
        imageio.mimsave(f"{out_dir}/{name}_animation.gif", frames_np, fps=4, loop=0)
        print(f" Saved: {name} (PNG, Spritesheet, GIF)")

    # Legacy Backward Compatibility files
    cat_generators["ginger_cat"](0).save(f"{out_dir}/cat_character_single.png")
    cat_sheet = Image.new('RGBA', (256, 64), (0, 0, 0, 0))
    for i in range(4):
        cat_sheet.paste(cat_generators["ginger_cat"](i), (i * 64, 0))
    cat_sheet.save(f"{out_dir}/cat_character_spritesheet.png")
    imageio.mimsave(f"{out_dir}/cat_character_animation.gif", [np.array(cat_generators["ginger_cat"](i)) for i in range(4)], fps=4, loop=0)

    dog_generators["shiba_dog"](0).save(f"{out_dir}/dog_character_single.png")
    dog_sheet = Image.new('RGBA', (256, 64), (0, 0, 0, 0))
    for i in range(4):
        dog_sheet.paste(dog_generators["shiba_dog"](i), (i * 64, 0))
    dog_sheet.save(f"{out_dir}/dog_character_spritesheet.png")
    imageio.mimsave(f"{out_dir}/dog_character_animation.gif", [np.array(dog_generators["shiba_dog"](i)) for i in range(4)], fps=4, loop=0)

    # Master Composite Showcase Sheets (256x256)
    master_cats = Image.new('RGBA', (256, 256), (20, 20, 30, 255))
    for idx, name in enumerate(["ginger_cat", "black_cat", "siamese_cat", "calico_cat"]):
        r, c = idx // 2, idx % 2
        sprite = cat_generators[name](0).resize((128, 128), Image.NEAREST)
        master_cats.paste(sprite, (c * 128, r * 128), sprite)
    master_cats.save(f"{out_dir}/master_cats_showcase.png")

    master_dogs = Image.new('RGBA', (256, 256), (20, 20, 30, 255))
    for idx, name in enumerate(["shiba_dog", "husky_dog", "dalmatian_dog", "cyber_dog"]):
        r, c = idx // 2, idx % 2
        sprite = dog_generators[name](0).resize((128, 128), Image.NEAREST)
        master_dogs.paste(sprite, (c * 128, r * 128), sprite)
    master_dogs.save(f"{out_dir}/master_dogs_showcase.png")

    print("\nMaster Showcase Sheets created successfully (master_cats_showcase.png & master_dogs_showcase.png)!")

if __name__ == "__main__":
    main()
