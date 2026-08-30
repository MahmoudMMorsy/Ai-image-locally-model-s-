import os
import imageio
import numpy as np
from PIL import Image, ImageDraw

def create_pixel_cat(frame_idx=0):
    # 64x64 RGBA pixel art cat
    img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Body & Fur (Orange / Ginger Cat)
    orange = (235, 130, 40, 255)
    dark_orange = (190, 90, 20, 255)
    white = (245, 245, 245, 255)
    pink = (240, 140, 160, 255)
    green = (40, 200, 80, 255)
    black = (20, 20, 25, 255)

    # Body
    draw.rectangle([22, 28, 42, 52], fill=orange, outline=dark_orange)
    # White belly patch
    draw.rectangle([27, 34, 37, 50], fill=white)

    # Head
    draw.rectangle([20, 16, 44, 32], fill=orange, outline=dark_orange)

    # Ears
    draw.polygon([(20, 16), (25, 6), (29, 16)], fill=orange, outline=dark_orange)
    draw.polygon([(22, 16), (25, 9), (27, 16)], fill=pink)
    draw.polygon([(35, 16), (39, 6), (44, 16)], fill=orange, outline=dark_orange)
    draw.polygon([(37, 16), (39, 9), (42, 16)], fill=pink)

    # Eyes
    draw.rectangle([25, 21, 28, 25], fill=green)
    draw.rectangle([26, 22, 27, 24], fill=black)
    draw.rectangle([36, 21, 39, 25], fill=green)
    draw.rectangle([37, 22, 38, 24], fill=black)

    # Nose & Mouth
    draw.rectangle([31, 26, 33, 27], fill=pink)
    draw.line([(32, 27), (32, 29)], fill=dark_orange, width=1)

    # Whiskers
    draw.line([(14, 25), (20, 26)], fill=white, width=1)
    draw.line([(14, 28), (20, 28)], fill=white, width=1)
    draw.line([(44, 26), (50, 25)], fill=white, width=1)
    draw.line([(44, 28), (50, 28)], fill=white, width=1)

    # Animated Paws & Tail
    offset = (frame_idx % 2) * 2
    tail_y = 44 + ((frame_idx % 4) - 2) * 2

    # Tail
    draw.line([(42, 46), (52, tail_y), (56, tail_y - 4)], fill=orange, width=3)

    # Paws
    draw.rectangle([22 + offset, 52, 27 + offset, 58], fill=white, outline=dark_orange)
    draw.rectangle([37 - offset, 52, 42 - offset, 58], fill=white, outline=dark_orange)

    return img

def create_pixel_dog(frame_idx=0):
    # 64x64 RGBA pixel art dog
    img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Colors (Brown / Golden Husky-Shiba mix)
    brown = (180, 110, 50, 255)
    dark_brown = (120, 65, 20, 255)
    cream = (245, 225, 190, 255)
    black = (25, 25, 30, 255)
    red_collar = (220, 40, 40, 255)
    tongue_pink = (245, 110, 140, 255)

    # Body
    draw.rectangle([20, 26, 44, 50], fill=brown, outline=dark_brown)
    # Cream chest
    draw.rectangle([26, 32, 38, 48], fill=cream)
    # Red Collar
    draw.rectangle([20, 26, 44, 29], fill=red_collar)

    # Head
    draw.rectangle([18, 12, 46, 28], fill=brown, outline=dark_brown)
    # Snout
    draw.rectangle([26, 20, 38, 28], fill=cream, outline=dark_brown)
    draw.rectangle([30, 20, 34, 23], fill=black) # Nose

    # Floppy / Perky Ears
    draw.rectangle([14, 12, 20, 24], fill=dark_brown)
    draw.rectangle([44, 12, 50, 24], fill=dark_brown)

    # Eyes
    draw.rectangle([23, 16, 27, 20], fill=black)
    draw.rectangle([24, 17, 25, 18], fill=(255, 255, 255, 255))
    draw.rectangle([37, 16, 41, 20], fill=black)
    draw.rectangle([38, 17, 39, 18], fill=(255, 255, 255, 255))

    # Tongue out on odd frames
    if frame_idx % 2 == 1:
        draw.rectangle([30, 25, 34, 29], fill=tongue_pink)

    # Tail (wagging)
    wag = ((frame_idx % 4) - 2) * 3
    draw.line([(44, 38), (54 + wag, 30)], fill=brown, width=4)

    # Paws
    offset = (frame_idx % 2) * 2
    draw.rectangle([20 + offset, 50, 26 + offset, 58], fill=cream, outline=dark_brown)
    draw.rectangle([38 - offset, 50, 44 - offset, 58], fill=cream, outline=dark_brown)

    return img

def main():
    out_dir = "examples/dog_and_cat_showcase"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Single Images
    cat_single = create_pixel_cat(0)
    cat_single.save(f"{out_dir}/cat_character_single.png")

    dog_single = create_pixel_dog(0)
    dog_single.save(f"{out_dir}/dog_character_single.png")

    # 2. Sprite Sheets (4 frames 256x64)
    cat_frames = [create_pixel_cat(i) for i in range(4)]
    cat_sheet = Image.new('RGBA', (256, 64), (0, 0, 0, 0))
    for i, frame in enumerate(cat_frames):
        cat_sheet.paste(frame, (i * 64, 0))
    cat_sheet.save(f"{out_dir}/cat_character_spritesheet.png")

    dog_frames = [create_pixel_dog(i) for i in range(4)]
    dog_sheet = Image.new('RGBA', (256, 64), (0, 0, 0, 0))
    for i, frame in enumerate(dog_frames):
        dog_sheet.paste(frame, (i * 64, 0))
    dog_sheet.save(f"{out_dir}/dog_character_spritesheet.png")

    # 3. Animated GIFs
    cat_frames_np = [np.array(f) for f in cat_frames]
    imageio.mimsave(f"{out_dir}/cat_character_animation.gif", cat_frames_np, fps=4, loop=0)

    dog_frames_np = [np.array(f) for f in dog_frames]
    imageio.mimsave(f"{out_dir}/dog_character_animation.gif", dog_frames_np, fps=4, loop=0)

    print("Dog and Cat assets created successfully in examples/dog_and_cat_showcase/")

if __name__ == "__main__":
    main()
