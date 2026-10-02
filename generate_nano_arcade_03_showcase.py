"""
Generate Nano Arcade 03 Showcase Samples
Generates 64x64 Text-to-Image, Image-to-Image, and GIF Action Animation samples using NanoArcade03Generator.
"""
import os
from PIL import Image
from models.nano_arcade_03.inference.generator import NanoArcade03Generator

def main():
    out_dir = "examples/2026-09-02_nano_arcade_03_showcase"
    os.makedirs(out_dir, exist_ok=True)

    print("[Showcase] Initializing NanoArcade03Generator...")
    gen = NanoArcade03Generator()

    # 1. Text-to-Image generation across archetypes
    prompts = [
        ("knight_t2i.png", "pixel knight warrior in golden armor"),
        ("wizard_t2i.png", "pixel mage wizard casting magic spell"),
        ("monster_t2i.png", "pixel dragon monster with fire wings"),
        ("robot_t2i.png", "pixel mecha robot hero")
    ]

    for fname, prompt in prompts:
        img = gen.text_to_sprite(prompt, seed=100)
        img.save(os.path.join(out_dir, fname))
        print(f"[Showcase] Saved Text-to-Image sample: {fname}")

    # 2. Image-to-Image (Img2Img) conversion
    base_img = Image.new("RGBA", (64, 64), (100, 150, 250, 255))
    img2img_out = gen.image_to_sprite(base_img, strength=0.6)
    img2img_out.save(os.path.join(out_dir, "img2img_sample.png"))
    print("[Showcase] Saved Image-to-Image sample: img2img_sample.png")

    # 3. GIF Action Animation generation
    gif_bytes = gen.generate_animation_gif("running arcade hero", num_frames=4)
    gif_path = os.path.join(out_dir, "hero_animation.gif")
    with open(gif_path, "wb") as f:
        f.write(gif_bytes)
    print(f"[Showcase] Saved GIF Animation sample: hero_animation.gif")

    print("[Showcase] All showcase samples successfully generated and saved!")

if __name__ == "__main__":
    main()
