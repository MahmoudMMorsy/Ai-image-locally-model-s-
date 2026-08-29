"""
Pixel Art Engine Command Line Interface (CLI)
Provides command-line commands for Text-to-Sprite, Image-to-Sprite, and Animation Generation.
"""
import argparse
import sys
from PIL import Image
from pixel_art_engine.engine import PixelSpriteEngine
from pixel_art_engine.converter import ImageToSpriteConverter
from pixel_art_engine.animation import SpriteAnimationGenerator


def main():
    parser = argparse.ArgumentParser(description="64x64 Pixel Art Character Generator & Animator Engine")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Command: generate (Text-to-Sprite)
    gen_parser = subparsers.add_parser("generate", help="Generate 64x64 pixel art character from text prompt")
    gen_parser.add_argument("--prompt", type=str, default="pixel knight warrior", help="Character description")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    gen_parser.add_argument("--out", type=str, default="generated_character.png", help="Output PNG path")

    # Command: convert (Image-to-Sprite)
    conv_parser = subparsers.add_parser("convert", help="Convert any image to 64x64 pixel art sprite")
    conv_parser.add_argument("--input", type=str, required=True, help="Path to input image")
    conv_parser.add_argument("--strength", type=float, default=0.4, help="Model conversion strength (0.0 to 1.0)")
    conv_parser.add_argument("--out", type=str, default="converted_sprite.png", help="Output PNG path")

    # Command: animate (Sequential Animation)
    anim_parser = subparsers.add_parser("animate", help="Generate sequential action animation for character")
    anim_parser.add_argument("--input", type=str, help="Path to base character PNG (or generates one if omitted)")
    anim_parser.add_argument("--prompt", type=str, default="pixel hero warrior", help="Prompt if generating base character")
    anim_parser.add_argument("--action", type=str, default="run", choices=["run", "walk", "attack", "jump", "idle"], help="Action sequence")
    anim_parser.add_argument("--frames", type=int, default=4, help="Number of animation frames")
    anim_parser.add_argument("--out_sheet", type=str, default="sprite_sheet.png", help="Output Sprite Sheet PNG path")
    anim_parser.add_argument("--out_gif", type=str, default="character_anim.gif", help="Output Animated GIF path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    engine = PixelSpriteEngine()

    if args.command == "generate":
        print(f"Generating sprite for prompt: '{args.prompt}' (Seed: {args.seed})...")
        img = engine.generate_sprite(prompt=args.prompt, seed=args.seed)
        img.save(args.out)
        print(f"Saved generated character to {args.out} (64x64 RGBA).")

    elif args.command == "convert":
        print(f"Converting image '{args.input}' to 64x64 pixel art sprite...")
        inp_img = Image.open(args.input)
        converter = ImageToSpriteConverter(engine)
        sprite = converter.convert_image_to_sprite(inp_img, strength=args.strength)
        sprite.save(args.out)
        print(f"Saved converted sprite to {args.out} (64x64 RGBA).")

    elif args.command == "animate":
        if args.input:
            print(f"Loading base character from '{args.input}'...")
            base_char = Image.open(args.input)
        else:
            print(f"Generating base character for prompt '{args.prompt}'...")
            base_char = engine.generate_sprite(prompt=args.prompt, seed=42)

        animator = SpriteAnimationGenerator(engine)
        print(f"Generating '{args.action}' animation ({args.frames} frames)...")
        frames, sheet, gif_bytes = animator.generate_animation(base_char, action=args.action, num_frames=args.frames)

        sheet.save(args.out_sheet)
        with open(args.out_gif, "wb") as f:
            f.write(gif_bytes)

        print(f"Saved sprite sheet to '{args.out_sheet}' and animated GIF to '{args.out_gif}'.")


if __name__ == "__main__":
    main()
