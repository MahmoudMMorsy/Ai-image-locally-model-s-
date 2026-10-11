"""CLI text-to-image generator for 64x64 arcade sprites."""
import os
import argparse
import torch
import numpy as np
from PIL import Image

from model_t2i import TextCVAE
from text_lang import TextProcessor

CLASS_KEYWORD_MAP = {
    "knight": 0, "فارس": 0,
    "mage": 1, "wizard": 1, "ساحر": 1,
    "robot": 2, "mech": 2, "روبوت": 2,
    "cat": 3, "قطة": 3, "قطه": 3,
    "dragon": 4, "تنين": 4,
    "warrior": 5, "محارب": 5,
    "archer": 6, "رامي": 6,
    "monster": 7, "وحش": 7,
    "ghost": 8, "شبح": 8,
    "rogue": 9, "لص": 9,
    "pirate": 10, "قرصان": 10,
    "princess": 11, "أميرة": 11, "اميره": 11,
    "animal": 12, "حيوان": 12,
}

def infer_class_id(prompt: str) -> int:
    prompt_norm = prompt.lower()
    for kw, cid in CLASS_KEYWORD_MAP.items():
        if kw in prompt_norm:
            return cid
    return 13

def indices_to_pil(indices: np.ndarray, palette_rgb: np.ndarray) -> Image.Image:
    rgb = palette_rgb[indices]
    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = np.where(indices == 0, 0, 255)
    return Image.fromarray(rgba, mode="RGBA")

def main():
    parser = argparse.ArgumentParser(description="Generate 64x64 retro arcade pixel art sprite from text prompt.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt in Arabic or English")
    parser.add_argument("--output", type=str, default="arcade_t2i/samples/output.png", help="Output PNG path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n", type=int, default=1, help="Number of variation samples")
    parser.add_argument("--gif", action="store_true", help="Generate 4-frame action GIF animation")
    parser.add_argument("--ckpt", type=str, default="arcade_t2i/checkpoints/t2i_cvae_best.pt", help="Checkpoint path")
    parser.add_argument("--palette", type=str, default="arcade_t2i/checkpoints/palette48.pt", help="Palette path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    text_proc = TextProcessor()
    tokens = text_proc.encode(args.prompt).to(device)
    class_id = torch.tensor(infer_class_id(args.prompt), dtype=torch.long, device=device)

    if not os.path.exists(args.palette):
        raise FileNotFoundError(f"Palette missing: {args.palette}")
    palette_tensor = torch.load(args.palette)
    palette_rgb = (palette_tensor.numpy() * 255.0).astype(np.uint8)

    model = TextCVAE(n_colors=48, latent=64, base=24, cond_dim=128, n_tokens=100).to(device)
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint missing: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.gif:
        frames = []
        base_z = torch.randn(1, model.latent, device=device)
        for i in range(4):
            z_offset = torch.randn_like(base_z) * 0.3 * i
            z_i = base_z + z_offset
            with torch.no_grad():
                gen_idx = model.generate(tokens, class_id, n=1, z=z_i, temperature=0.8).squeeze(0).cpu().numpy()
            img = indices_to_pil(gen_idx, palette_rgb)
            frames.append(img.resize((128, 128), Image.Resampling.NEAREST))

        gif_path = args.output if args.output.endswith(".gif") else args.output.replace(".png", ".gif")
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=200, loop=0)
        print(f"Saved 4-frame animation GIF to: {gif_path}")
    else:
        with torch.no_grad():
            gen_indices = model.generate(tokens, class_id, n=args.n, temperature=0.8).cpu().numpy()

        if args.n == 1:
            img = indices_to_pil(gen_indices[0], palette_rgb)
            img.save(args.output)
            print(f"Saved generated sprite to: {args.output}")
        else:
            grid_w = args.n * 64
            composite = Image.new("RGBA", (grid_w, 64), (0, 0, 0, 0))
            for idx in range(args.n):
                sub_img = indices_to_pil(gen_indices[idx], palette_rgb)
                composite.paste(sub_img, (idx * 64, 0))
            composite.save(args.output)
            print(f"Saved {args.n} sample grid to: {args.output}")

if __name__ == "__main__":
    main()
