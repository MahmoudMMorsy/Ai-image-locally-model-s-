"""Inference script for StudentT2I compact neural character generator."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from student.model import StudentT2I
from student.filter import is_safe_prompt


def tokenize(text: str, vocab: int = 16384, max_len: int = 20) -> torch.Tensor:
    ids = [1]
    for w in text.lower().split()[: max_len - 1]:
        ids.append((hash(w) % (vocab - 2)) + 2)
    ids += [0] * (max_len - len(ids))
    return torch.tensor([ids[:max_len]], dtype=torch.long)


def sample_character(
    prompt: str,
    checkpoint_path: Path,
    out_path: Path,
    target_size: int = 64,
    steps: int = 25,
    base: int = 48,
    device: str = "cpu"
):
    if not is_safe_prompt(prompt):
        raise ValueError(f"Prompt violated content safety policy: {prompt}")

    dev = torch.device(device)
    model = StudentT2I(base=base).to(dev)

    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=dev, weights_only=False)
        if "model" in state:
            model.load_state_dict(state["model"], strict=False)
        else:
            model.load_state_dict(state, strict=False)
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using initial weights.")

    model.eval()
    ids = tokenize(prompt).to(dev)

    with torch.no_grad():
        tensors = model.sample(ids, size=64, steps=steps, device=dev)
        arr = ((tensors[0].clamp(-1, 1) + 1.0) * 127.5).permute(1, 2, 0).cpu().numpy().astype("uint8")
        img = Image.fromarray(arr)

        if target_size != 64:
            resample = Image.Resampling.NEAREST if target_size <= 64 else Image.Resampling.LANCZOS
            img = img.resize((target_size, target_size), resample=resample)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        print(f"Generated sample saved to {out_path} ({target_size}x{target_size})")
        return img


def main():
    parser = argparse.ArgumentParser(description="Student T2I Character Inference")
    parser.add_argument("--prompt", type=str, required=True, help="Character description prompt")
    parser.add_argument("--checkpoint", type=str, default=str(ROOT / "checkpoints" / "student_final.pt"))
    parser.add_argument("--out", type=str, default=str(ROOT / "samples" / "sample.png"))
    parser.add_argument("--size", type=int, default=64, choices=[32, 64, 128, 256])
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--base", type=int, default=48)
    parser.add_argument("--cpu", action="store_true", default=True)

    args = parser.parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    sample_character(
        prompt=args.prompt,
        checkpoint_path=Path(args.checkpoint),
        out_path=Path(args.out),
        target_size=args.size,
        steps=args.steps,
        base=args.base,
        device=device
    )


if __name__ == "__main__":
    main()
