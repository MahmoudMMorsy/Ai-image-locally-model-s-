"""Generate synthetic training images from a teacher model, then resize to 32/64/128/256.

Preferred teacher: FLUX.1-schnell (heavy download).
Fallback: sd-turbo / small SD for pipeline bootstrap when disk/RAM limited.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import torch
import sys
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from student.filter import is_safe_prompt

SIZES = (32, 64, 128, 256)


def load_prompts(path: Path, limit: int | None) -> list[str]:
    raw_lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    lines = [ln for ln in raw_lines if is_safe_prompt(ln)]
    if limit:
        lines = lines[:limit]
    return lines


def save_pyramid(img: Image.Image, stem: Path):
    rgb = img.convert("RGB")
    for s in SIZES:
        out = rgb.resize((s, s), Image.Resampling.NEAREST if s <= 64 else Image.Resampling.LANCZOS)
        out_path = ROOT / "data" / "resized" / str(s) / f"{stem.name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.save(out_path)


def try_flux(prompts: list[str], out_meta: list, max_n: int, steps: int):
    from diffusers import FluxPipeline
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[teacher] loading FLUX.1-schnell on {device} ...")
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=dtype)
    if device == "cpu":
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
    else:
        pipe.to(device)
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(prompts[:max_n]):
        print(f"[FLUX] {i+1}/{min(max_n,len(prompts))}: {p[:60]}")
        img = pipe(p, num_inference_steps=steps, guidance_scale=0.0, height=256, width=256).images[0]
        stem = Path(f"flux_{i:05d}")
        img.save(raw / f"{stem}.png")
        save_pyramid(img, stem)
        out_meta.append({"id": stem.name, "prompt": p, "teacher": "flux-schnell"})
    return True


def try_sd_turbo(prompts: list[str], out_meta: list, max_n: int, steps: int):
    from diffusers import AutoPipelineForText2Image
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[teacher] loading sd-turbo on {device} (FLUX fallback)...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
    )
    pipe = pipe.to(device)
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(prompts[:max_n]):
        print(f"[sd-turbo] {i+1}/{min(max_n,len(prompts))}: {p[:60]}")
        img = pipe(p, num_inference_steps=max(steps, 1), guidance_scale=0.0, height=256, width=256).images[0]
        stem = Path(f"turbo_{i:05d}")
        img.save(raw / f"{stem}.png")
        save_pyramid(img, stem)
        out_meta.append({"id": stem.name, "prompt": p, "teacher": "sd-turbo"})
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=str, default=str(ROOT / "scripts" / "prompts_seed.txt"))
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--teacher", choices=["auto", "flux", "sd-turbo"], default="auto")
    args = ap.parse_args()

    prompts = load_prompts(Path(args.prompts), None)
    while len(prompts) < args.n:
        prompts += [p + " , highly detailed" for p in prompts[: max(1, args.n - len(prompts))]]
    meta = []
    ok = False
    if args.teacher in ("auto", "flux"):
        try:
            ok = try_flux(prompts, meta, args.n, args.steps)
        except Exception as e:
            print("[FLUX] failed:", e)
            ok = False
    if not ok and args.teacher in ("auto", "sd-turbo"):
        ok = try_sd_turbo(prompts, meta, args.n, args.steps)
    meta_path = ROOT / "data" / "manifest.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for row in meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(meta)} samples -> {meta_path}")


if __name__ == "__main__":
    main()
