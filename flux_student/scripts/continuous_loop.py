#!/usr/bin/env python3
"""Continuous teacher-gen + student-train loop. No interactive prompts."""
from __future__ import annotations
import json, os, random, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

LOG = ROOT / "continuous.log"
MANIFEST = ROOT / "data" / "manifest.jsonl"
CKPT = ROOT / "checkpoints" / "student_final.pt"
PROMPTS = ROOT / "scripts" / "prompts_seed.txt"
SIZES = (32, 64, 128, 256)

def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

def load_prompt_list():
    if PROMPTS.exists():
        ps = [ln.strip() for ln in PROMPTS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        ps = ["a red apple", "a knight", "a robot", "a cat", "a tree"]
    extra = []
    for p in list(ps):
        extra.append(p + ", highly detailed")
        extra.append(p + ", simple background")
        extra.append("pixel art " + p)
    return ps + extra

def try_generate_batch(n: int = 4) -> int:
    try:
        import torch
        from PIL import Image
        from diffusers import AutoPipelineForText2Image
    except Exception as e:
        log(f"gen skip (import): {e}")
        return 0
    prompts = load_prompt_list()
    meta = []
    if MANIFEST.exists():
        meta = [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    existing = {m["id"] for m in meta}
    start_idx = 0
    for m in meta:
        if m["id"].startswith("turbo_"):
            try:
                start_idx = max(start_idx, int(m["id"].split("_")[1]) + 1)
            except Exception:
                pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=dtype,
            variant="fp16" if device == "cuda" else None,
        )
        pipe = pipe.to(device)
    except Exception as e:
        log(f"gen skip (load pipe): {e}")
        return 0
    added = 0
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        idx = start_idx + k
        sid = f"turbo_{idx:05d}"
        if sid in existing:
            continue
        p = prompts[(start_idx + k) % len(prompts)]
        try:
            log(f"gen {sid}: {p[:50]}")
            img = pipe(p, num_inference_steps=1, guidance_scale=0.0, height=256, width=256).images[0]
            img.save(raw / f"{sid}.png")
            rgb = img.convert("RGB")
            for s in SIZES:
                out = rgb.resize((s, s), Image.Resampling.NEAREST if s <= 64 else Image.Resampling.LANCZOS)
                d = ROOT / "data" / "resized" / str(s)
                d.mkdir(parents=True, exist_ok=True)
                out.save(d / f"{sid}.png")
            meta.append({"id": sid, "prompt": p, "teacher": "sd-turbo"})
            added += 1
        except Exception as e:
            log(f"gen fail {sid}: {e}")
            break
    with MANIFEST.open("w", encoding="utf-8") as f:
        for row in meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"gen done added={added} total={len(meta)}")
    del pipe
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return added

def train_epochs(epochs: int = 10) -> float:
    try:
        import numpy as np
        from PIL import Image, ImageOps
        import torch
        import torch.nn.functional as F
        from student.model import StudentT2I
    except Exception as e:
        log(f"train skip (import): {e}")
        return -1.0
    items = []
    if MANIFEST.exists():
        for ln in MANIFEST.read_text(encoding="utf-8").splitlines():
            row = json.loads(ln)
            p = ROOT / "data" / "resized" / "64" / f"{row['id']}.png"
            if p.exists():
                items.append((p, row["prompt"]))
    if not items:
        log("train skip: no data")
        return -1.0
    def tok(text, vocab=16384, max_len=16):
        ids = [1] + [((hash(w) % (vocab - 2)) + 2) for w in text.lower().split()[: max_len - 1]]
        return torch.tensor((ids + [0] * max_len)[:max_len])
    def lx(path, flip=False):
        im = Image.open(path).convert("RGB").resize((64, 64), Image.NEAREST)
        if flip:
            im = ImageOps.mirror(im)
        return torch.from_numpy(np.asarray(im).astype("float32") / 127.5 - 1).permute(2, 0, 1)
    model = StudentT2I(base=48)
    if CKPT.exists():
        try:
            st = torch.load(CKPT, map_location="cpu", weights_only=False)
            model.load_state_dict(st["model"], strict=False)
            log("train resumed checkpoint")
        except Exception as e:
            log(f"train resume fail: {e}")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    last = 0.0
    for ep in range(epochs):
        random.shuffle(items)
        total = 0.0
        n = 0
        for path, pr in items:
            for flip in (False, True):
                x = lx(path, flip).unsqueeze(0)
                ids = tok(pr).unsqueeze(0)
                t = torch.randint(0, 1000, (1,))
                noise = torch.randn_like(x)
                a = (1 - t.float() / 1000).view(1, 1, 1, 1)
                loss = F.mse_loss(model(a * x + (1 - a) * noise, t, ids), noise)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
                n += 1
        last = total / max(n, 1)
        log(f"train ep {ep+1}/{epochs} loss={last:.4f} n={len(items)}")
    ROOT.joinpath("checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "base": 48, "n": len(items), "loss": last}, CKPT)
    log(f"train saved loss={last:.4f}")
    return last

def dump_samples():
    try:
        import numpy as np
        from PIL import Image
        import torch
        from student.model import StudentT2I
    except Exception as e:
        log(f"sample skip: {e}")
        return
    if not CKPT.exists():
        return
    def tok(text, vocab=16384, max_len=16):
        ids = [1] + [((hash(w) % (vocab - 2)) + 2) for w in text.lower().split()[: max_len - 1]]
        return torch.tensor((ids + [0] * max_len)[:max_len])
    model = StudentT2I(base=48)
    st = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(st["model"], strict=False)
    model.eval()
    out = ROOT / "samples"
    out.mkdir(exist_ok=True)
    prompts = ["a red apple on a wooden table", "a knight in shiny armor", "a robot standing in a laboratory"]
    with torch.no_grad():
        for i, pr in enumerate(prompts):
            ids = tok(pr).unsqueeze(0)
            x = torch.randn(1, 3, 64, 64)
            for ti in reversed(range(0, 1000, 50)):
                t = torch.tensor([ti])
                eps = model(x, t, ids)
                a = 1 - ti / 1000.0 * 0.9
                x = (x - (1 - a) * eps) / max(a, 1e-3) ** 0.5
                if ti > 0:
                    x = x + torch.randn_like(x) * (ti / 1000) * 0.1
            img = ((x[0].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).numpy().astype("uint8")
            Image.fromarray(img).resize((256, 256), Image.NEAREST).save(out / f"loop_gen_{i}.png")
    log("samples dumped")

def main():
    once = "--once" in sys.argv
    log("=== continuous_loop START ===")
    cycle = 0
    while True:
        cycle += 1
        log(f"=== cycle {cycle} ===")
        try:
            try_generate_batch(n=1)
        except Exception:
            log("gen exception:\n" + traceback.format_exc())
        try:
            train_epochs(epochs=2)
        except Exception:
            log("train exception:\n" + traceback.format_exc())
        try:
            dump_samples()
        except Exception:
            log("sample exception:\n" + traceback.format_exc())
        status = {
            "cycle": cycle,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "manifest_lines": sum(1 for _ in MANIFEST.open()) if MANIFEST.exists() else 0,
            "ckpt": str(CKPT) if CKPT.exists() else None,
        }
        (ROOT / "continuous_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        if once:
            log(f"cycle {cycle} completed (--once flag). Exiting.")
            break
        log(f"cycle {cycle} sleep 30s")
        time.sleep(30)

if __name__ == "__main__":
    main()
