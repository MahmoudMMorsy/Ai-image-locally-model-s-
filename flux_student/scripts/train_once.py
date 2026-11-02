#!/usr/bin/env python3
import json, sys, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from student.model import StudentT2I

items = []
for ln in open(ROOT / "data" / "manifest.jsonl"):
    r = json.loads(ln)
    p = ROOT / "data" / "resized" / "64" / f"{r['id']}.png"
    if p.exists():
        items.append((p, r["prompt"]))
print("n", len(items), flush=True)

def tok(t, v=16384, m=16):
    ids = [1] + [((hash(w) % (v - 2)) + 2) for w in t.lower().split()[: m - 1]]
    return torch.tensor((ids + [0] * m)[:m])

def lx(path, flip=False):
    im = Image.open(path).convert("RGB").resize((64, 64), Image.NEAREST)
    if flip:
        im = ImageOps.mirror(im)
    return torch.from_numpy(np.asarray(im).astype("float32") / 127.5 - 1).permute(2, 0, 1)

m = StudentT2I(base=48)
ck = ROOT / "checkpoints" / "student_final.pt"
m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"], strict=False)
opt = torch.optim.AdamW(m.parameters(), lr=1.5e-4)
last = 0.0
for ep in range(20):
    random.shuffle(items)
    tot = 0.0
    n = 0
    for path, pr in items:
        for flip in (False, True):
            x = lx(path, flip).unsqueeze(0)
            ids = tok(pr).unsqueeze(0)
            t = torch.randint(0, 1000, (1,))
            noise = torch.randn_like(x)
            a = (1 - t.float() / 1000).view(1, 1, 1, 1)
            loss = F.mse_loss(m(a * x + (1 - a) * noise, t, ids), noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1
    last = tot / max(n, 1)
    if (ep + 1) % 5 == 0 or ep == 0:
        print(f"ep{ep+1} {last:.4f}", flush=True)
torch.save({"model": m.state_dict(), "base": 48, "n": len(items), "loss": last}, ck)
print("SAVED", last, flush=True)
