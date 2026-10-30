"""Train student from resized teacher images."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from student.model import StudentT2I, count_params


class CaptionImageDS(Dataset):
    def __init__(self, size: int = 64, vocab: int = 16384):
        self.size = size
        self.vocab = vocab
        self.items = []
        manifest = ROOT / "data" / "manifest.jsonl"
        if manifest.exists():
            for ln in manifest.read_text(encoding="utf-8").splitlines():
                row = json.loads(ln)
                img_path = ROOT / "data" / "resized" / str(size) / f"{row['id']}.png"
                if img_path.exists():
                    self.items.append((img_path, row["prompt"]))
        if not self.items:
            raise RuntimeError(f"No training images at size {size}. Run generate_teacher_data.py first.")

    def _tok(self, text: str, max_len: int = 20) -> torch.Tensor:
        ids = [1]
        for w in text.lower().split()[: max_len - 1]:
            ids.append((hash(w) % (self.vocab - 2)) + 2)
        ids += [0] * (max_len - len(ids))
        return torch.tensor(ids[:max_len], dtype=torch.long)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, prompt = self.items[idx]
        img = Image.open(path).convert("RGB").resize((self.size, self.size), Image.NEAREST)
        arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0
        x = torch.from_numpy(arr).permute(2, 0, 1)
        return x, self._tok(prompt)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ds = CaptionImageDS(size=args.size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=False)
    model = StudentT2I(base=args.base).to(device)
    print(f"student params={count_params(model)/1e6:.2f}M device={device} n={len(ds)}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        total = 0.0
        for x, ids in dl:
            x, ids = x.to(device), ids.to(device)
            t = torch.randint(0, 1000, (x.size(0),), device=device)
            noise = torch.randn_like(x)
            a = (1 - t.float() / 1000.0).view(-1, 1, 1, 1)
            xt = a * x + (1 - a) * noise
            pred = model(xt, t, ids)
            loss = F.mse_loss(pred, noise)
            if args.recon > 0:
                t0 = torch.zeros_like(t)
                loss = loss + args.recon * F.mse_loss(model(x, t0, ids), x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / max(1, len(dl))
        print(f"epoch {ep+1}/{args.epochs} loss={avg:.4f}")
        torch.save({"model": model.state_dict(), "epoch": ep + 1, "loss": avg}, ckpt_dir / "student_latest.pt")
    torch.save({"model": model.state_dict(), "epoch": args.epochs}, ckpt_dir / "student_final.pt")
    print("saved", ckpt_dir / "student_final.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64, choices=[32, 64, 128, 256])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=96)
    ap.add_argument("--recon", type=float, default=0.05)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
