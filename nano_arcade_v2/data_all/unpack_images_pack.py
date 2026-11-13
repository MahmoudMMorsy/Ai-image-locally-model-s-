#!/usr/bin/env python3
"""Unpack base64 image pack JSON files into all_64/."""
import json, base64, sys
from pathlib import Path
root = Path(__file__).resolve().parent
out = root / 'all_64'
out.mkdir(parents=True, exist_ok=True)
packs = sorted(root.glob('images_pack_*.json'))
if not packs:
    print('no packs found'); sys.exit(1)
n = 0
for pack in packs:
    data = json.loads(pack.read_text(encoding='utf-8'))
    for name, b64 in data.items():
        (out / name).write_bytes(base64.b64decode(b64))
        n += 1
    print('unpacked', pack.name, 'total', n)
print('DONE', n, 'images ->', out)
