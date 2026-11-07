#!/usr/bin/env python3
"""Build a palette-index dataset from isolated RGBA sprites.
Transparent pixels are index 0; each image gets its own 16-color RGB palette.
Nearest-neighbor is used for normalization so pixel edges are never blurred.
"""
from pathlib import Path
import argparse, torch, numpy as np
from PIL import Image

def encode(path, size=32, colors=16):
    im=Image.open(path).convert('RGBA').resize((size,size),Image.Resampling.NEAREST)
    rgba=np.array(im); alpha=rgba[...,3]
    rgb=rgba[...,:3]
    # Quantize only visible pixels, then reserve index 0 for transparency.
    visible=Image.fromarray(rgb).convert('RGB')
    q=visible.quantize(colors=colors,method=Image.Quantize.MEDIANCUT,dither=Image.Dither.NONE)
    idx=np.array(q,dtype=np.int64)+1
    idx[alpha<16]=0
    raw=q.getpalette()[:colors*3]
    if len(raw)<colors*3: raw=raw+[0]*(colors*3-len(raw))
    pal=np.array(raw,dtype=np.uint8).reshape(colors,3)
    # Index 0 is reserved for transparency; shift visible palette entries.
    padded=np.zeros((colors+1,3),dtype=np.uint8)
    padded[1:]=pal
    return idx,padded

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--out',required=True);ap.add_argument('--size',type=int,default=32);args=ap.parse_args()
    root=Path(args.root); files=[]
    for p in sorted(root.rglob('*.png')):
        try:
            im=Image.open(p)
            if im.mode in ('RGBA','LA') and im.size[0] <= 128 and 'sheet' not in p.name.lower(): files.append(p)
        except Exception: pass
    # Deduplicate by source hash while retaining different examples.
    import hashlib
    seen=set(); selected=[]
    for p in files:
        h=hashlib.sha1(p.read_bytes()).hexdigest()
        if h not in seen: seen.add(h); selected.append(p)
    if not selected: raise SystemExit('no isolated RGBA sprites found')
    xs=[]; ps=[]; names=[]
    for p in selected:
        x,pal=encode(p,args.size);xs.append(x);ps.append(pal);names.append(str(p))
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'indices':torch.from_numpy(np.stack(xs)).long(),'palettes':torch.from_numpy(np.stack(ps)).to(torch.uint8),'count':len(xs),'sources':names},out)
    print({'count':len(xs),'unique_exact':len(np.unique(np.stack(xs).reshape(len(xs),-1),axis=0)),'out':str(out)})
if __name__=='__main__':main()
