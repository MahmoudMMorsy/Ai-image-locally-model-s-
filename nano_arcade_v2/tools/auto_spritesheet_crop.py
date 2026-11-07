#!/usr/bin/env python3
"""Auto spritesheet crop for Nano Arcade training.
Usage:
  python auto_spritesheet_crop.py sheet.jpg --out crops/ --size 64
"""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image

def detect_grid_equal(img):
    arr = np.array(img.convert("RGB"))
    H, W = arr.shape[:2]
    border = np.concatenate([arr[0].reshape(-1,3), arr[-1].reshape(-1,3),
                             arr[:,0].reshape(-1,3), arr[:,-1].reshape(-1,3)])
    bg = np.median(border, 0).astype(np.float32)
    dist = np.linalg.norm(arr.astype(np.float32) - bg, axis=2)
    fg = dist > 20
    best = None
    for nc in range(8, 16):
        for nr in range(6, 16):
            cw, ch = W/nc, H/nr
            if not (40 <= cw <= 250 and 40 <= ch <= 250):
                continue
            occs=[]
            for ri in range(nr):
                for ci in range(nc):
                    y0,y1=int(ri*ch), int((ri+1)*ch)
                    x0,x1=int(ci*cw), int((ci+1)*cw)
                    cell=fg[y0:y1, x0:x1]
                    occs.append(float(cell.mean()) if cell.size else 0.0)
            occs=np.array(occs)
            good=float(((occs>0.08)&(occs<0.7)).mean())
            empty=float((occs<0.03).mean())
            score=good-0.5*empty+0.01*min(nr*nc,120)/120
            if best is None or score>best[0]:
                best=(score,nr,nc,bg)
    score,nr,nc,bg=best
    r_splits=[int(round(i*H/nr)) for i in range(nr+1)]
    c_splits=[int(round(i*W/nc)) for i in range(nc+1)]
    return r_splits,c_splits,nr,nc,bg,score

def process_sheet(path, out_dir, size=64, prefix=None):
    path=Path(path); out_dir=Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or path.stem
    img=Image.open(path).convert("RGB")
    r_splits,c_splits,nr,nc,bg,score=detect_grid_equal(img)
    arr=np.array(img.convert("RGBA"))
    rgb=arr[...,:3].astype(np.float32)
    dist=np.linalg.norm(rgb-bg[None,None,:],axis=2)
    arr[...,3]=(np.clip((dist-12)/10,0,1)*255).astype(np.uint8)
    kept=[]
    for ri in range(nr):
        for ci in range(nc):
            pad=2
            y0=r_splits[ri]+pad; y1=r_splits[ri+1]-pad
            x0=c_splits[ci]+pad; x1=c_splits[ci+1]-pad
            if y1<=y0 or x1<=x0: continue
            cell=arr[y0:y1,x0:x1].copy()
            a=cell[...,3]
            if float((a>20).mean())<0.05: continue
            ys,xs=np.where(a>20)
            if len(ys)<30: continue
            crop=cell[ys.min():ys.max()+1, xs.min():xs.max()+1]
            h,w=crop.shape[:2]
            if min(h,w)<10: continue
            side=max(h,w)
            canvas=np.zeros((side,side,4),np.uint8)
            oy=(side-h)//2; ox=(side-w)//2
            canvas[oy:oy+h,ox:ox+w]=crop
            im=Image.fromarray(canvas,"RGBA").resize((size,size), Image.NEAREST)
            name=f"{prefix}_{ri:02d}_{ci:02d}.png"
            im.save(out_dir/name); kept.append(name)
    cols=10; rows=(len(kept)+cols-1)//cols
    grid=Image.new("RGB",(cols*66,max(1,rows)*66),(40,40,45))
    for j,name in enumerate(kept):
        im=Image.open(out_dir/name).convert("RGBA")
        bgim=Image.new("RGBA",(size,size),(40,40,45,255))
        bgim.paste(im,(0,0),im)
        grid.paste(bgim.convert("RGB"),((j%cols)*66,(j//cols)*66))
    grid.save(out_dir/f"{prefix}_contact.png")
    meta={"source":str(path),"grid":[nr,nc],"n_kept":len(kept),"score":float(score),"size":size}
    (out_dir/f"{prefix}_meta.json").write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))
    return kept, meta

if __name__=="__main__":
    ap=argparse.ArgumentParser(description="Auto crop spritesheet into individual 64x64 sprites")
    ap.add_argument("input")
    ap.add_argument("--out", default="crops")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--prefix", default=None)
    args=ap.parse_args()
    process_sheet(args.input, args.out, size=args.size, prefix=args.prefix)
