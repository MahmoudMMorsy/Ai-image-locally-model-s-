#!/usr/bin/env python3
from pathlib import Path
import json, time, hashlib
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
from vqvae_prior import VQVAEModel

ROOT=Path('/home/ubuntu/nano_arcade/clean_hq/v2')
OUT=Path('/home/ubuntu/nano_arcade/clean_hq/v2/codec_runs'); OUT.mkdir(exist_ok=True)
EPOCHS=20; BATCH=64; SEED=20261107

def main():
    torch.manual_seed(SEED); torch.set_num_threads(max(1,min(8,torch.get_num_threads())))
    train=torch.load(ROOT/'train.pt',map_location='cpu',weights_only=False)['indices'].long().clamp(0,47)
    val_pack=torch.load(ROOT/'val.pt',map_location='cpu',weights_only=False); val=val_pack['indices'].long().clamp(0,47)
    pal=torch.load('/home/ubuntu/nano_arcade/clean_hq/clean_hq_dataset.pt',map_location='cpu',weights_only=False)['palette'][:,:3].numpy()
    model=VQVAEModel(num_classes=48,num_embeddings=64,embedding_dim=32)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-3)
    hist=[]; best=None; start=time.time()
    for ep in range(1,EPOCHS+1):
        model.train(); order=torch.randperm(len(train)); total=0.0
        for i in range(0,len(order),BATCH):
            b=train[order[i:i+BATCH]]; logits,vql,codes=model(b)
            loss=F.cross_entropy(logits,b)+vql
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); total+=float(loss.detach())*len(b)
        model.eval()
        with torch.no_grad():
            logits,vql,codes=model(val); recon=logits.argmax(1)
            acc=float((recon==val).float().mean()); per=(recon==val).float().flatten(1).mean(1)
            used=int(torch.unique(codes).numel())
        row={'epoch':ep,'train_loss':total/len(train),'val_accuracy':acc,'val_min_accuracy':float(per.min()),'val_mean_sample_accuracy':float(per.mean()),'codebook_used':used}
        hist.append(row); print(json.dumps(row),flush=True)
        torch.save(model.state_dict(),OUT/f'codec_epoch_{ep:03d}.pt')
        score=(acc,float(per.min()))
        if best is None or score>(best['score'][0],best['score'][1]): best={'epoch':ep,'score':score,'path':str(OUT/f'codec_epoch_{ep:03d}.pt')}
    # Visual grid from the best checkpoint.
    state=torch.load(best['path'],map_location='cpu'); model.load_state_dict(state); model.eval()
    with torch.no_grad(): logits,_,_=model(val[:16]); recon=logits.argmax(1)
    grid=Image.new('RGB',(8*96,4*96),(35,35,35))
    for j in range(8):
        for row,arr in [(0,val[j].numpy()),(1,recon[j].numpy()),(2,val[j].numpy()),(3,recon[j].numpy())]:
            rgb=pal[np.asarray(arr).clip(0,47)].astype(np.uint8)
            im=Image.fromarray(rgb).resize((96,96),Image.Resampling.NEAREST)
            grid.paste(im,(j*96,row*96))
    grid_path=OUT/'codec_best_grid.png'; grid.save(grid_path)
    report={'seed':SEED,'epochs':EPOCHS,'batch':BATCH,'train_count':len(train),'val_count':len(val),'classes':48,'best':best,'history':hist,'grid':str(grid_path),'elapsed_seconds':time.time()-start,'accepted':bool(best['score'][0]>=.90 and best['score'][1]>=.80 and hist[-1]['codebook_used']>=8)}
    json.dump(report,open(OUT/'report.json','w'),indent=2)
    print(json.dumps({'best':best,'accepted':report['accepted'],'grid':str(grid_path)},indent=2),flush=True)
if __name__=='__main__': main()
