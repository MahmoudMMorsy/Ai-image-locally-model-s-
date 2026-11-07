#!/usr/bin/env python3
from pathlib import Path
import json,time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from vqvae_prior import VQVAEModel
ROOT=Path('/home/ubuntu/nano_arcade/clean_hq/strict_v2'); OUT=ROOT/'codec_runs_v5'; OUT.mkdir(exist_ok=True)
EPOCHS=20; BATCH=64; SEED=20261107

def score(pred,target):
    fg=target!=0; pred_fg=pred!=0; inter=(fg&pred_fg).sum().item(); union=(fg|pred_fg).sum().item()
    per=(pred==target).float().flatten(1).mean(1)
    fgacc=(pred[fg]==target[fg]).float().mean().item()
    return float((pred==target).float().mean()),inter/max(1,union),fgacc,float(per.min())

def main():
    torch.manual_seed(SEED); torch.set_num_threads(max(1,min(8,torch.get_num_threads())))
    train=torch.load(ROOT/'train.pt',map_location='cpu',weights_only=False)['indices'].long().clamp(0,47)
    val=torch.load(ROOT/'val.pt',map_location='cpu',weights_only=False)['indices'].long().clamp(0,47)
    counts=torch.bincount(train.flatten(),minlength=48).float(); weights=(counts.sum()/counts.clamp_min(1)).sqrt(); weights[0]=0.12; weights=weights/weights.mean().clamp_min(1e-6)
    model=VQVAEModel(num_classes=48,num_embeddings=64,embedding_dim=32); init=ROOT/'codec_runs_v4/codec_epoch_020.pt'; model.load_state_dict(torch.load(init,map_location='cpu'))
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3); hist=[];start=time.time();best=None
    for ep in range(1,EPOCHS+1):
        model.train();order=torch.randperm(len(train));total=0
        for i in range(0,len(order),BATCH):
            b=train[order[i:i+BATCH]];logits,vql,_=model(b);loss=F.cross_entropy(logits,b,weight=weights)+vql
            opt.zero_grad(set_to_none=True);loss.backward();opt.step();total+=float(loss.detach())*len(b)
        model.eval();
        with torch.no_grad():logits,_,codes=model(val);pred=logits.argmax(1);pix,iou,fgacc,worst=score(pred,val);used=int(torch.unique(codes).numel())
        row={'epoch':ep,'train_loss':total/len(train),'pixel_accuracy':pix,'foreground_iou':iou,'foreground_accuracy':fgacc,'worst_sample_accuracy':worst,'codebook_used':used};hist.append(row);print(json.dumps(row),flush=True)
        path=OUT/f'codec_epoch_{ep:03d}.pt';torch.save(model.state_dict(),path); sc=(fgacc,worst,iou)
        if best is None or sc>(best['score'][0],best['score'][1],best['score'][2]):best={'epoch':ep,'score':sc,'path':str(path)}
    state=torch.load(best['path'],map_location='cpu');model.load_state_dict(state);model.eval();
    with torch.no_grad():logits,_,_=model(val[:16]);pred=logits.argmax(1)
    pal=torch.load('/home/ubuntu/nano_arcade/clean_hq/clean_hq_dataset.pt',map_location='cpu',weights_only=False)['palette'][:,:3].numpy();grid=Image.new('RGB',(8*96,2*96),(35,35,35))
    for j in range(8):
        for row,a in [(0,val[j].numpy()),(1,pred[j].numpy())]:grid.paste(Image.fromarray(pal[np.asarray(a).clip(0,47)].astype(np.uint8)).resize((96,96),Image.Resampling.NEAREST),(j*96,row*96))
    gp=OUT/'codec_best_grid.png';grid.save(gp);report={'init':str(init),'weights':weights.tolist(),'history':hist,'best':best,'grid':str(gp),'elapsed_seconds':time.time()-start,'accepted':bool(best['score'][0]>=.80 and best['score'][1]>=.80 and best['score'][2]>=.85)};json.dump(report,open(OUT/'report.json','w'),indent=2);print(json.dumps({'best':best,'accepted':report['accepted'],'grid':str(gp)},indent=2))
if __name__=='__main__':main()
