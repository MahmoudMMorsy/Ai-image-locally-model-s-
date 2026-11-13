#!/usr/bin/env python3
from pathlib import Path
import json,time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from models.train_unet_palette import PaletteUNet
from models.train_gen_decoder import LatentDecoder
ROOT=Path('/home/ubuntu/nano_arcade/clean_hq/strict_v2');OUT=ROOT/'codec_runs_v6';OUT.mkdir(exist_ok=True)
EPOCHS=25;BATCH=64;SEED=20261107

def metric(pred,t):
 fg=t!=0;pf=pred!=0; inter=(fg&pf).sum().item();union=(fg|pf).sum().item(); per=(pred==t).float().flatten(1).mean(1)
 return float((pred==t).float().mean()),inter/max(1,union),float((pred[fg]==t[fg]).float().mean()),float(per.min())

def main():
 torch.manual_seed(SEED);torch.set_num_threads(max(1,min(8,torch.get_num_threads())))
 tr=torch.load(ROOT/'train.pt',map_location='cpu',weights_only=False)['indices'].long().clamp(0,47);va=torch.load(ROOT/'val.pt',map_location='cpu',weights_only=False)['indices'].long().clamp(0,47)
 enc=PaletteUNet(48,128,48);dec=LatentDecoder(48,128,48);opt=torch.optim.AdamW(list(enc.parameters())+list(dec.parameters()),lr=1e-3,weight_decay=1e-3);w=torch.ones(48);w[0]=.15;hist=[];best=None;start=time.time()
 for ep in range(1,EPOCHS+1):
  enc.train();dec.train();order=torch.randperm(len(tr));tot=0
  for i in range(0,len(order),BATCH):
   b=tr[order[i:i+BATCH]];z=enc.encode(b);logits=dec(z);loss=F.cross_entropy(logits,b,weight=w);opt.zero_grad(set_to_none=True);loss.backward();opt.step();tot+=float(loss.detach())*len(b)
  enc.eval();dec.eval();
  with torch.no_grad(): z=enc.encode(va);pred=dec(z).argmax(1);pix,iou,fgacc,worst=metric(pred,va)
  row={'epoch':ep,'train_loss':tot/len(tr),'pixel_accuracy':pix,'foreground_iou':iou,'foreground_accuracy':fgacc,'worst_sample_accuracy':worst};hist.append(row);print(json.dumps(row),flush=True)
  path=OUT/f'codec_epoch_{ep:03d}.pt';torch.save({'encoder':enc.state_dict(),'decoder':dec.state_dict(),'n_colors':48,'latent':128,'base':48,'epoch':ep},path);sc=(fgacc,worst,iou)
  if best is None or sc>(best['score'][0],best['score'][1],best['score'][2]):best={'epoch':ep,'score':sc,'path':str(path)}
 state=torch.load(best['path'],map_location='cpu');enc.load_state_dict(state['encoder']);dec.load_state_dict(state['decoder']);enc.eval();dec.eval();
 with torch.no_grad():pred=dec(enc.encode(va[:16])).argmax(1)
 pal=torch.load('/home/ubuntu/nano_arcade/clean_hq/clean_hq_dataset.pt',map_location='cpu',weights_only=False)['palette'][:,:3].numpy();grid=Image.new('RGB',(8*96,2*96),(35,35,35))
 for j in range(8):
  for row,a in [(0,va[j].numpy()),(1,pred[j].numpy())]:grid.paste(Image.fromarray(pal[np.asarray(a).clip(0,47)].astype(np.uint8)).resize((96,96),Image.Resampling.NEAREST),(j*96,row*96))
 gp=OUT/'codec_best_grid.png';grid.save(gp);report={'architecture':'PaletteUNet encoder + latent-only decoder','history':hist,'best':best,'grid':str(gp),'elapsed_seconds':time.time()-start,'accepted':bool(best['score'][0]>=.80 and best['score'][1]>=.80 and best['score'][2]>=.85)};json.dump(report,open(OUT/'report.json','w'),indent=2);print(json.dumps({'best':best,'accepted':report['accepted'],'grid':str(gp)},indent=2))
if __name__=='__main__':main()
