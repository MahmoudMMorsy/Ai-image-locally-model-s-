#!/usr/bin/env python3
from pathlib import Path
import json, time, hashlib
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE=Path('/home/ubuntu/nano_arcade/user_drop'); RAW=BASE/'raw'; OUT=Path('/home/ubuntu/nano_arcade/combined_all64/rgb_codec_v1'); OUT.mkdir(parents=True,exist_ok=True)
SEED=20261113; EPOCHS=8; BATCH=64

class RGBCodec(nn.Module):
    def __init__(self, latent=128):
        super().__init__(); self.latent=latent
        self.enc=nn.Sequential(nn.Conv2d(4,32,4,2,1),nn.ReLU(),nn.Conv2d(32,64,4,2,1),nn.ReLU(),nn.Conv2d(64,128,4,2,1),nn.ReLU(),nn.Conv2d(128,192,4,2,1),nn.ReLU())
        self.fc=nn.Linear(192*4*4,latent); self.up=nn.Linear(latent,192*4*4)
        self.dec=nn.Sequential(nn.ConvTranspose2d(192,128,4,2,1),nn.ReLU(),nn.ConvTranspose2d(128,64,4,2,1),nn.ReLU(),nn.ConvTranspose2d(64,32,4,2,1),nn.ReLU(),nn.ConvTranspose2d(32,4,4,2,1),nn.Sigmoid())
    def encode(self,x): return self.fc(self.enc(x).flatten(1))
    def decode(self,z): return self.dec(self.up(z).view(-1,192,4,4))
    def forward(self,x): return self.decode(self.encode(x))

def fit64(im):
 im=im.convert('RGBA');w,h=im.size; scale=min(64/w,64/h);nw=max(1,round(w*scale));nh=max(1,round(h*scale));res=Image.Resampling.NEAREST if max(w,h)<=128 else Image.Resampling.LANCZOS;im=im.resize((nw,nh),res);c=Image.new('RGBA',(64,64),(0,0,0,0));c.alpha_composite(im,((64-nw)//2,(64-nh)//2));return np.asarray(c).astype('float32')/255

def load_all():
 rows=json.load(open(BASE/'raw_manifest.json'))['records']; arr=[];hs=[]
 for r in rows:
  try:
   a=fit64(Image.open(RAW/r['path']));h=hashlib.sha256(a.tobytes()).hexdigest()
   if h in hs:continue
   hs.append(h);arr.append(a.transpose(2,0,1))
  except:pass
 x=torch.from_numpy(np.stack(arr)).float();rng=np.random.default_rng(SEED);o=rng.permutation(len(x));c1=int(.8*len(x));c2=int(.9*len(x));return x[o[:c1]],x[o[c1:c2]],x[o[c2:]]

def metrics(pred,x):
 l1=float((pred-x).abs().mean()); mse=float((pred-x).pow(2).mean()); alpha=x[:,3:]>0.03; alpha_acc=float(((pred[:,3:]>0.03)==alpha).float().mean()); return l1,mse,alpha_acc

def main():
 torch.manual_seed(SEED);torch.set_num_threads(max(1,min(8,torch.get_num_threads())));tr,va,te=load_all();print({'train':len(tr),'val':len(va),'test':len(te)},flush=True)
 m=RGBCodec();opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4);hist=[];best=None;start=time.time()
 for ep in range(1,EPOCHS+1):
  m.train();o=torch.randperm(len(tr));tot=0
  for i in range(0,len(o),BATCH):
   x=tr[o[i:i+BATCH]];y=m(x);loss=F.l1_loss(y,x)+0.25*F.mse_loss(y,x);opt.zero_grad(set_to_none=True);loss.backward();opt.step();tot+=float(loss.detach())*len(x)
  m.eval();
  with torch.no_grad(): l1,mse,aa=metrics(m(va),va)
  row={'epoch':ep,'train_loss':tot/len(tr),'val_l1':l1,'val_mse':mse,'alpha_accuracy':aa};hist.append(row);print(json.dumps(row),flush=True);path=OUT/f'rgb_codec_epoch_{ep:03d}.pt';torch.save({'model':m.state_dict(),'epoch':ep,'arch':'RGBCodec','latent':128},path)
  if best is None or l1<best['val_l1']:best={'epoch':ep,'val_l1':l1,'path':str(path)}
 m.load_state_dict(torch.load(best['path'],map_location='cpu')['model']);m.eval();
 with torch.no_grad(): pred=m(va[:16]).clamp(0,1)
 grid=np.concatenate([va[:16].numpy(),pred.numpy()],axis=0).transpose(0,2,3,1); canvas=np.zeros((128,8*64,4),dtype=np.float32)
 for j in range(8):canvas[:64,j*64:(j+1)*64]=grid[j];canvas[64:,j*64:(j+1)*64]=grid[16+j]
 Image.fromarray((canvas[...,:3]*255).astype('uint8')).resize((8*96,2*96),Image.Resampling.NEAREST).save(OUT/'rgb_codec_grid.png')
 report={'arch':'RGBCodec','history':hist,'best':best,'grid':str(OUT/'rgb_codec_grid.png'),'elapsed_seconds':time.time()-start,'accepted':bool(best['val_l1']<.12)};json.dump(report,open(OUT/'report.json','w'),indent=2);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
