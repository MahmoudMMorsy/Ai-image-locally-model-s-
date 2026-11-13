#!/usr/bin/env python3
from pathlib import Path
import json,time,hashlib
import numpy as np
from PIL import Image
import torch,torch.nn as nn,torch.nn.functional as F
BASE=Path('/home/ubuntu/nano_arcade/user_drop');RAW=BASE/'raw';OUT=Path('/home/ubuntu/nano_arcade/combined_all64/rgb_codec_v2');OUT.mkdir(parents=True,exist_ok=True)
SEED=20261113;EPOCHS=6;BATCH=64
class RGBCodecV2(nn.Module):
 def __init__(self,latent=512):
  super().__init__();self.enc=nn.Sequential(nn.Conv2d(4,48,4,2,1),nn.ReLU(),nn.Conv2d(48,96,4,2,1),nn.ReLU(),nn.Conv2d(96,192,4,2,1),nn.ReLU(),nn.Conv2d(192,256,4,2,1),nn.ReLU());self.fc=nn.Linear(256*4*4,latent);self.up=nn.Linear(latent,256*4*4);self.dec=nn.Sequential(nn.ConvTranspose2d(256,192,4,2,1),nn.ReLU(),nn.ConvTranspose2d(192,96,4,2,1),nn.ReLU(),nn.ConvTranspose2d(96,48,4,2,1),nn.ReLU(),nn.ConvTranspose2d(48,4,4,2,1),nn.Sigmoid())
 def encode(self,x):return self.fc(self.enc(x).flatten(1))
 def decode(self,z):return self.dec(self.up(z).view(-1,256,4,4))
 def forward(self,x):return self.decode(self.encode(x))
def fit64(im):
 im=im.convert('RGBA');w,h=im.size;s=min(64/w,64/h);nw=max(1,round(w*s));nh=max(1,round(h*s));res=Image.Resampling.NEAREST if max(w,h)<=128 else Image.Resampling.LANCZOS;im=im.resize((nw,nh),res);c=Image.new('RGBA',(64,64),(0,0,0,0));c.alpha_composite(im,((64-nw)//2,(64-nh)//2));return np.asarray(c).astype('float32').transpose(2,0,1)/255
def load():
 rows=json.load(open(BASE/'raw_manifest.json'))['records'];a=[];seen=set()
 for r in rows:
  try:
   x=fit64(Image.open(RAW/r['path']));h=hashlib.sha256(x.tobytes()).hexdigest()
   if h not in seen:seen.add(h);a.append(x)
  except:pass
 x=torch.from_numpy(np.stack(a));g=np.random.default_rng(SEED);o=g.permutation(len(x));c1=int(.8*len(x));c2=int(.9*len(x));return x[o[:c1]],x[o[c1:c2]],x[o[c2:]]
def edge(x):return torch.abs(x[:,:,1:,:]-x[:,:,:-1,:]).mean()+torch.abs(x[:,:,:,1:]-x[:,:,:,:-1]).mean()
def main():
 torch.manual_seed(SEED);torch.set_num_threads(max(1,min(8,torch.get_num_threads())));tr,va,te=load();m=RGBCodecV2();opt=torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=1e-4);hist=[];best=None;start=time.time();print({'train':len(tr),'val':len(va),'test':len(te)},flush=True)
 for ep in range(1,EPOCHS+1):
  m.train();o=torch.randperm(len(tr));tot=0
  for i in range(0,len(o),BATCH):
   x=tr[o[i:i+BATCH]];y=m(x);loss=F.l1_loss(y,x)+0.35*F.mse_loss(y,x)+0.5*torch.abs(edge(y)-edge(x));opt.zero_grad(set_to_none=True);loss.backward();opt.step();tot+=float(loss.detach())*len(x)
  m.eval();
  with torch.no_grad():y=m(va);l1=float((y-va).abs().mean());mse=float((y-va).pow(2).mean());ed=float(torch.abs(edge(y)-edge(va)));aa=float((((y[:,3:]>0.03)==(va[:,3:]>0.03)).float().mean()))
  row={'epoch':ep,'train_loss':tot/len(tr),'val_l1':l1,'val_mse':mse,'edge_error':ed,'alpha_accuracy':aa};hist.append(row);print(json.dumps(row),flush=True);p=OUT/f'rgb_codec_epoch_{ep:03d}.pt';torch.save({'model':m.state_dict(),'epoch':ep,'arch':'RGBCodecV2','latent':512},p)
  if best is None or (l1+ed)<(best['val_l1']+best['edge_error']):best={'epoch':ep,'val_l1':l1,'edge_error':ed,'path':str(p)}
 m.load_state_dict(torch.load(best['path'],map_location='cpu')['model']);m.eval();
 with torch.no_grad():y=m(va[:16]).clamp(0,1)
 grid=np.concatenate([va[:16].numpy(),y.numpy()]).transpose(0,2,3,1);can=np.zeros((128,8*64,4),dtype=np.float32)
 for j in range(8):can[:64,j*64:(j+1)*64]=grid[j];can[64:,j*64:(j+1)*64]=grid[16+j]
 Image.fromarray((can[...,:3]*255).astype('uint8')).resize((8*96,2*96),Image.Resampling.NEAREST).save(OUT/'rgb_codec_grid.png');rep={'arch':'RGBCodecV2','history':hist,'best':best,'grid':str(OUT/'rgb_codec_grid.png'),'elapsed_seconds':time.time()-start,'accepted':False};json.dump(rep,open(OUT/'report.json','w'),indent=2);print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
