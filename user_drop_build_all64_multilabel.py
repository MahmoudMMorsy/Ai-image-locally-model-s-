#!/usr/bin/env python3
from pathlib import Path
from PIL import Image
import json, hashlib, re, unicodedata, collections
import numpy as np, torch

BASE=Path('/home/ubuntu/nano_arcade/user_drop'); RAW=BASE/'raw'; OUT=BASE/'all64'; OUT.mkdir(exist_ok=True)
PAL=torch.load('/home/ubuntu/nano_arcade/clean_hq/runtime/checkpoints/palette48_full.pt',map_location='cpu',weights_only=False).float().numpy()
RAW_MAN=json.load(open(BASE/'raw_manifest.json'))

KEYWORDS={
 'donkey':['donkey','حمار','حمير'], 'horse':['horse','حصان','خيول'], 'cow':['cow','بقرة','ثور'], 'cat':['cat','قطة','قط'], 'dog':['dog','كلب'],
 'monster':['monster','وحش','مخلوق'], 'ghost':['ghost','شبح','اشباح','أشباح'], 'farmer':['farmer','فلاح','فلاحة','صعيدي','صعيديه','صعيدية'],
 'female':['female','woman','girl','انثى','أنثى','امرأة','بنت','فتاة'], 'male':['male','man','boy','رجل','ولد','ذكر'],
 'warrior':['warrior','knight','محارب','فارس'], 'robot':['robot','روبوت'], 'zombie':['zombie','زومبي'], 'dragon':['dragon','تنين'],
 'vehicle':['car','truck','vehicle','سيارة','مركبة'], 'building':['building','house','castle','منزل','قلعة'], 'landscape':['landscape','background','منظر','خلفية'],
 'text_ui':['text','logo','font','النص','كتابة','شعار','واجهة'], 'sprite_sheet':['sheet','spritesheet','sheet','شيت'],
}

def norm(s):
 s=unicodedata.normalize('NFKC',s).lower(); return re.sub(r'[_\-]+',' ',s)
def tags_from_name(name):
 t=norm(name); hits=[k for k,ws in KEYWORDS.items() if any(w in t for w in ws)]
 return hits

def fit64(im):
 im=im.convert('RGBA'); w,h=im.size
 resample=Image.Resampling.NEAREST if max(w,h)<=128 or 'A' in im.getbands() else Image.Resampling.LANCZOS
 scale=min(64/w,64/h); nw=max(1,round(w*scale)); nh=max(1,round(h*scale)); im=im.resize((nw,nh),resample)
 canvas=Image.new('RGBA',(64,64),(0,0,0,0)); canvas.alpha_composite(im,((64-nw)//2,(64-nh)//2)); return canvas

def quant(im):
 a=np.asarray(im); rgb=a[...,:3].astype(np.float32); d=((rgb[...,None,:]-PAL[None,None,:,:3])**2).sum(-1); idx=d.argmin(-1).astype(np.int64); idx[a[...,3]<8]=0; return idx

def style_tags(im, idx, rec):
 a=np.asarray(im); colors=len(np.unique(a.reshape(-1,4),axis=0)); opaque=float((a[...,3]>8).mean());
 out=[]
 if rec['width']<=128 and colors<=96: out.append('pixel_art_candidate')
 if colors>500 or rec['width']>256 or rec['height']>256: out.append('realistic_or_high_detail_candidate')
 if opaque<.25: out.append('transparent_asset')
 if not out: out.append('illustration_or_mixed')
 return out

records=[]; tensors=[]; hashes=set(); skipped=0
for r in RAW_MAN['records']:
 p=RAW/r['path']
 try:
  with Image.open(p) as src:
   im=fit64(src); idx=quant(im); ph=hashlib.sha256(idx.tobytes()).hexdigest()
   # Keep duplicates as records for provenance, but mark exact duplicate groups.
   name=Path(r['path']).name; hits=tags_from_name(name)
   subject=[x for x in hits if x not in {'female','male'}]
   gender=[x for x in hits if x in {'female','male'}]
   if not subject: subject=['unknown_subject']
   if not gender: gender=['unknown_gender']
   tags=style_tags(im,idx,r)
   if 'sprite_sheet' in hits: tags.append('sheet_or_multi_asset')
   rec={'id':f'user_all64_{len(records):05d}','source_path':r['path'],'source_sha256':r['sha256_pixels'],'normalized_index_sha256':ph,'original_width':r['width'],'original_height':r['height'],'original_mode':r['mode'],'has_alpha':r['has_alpha'],'style_tags':sorted(set(tags)),'subject_tags':sorted(set(subject)),'gender_tags':sorted(set(gender)),'filename_keyword_tags':sorted(set(hits)),'quality_tags':['readable','needs_semantic_review'],'classification_status':'heuristic_filename_and_geometry_only'}
   records.append(rec); tensors.append(idx)
 except Exception as e:
  skipped+=1
X=torch.from_numpy(np.stack(tensors)).long(); torch.save({'indices':X,'palette':torch.from_numpy(PAL),'records':records},OUT/'user_all64_dataset.pt')
with open(OUT/'user_all64_manifest.jsonl','w',encoding='utf-8') as f:
 for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
counts={k:collections.Counter(x for r in records for x in r[k]) for k in ('style_tags','subject_tags','gender_tags')}
json.dump({'version':'nano_arcade_user_all64_multilabel_v1','source_raw_count':len(RAW_MAN['records']),'converted_count':len(records),'skipped_decode_errors':skipped,'shape':list(X.shape),'classification_note':'All images are retained. Semantic tags are conservative heuristics from filename/geometry and must be reviewed before claiming class accuracy.','counts':{k:dict(v) for k,v in counts.items()}},open(OUT/'dataset_summary.json','w'),ensure_ascii=False,indent=2)
print(json.dumps({'converted_count':len(records),'skipped':skipped,'shape':list(X.shape),'styles':dict(counts['style_tags']),'subjects_top':counts['subject_tags'].most_common(20),'genders':dict(counts['gender_tags'])},ensure_ascii=False,indent=2))
