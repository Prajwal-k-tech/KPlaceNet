"""Download one small batch, encode, discard image bytes; retain licensed provenance/features.

Metadata is candidate input, not an independently verified training dataset.
No generated captions or EXIF GPS enter the encoders.
"""
import sys,json,time,io,hashlib,collections
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests,numpy as np,pandas as pd,torch
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.model import build_model
from src.dataset import get_transforms

def main():
 torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
 source=ROOT.parent.parent/'work/research-20261021/panoramax-first1000.json';out=ROOT/'data/panoramax_kerala';out.mkdir(exist_ok=True);features=json.loads(source.read_text())['features']
 records=[]
 for f in features:
  props=f['properties'];lon,lat=f['geometry']['coordinates'][:2]
  if props.get('license')!='CC-BY-SA-4.0' or not (8<=lat<=12.8 and 74.5<=lon<=77.5):continue
  records.append(dict(id=f['id'],latitude=lat,longitude=lon,collection=f.get('collection',''),datetime=props.get('datetime',''),license=props.get('license'),author=props.get('exif',{}).get('Exif.Image.Artist',''),source_url=next((l['href'] for l in f.get('links',[]) if l.get('rel')=='self'),''),image_url=f['assets']['thumb']['href'],horizontal_accuracy_m=props.get('quality:horizontal_accuracy'),review='candidate_source_coordinates_not_independently_verified'))
 pd.DataFrame(records).to_csv(out/'source_manifest.csv',index=False)
 model=build_model(num_cells=300,pretrained=False,freeze_backbone=False);ckpt=torch.load(ROOT/'checkpoints/l6_kerala_geographic/selected.pt',weights_only=True,map_location='cpu');model.load_state_dict(ckpt['model_state']);model.to(device).eval()
 dino=torch.hub.load('facebookresearch/dinov2:main','dinov2_vits14',pretrained=True,trust_repo=True,skip_validation=True).to(device).eval()
 tfm=get_transforms(train=False);xs=[];ds=[];saved=[];failures=[];start=time.monotonic();seen=set();thumbs=[]
 def fetch(row):
  try:
   r=requests.get(row['image_url'],timeout=20);r.raise_for_status();sha=hashlib.sha256(r.content).hexdigest();im=Image.open(io.BytesIO(r.content)).convert('RGB');im.load()
   if min(im.size)<100:return row,None,None,'too small'
   row={**row,'image_sha256':sha,'width':im.width,'height':im.height,'average_hash':hex(int(''.join('1' if v>np.asarray(im.resize((8,8)).convert('L')).mean() else '0' for v in np.asarray(im.resize((8,8)).convert('L')).flatten()),2))}
   return row,tfm(im),im.copy(),None
  except Exception as e:return row,None,None,str(e)
 with ThreadPoolExecutor(max_workers=4) as pool,torch.inference_mode():
  for start_idx in range(0,len(records),32):
   batch=[];batch_rows=[]
   for row,x,im,error in pool.map(fetch,records[start_idx:start_idx+32]):
    if error:failures.append(dict(id=row['id'],error=error));continue
    if row['image_sha256'] in seen:continue
    seen.add(row['image_sha256']);batch.append(x);batch_rows.append(row)
    if len(thumbs)<144:
     thumb=ImageOps.contain(im,(160,120));thumbs.append((thumb,row['id'],len(saved)+len(batch_rows)-1))
   if not batch:continue
   images=torch.stack(batch).to(device)
   with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device=='cuda'):
    rx=model.backbone(images).flatten(1);dx=dino(images)
   xs.append(rx.float().cpu());ds.append(dx.float().cpu());saved.extend(batch_rows)
   # Only compact tensors/manifests are checkpointed. Full photo bytes have no disk copy.
   torch.save(dict(resnet_x=torch.cat(xs),dino_x=torch.cat(ds),coords=np.array([[r['latitude'],r['longitude']] for r in saved]),ids=[r['id'] for r in saved]),out/'features.pt')
   pd.DataFrame(saved).to_csv(out/'encoded_manifest.csv',index=False)
   print('encoded',len(saved),'/',len(records),'elapsed',round(time.monotonic()-start,1),'seconds',flush=True)
 for block in range(0,len(thumbs),36):
  sheet=Image.new('RGB',(960,864),'white');draw=ImageDraw.Draw(sheet)
  for j,(im,id,index) in enumerate(thumbs[block:block+36]):
   x=j%6*160;y=j//6*144;sheet.paste(im,(x,y));draw.text((x+3,y+121),str(index)+' '+id[:8],fill='black')
  sheet.save(out/f'review-{block//36+1}.jpg',quality=90)
 (out/'collection_report.json').write_text(json.dumps(dict(source_candidates=len(records),encoded=len(saved),failed=failures,elapsed_seconds=time.monotonic()-start,stored_full_images=0,purpose='Candidate feature bank; split and source-quality audit required before any use'),indent=2))
 print('DONE',len(saved),len(failures),flush=True)
if __name__=='__main__':main()
