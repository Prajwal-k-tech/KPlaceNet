"""Compact frozen DINOv2 retrieval comparison. No pretrained-image-overlap guarantee."""
import sys,time,json,hashlib
from pathlib import Path
import torch,numpy as np,pandas as pd
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dataset import GeoDataset
from src.fine_retrieval import fine_predict
from scripts.evaluate_l8_fine_retrieval import location_metrics

def main():
 torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';out=ROOT/'checkpoints/l9_dinov2';out.mkdir(exist_ok=True)
 protocol=dict(backbone='official dinov2_vits14 frozen pretrained; 384-dim',input='same 256 resize/224 center crop/ImageNet normalization',recipes=[dict(name='nearest',k=1,radius_km=25.,prior_weight=0.),dict(name='local5',k=5,radius_km=25.,prior_weight=0.),dict(name='local20',k=20,radius_km=25.,prior_weight=0.)],selection='validation within25 then median; preserve L8 baseline if no improvement',test_caveat='Reused benchmark, exploratory. Pretraining-source overlap has not been audited.',storage='Features cached per split; no copied image datasets')
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 model=torch.hub.load('facebookresearch/dinov2:main','dinov2_vits14',pretrained=True,trust_repo=True,skip_validation=True).to(device).eval()
 weights=Path(torch.hub.get_dir())/'checkpoints/dinov2_vits14_pretrain.pth'
 with weights.open('rb') as f:protocol['weights_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 def features(split):
  cache=out/f'features_{split}.pt';csv=ROOT/f'data/kerala_osv5m/kerala_{split}.csv'
  with csv.open('rb') as f:csv_sha=hashlib.file_digest(f,'sha256').hexdigest()
  if cache.exists():
   value=torch.load(cache,weights_only=False)
   if value['csv_sha256']==csv_sha and value['weights_sha256']==protocol['weights_sha256']:return value
  ds=GeoDataset(csv,train=False);loader=DataLoader(ds,batch_size=64,num_workers=2,shuffle=False);xs=[];cs=[];start=time.monotonic()
  with torch.inference_mode():
   for i,(images,xy) in enumerate(loader):
    with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device=='cuda'):f=model(images.to(device))
    xs.append(f.float().cpu());cs.append(xy)
    if (i+1)%16==0:print('features',split,min((i+1)*64,len(ds)),len(ds),round(time.monotonic()-start,1),flush=True)
  value=dict(x=torch.cat(xs),coords=torch.cat(cs).numpy(),ids=ds.df.osv_id.astype(str).tolist(),csv_sha256=csv_sha,weights_sha256=protocol['weights_sha256']);torch.save(value,cache);return value
 train=features('train');val=features('val');assert not set(train['ids'])&set(val['ids'])
 bank=train['x'].to(device);xy=torch.tensor(train['coords'],dtype=torch.float32,device=device)
 def predict(data,recipe):
  args={k:v for k,v in recipe.items() if k!='name'}
  with torch.inference_mode():return torch.cat([fine_predict(x.to(device),bank,xy,**args)['coords'].cpu() for x in data['x'].split(128)]).numpy()
 baseline=json.loads((ROOT/'checkpoints/l8_fine_retrieval/results.json').read_text())['selection']['selected_validation'];best=baseline;chosen=None;all_scores={}
 key=lambda m:(m['within_25km'],-m['median_km'])
 for recipe in protocol['recipes']:
  m,_=location_metrics(predict(val,recipe),val['coords']);all_scores[recipe['name']]=m;print('validation',recipe['name'],m,flush=True)
  if key(m)>key(best):best=m;chosen=recipe
 eligible=chosen is not None and best['within_25km']>=baseline['within_25km']+.5 and best['median_km']<=baseline['median_km']*1.02
 selection=dict(recipe=chosen,promotion_eligible=eligible,baseline_validation=baseline,selected_validation=best,all_validation=all_scores)
 (out/'selection_before_test.json').write_text(json.dumps(selection,indent=2));result=dict(protocol=protocol,selection=selection,device=device)
 if eligible:
  test=features('test');assert not set(test['ids'])&(set(train['ids'])|set(val['ids']))
  pred=predict(test,chosen);m,errors=location_metrics(pred,test['coords']);result['selected_test']=m
  pd.DataFrame(dict(osv_id=test['ids'],lat=test['coords'][:,0],lon=test['coords'][:,1],pred_lat=pred[:,0],pred_lon=pred[:,1],error_km=errors)).to_csv(out/'test_predictions.csv',index=False)
 (out/'results.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps(result),flush=True)
if __name__=='__main__':main()
