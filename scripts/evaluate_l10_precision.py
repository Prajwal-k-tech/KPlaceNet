"""Precision-first validation selection for the user's finer-location objective."""
import sys,json,math
from pathlib import Path
import torch,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.fine_retrieval import fine_predict
from src.model import build_model
from scripts.evaluate_l8_fine_retrieval import location_metrics

def main():
 torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';out=ROOT/'checkpoints/l10_precision';out.mkdir(exist_ok=True)
 recipes=[dict(name=f'resnet{w}_k{k}',resnet_weight=w,k=k,radius_km=25.,prior_weight=0.) for w in [1.,.75,.5] for k in [1,5]]
 protocol=dict(primary_objective='within5km, then median; latest requested fine-location precision rather than coarse25km',recipes=recipes,baseline='L8 local20 classifier-prior',guardrail='validation within25 no worse than L8 by >2pp',promotion='within5 improves >=.5pp; median <=1.02 baseline',selection='Validation only. Reused official test only after freeze; no fresh confirmation claimed.',storage='cached 384-dim DINO and 2048-dim ResNet; no duplicated source photos')
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 source=ROOT/'checkpoints/l6_kerala_head_regularized';dino=ROOT/'checkpoints/l9_dinov2'
 train=torch.load(source/'features_train.pt',weights_only=False);val=torch.load(source/'features_val.pt',weights_only=False);dt=torch.load(dino/'features_train.pt',weights_only=False);dv=torch.load(dino/'features_val.pt',weights_only=False)
 assert train['ids']==dt['ids'] and val['ids']==dv['ids'] and not set(train['ids'])&set(val['ids'])
 xy=torch.tensor(train['coords'],dtype=torch.float32,device=device)
 baseline=json.loads((ROOT/'checkpoints/l8_fine_retrieval/results.json').read_text());bm=baseline['selection']['selected_validation'];best=bm;chosen=None;scores={}
 def combined(a,b,w):return torch.cat((torch.nn.functional.normalize(a,dim=1)*math.sqrt(w),torch.nn.functional.normalize(b,dim=1)*math.sqrt(1-w)),dim=1)
 def predict(data,dd,recipe):
  w=recipe['resnet_weight'];bank=combined(train['x'].to(device),dt['x'].to(device),w);query=combined(data['x'].to(device),dd['x'].to(device),w)
  args={k:recipe[k] for k in ['k','radius_km','prior_weight']}
  with torch.inference_mode():return torch.cat([fine_predict(q,bank,xy,**args)['coords'].cpu() for q in query.split(128)]).numpy()
 for recipe in recipes:
  m,_=location_metrics(predict(val,dv,recipe),val['coords']);scores[recipe['name']]=m;print(recipe['name'],m,flush=True)
  key=lambda x:(x['within_5km'],-x['median_km'],x['within_25km'])
  if m['within_25km']>=bm['within_25km']-2 and key(m)>key(best):best=m;chosen=recipe
 gate=chosen is not None and best['within_5km']>=bm['within_5km']+.5 and best['median_km']<=bm['median_km']*1.02
 selection=dict(recipe=chosen,promotion_eligible=gate,baseline_validation=bm,selected_validation=best,all_validation=scores)
 (out/'selection_before_test.json').write_text(json.dumps(selection,indent=2));result=dict(protocol=protocol,selection=selection)
 if gate:
  test=torch.load(source/'features_test.pt',weights_only=False)
  if chosen['resnet_weight']==1.:dd={'x':torch.zeros((len(test['x']),384))}
  else:
   import scripts.evaluate_l9_dinov2
   # Only selected mixed encoder needs test DINO features; never select on them.
   model=torch.hub.load('facebookresearch/dinov2:main','dinov2_vits14',pretrained=True,trust_repo=True,skip_validation=True).to(device).eval()
   from src.dataset import GeoDataset
   ds=GeoDataset(ROOT/'data/kerala_osv5m/kerala_test.csv',train=False);xs=[]
   with torch.inference_mode():
    for images,_ in torch.utils.data.DataLoader(ds,batch_size=64,num_workers=2):
     with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device=='cuda'):x=model(images.to(device))
     xs.append(x.float().cpu())
   dd={'x':torch.cat(xs)};assert ds.df.osv_id.astype(str).tolist()==test['ids']
  pred=predict(test,dd,chosen);m,error=location_metrics(pred,test['coords']);result['selected_test']=m
  pd.DataFrame(dict(osv_id=test['ids'],lat=test['coords'][:,0],lon=test['coords'][:,1],pred_lat=pred[:,0],pred_lon=pred[:,1],error_km=error)).to_csv(out/'test_predictions.csv',index=False)
 (out/'results.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps(result),flush=True)
if __name__=='__main__':main()
