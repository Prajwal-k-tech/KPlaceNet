"""Evaluate the validation-selected fixed candidate once on the reused official test."""
import sys,json,math
from pathlib import Path
import torch,numpy as np,pandas as pd
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.geographic_prototypes import rerank_references
from scripts.evaluate_l8_fine_retrieval import location_metrics

def main():
 torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';out=ROOT/'checkpoints/geographic_prototypes'
 selection=json.loads((out/'selection_before_test.json').read_text());assert selection['promotion_eligible']
 recipe=selection['selected_recipe'];assert recipe['kind']=='prototype_reranked_reference'
 protocol=dict(recipe=recipe,decision='Frozen before evaluation. No test tuning. Deploy only if within5/25 lose no more than.5pp and median <=1.02x current.',test_caveat='Already reused official benchmark; exploratory follow-up, not independent confirmation',selection_amendment=selection['protocol']['selection_amendment'])
 (out/'test_protocol.json').write_text(json.dumps(protocol,indent=2))
 source=ROOT/'checkpoints/l6_kerala_head_regularized';dino=ROOT/'checkpoints/l9_dinov2'
 train=torch.load(source/'features_train.pt',weights_only=False);test=torch.load(source/'features_test.pt',weights_only=False);dt=torch.load(dino/'features_train.pt',weights_only=False)
 from src.dataset import GeoDataset
 ds=GeoDataset(ROOT/'data/kerala_osv5m/kerala_test.csv',train=False)
 assert ds.df.osv_id.astype(str).tolist()==test['ids'] and not set(test['ids'])&set(train['ids'])
 model=torch.hub.load(str(ROOT/'checkpoints/fine_retrieval/dinov2_source'),'dinov2_vits14',source='local',pretrained=False).to(device).eval();model.load_state_dict(torch.load(ROOT/'checkpoints/fine_retrieval/dinov2_vits14_pretrain.pth',weights_only=True));xs=[]
 with torch.inference_mode():
  for images,_ in torch.utils.data.DataLoader(ds,batch_size=64,num_workers=2):
   with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device=='cuda'):z=model(images.to(device))
   xs.append(z.float().cpu())
 torch.save(dict(x=torch.cat(xs),ids=test['ids'],coords=test['coords']),dino/'features_test.pt')
 def combine(a,b):return torch.cat((F.normalize(a.float(),dim=1),F.normalize(b.float(),dim=1)),1)/math.sqrt(2)
 bank=combine(train['x'],dt['x']).to(device);q=combine(test['x'],torch.cat(xs)).to(device)
 saved=torch.load(out/'selected.pt',weights_only=False);grid=saved['grid'];p=F.normalize(saved['prototypes'].to(device),dim=1)
 idx=grid['indices'].to(device);weights=grid['weights'].to(device)
 codes=torch.cat([F.normalize((p[ii]*ww[:,:,None]).sum(1),dim=1) for ii,ww in zip(idx.split(512),weights.split(512))])
 with torch.inference_mode():
  chosen=torch.cat([rerank_references(z,bank,codes,recipe['rerank_weight'],50).cpu() for z in q.split(128)]).numpy()
  baseline=(q@bank.T).argmax(1).cpu().numpy()
 xy=np.asarray(train['coords']);pred=xy[chosen];metrics,err=location_metrics(pred,test['coords']);bm,be=location_metrics(xy[baseline],test['coords'])
 gate=metrics['within_5km']>=bm['within_5km']-.5 and metrics['within_25km']>=bm['within_25km']-.5 and metrics['median_km']<=bm['median_km']*1.02
 result=dict(protocol=protocol,baseline_test=bm,selected_test=metrics,deployment_eligible=gate,queries_with_changed_reference=int((chosen!=baseline).sum()))
 pd.DataFrame(dict(osv_id=test['ids'],lat=test['coords'][:,0],lon=test['coords'][:,1],baseline_error_km=be,error_km=err,pred_lat=pred[:,0],pred_lon=pred[:,1])).to_csv(out/'test_predictions.csv',index=False)
 (out/'test_results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
