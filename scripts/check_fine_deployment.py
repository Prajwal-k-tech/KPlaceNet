"""Verify the compressed deployment bank, and gate reviewed new imagery on validation."""
import sys,json,math
from pathlib import Path
import torch,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.evaluate_l8_fine_retrieval import location_metrics
from src.eval import haversine_km_vec

def main():
 torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
 bank=torch.load(ROOT/'checkpoints/fine_retrieval/bank.pt',weights_only=True)
 source=ROOT/'checkpoints/l6_kerala_head_regularized';dino=ROOT/'checkpoints/l9_dinov2'
 def combine(a,b):return torch.cat((torch.nn.functional.normalize(a.float(),dim=1),torch.nn.functional.normalize(b.float(),dim=1)),1)/math.sqrt(2)
 val=torch.load(source/'features_val.pt',weights_only=False);dv=torch.load(dino/'features_val.pt',weights_only=False)
 x=torch.nn.functional.normalize(bank['x'].float(),dim=1).to(device);xy=bank['coords'].numpy()
 def pred(q,b=x,coords=xy):
  ids=torch.cat([(z.to(device)@b.T).argmax(1).cpu() for z in q.split(128)]).numpy();return coords[ids]
 vq=combine(val['x'],dv['x']);vm,_=location_metrics(pred(vq),val['coords'])
 # Review restricted to the contact-sheet images actually inspected. Menus, food and interiors excluded.
 manifest=pd.read_csv(ROOT/'data/panoramax_kerala/encoded_manifest.csv');new=torch.load(ROOT/'data/panoramax_kerala/features.pt',weights_only=False)
 keep=list(range(28))+[48,50,53,54,55,58,59,60,61,62]+list(range(64,144))
 test=torch.load(source/'features_test.pt',weights_only=False)
 held=np.concatenate([val['coords'],test['coords']]);retained=[]
 for i in keep:
  p=new['coords'][i]
  if np.min(haversine_km_vec(held[:,0],held[:,1],p[0],p[1]))<1:continue
  if retained:
   points=np.asarray(new['coords'])[retained]
   if np.min(haversine_km_vec(points[:,0],points[:,1],p[0],p[1]))<.1:continue
  retained.append(i)
 nx=combine(new['resnet_x'][retained],new['dino_x'][retained]);aug=torch.cat([x,nx.to(device)])
 axy=np.concatenate([xy,np.asarray(new['coords'])[retained]])
 am,_=location_metrics(pred(vq,aug,axy),val['coords'])
 result={'compact_bank_validation':vm,'new_data_protocol':'Manual outdoor review of first 144; exclude within 1km of existing validation/test; thin new reference locations within100m. Frozen encoders; candidate bank expansion only. No model fitting on test.', 'reviewed_candidates':len(keep),'new_references_after_filters':len(retained),'augmented_validation':am,'promotion_eligible':am['within_5km']>=vm['within_5km']+.5 and am['within_25km']>=vm['within_25km'] and am['median_km']<=vm['median_km'],'new_data_used_in_deployed_bank':False}
 # Compression verification of the already frozen winner; no model selection on test.
 model=torch.hub.load(str(ROOT/'checkpoints/fine_retrieval/dinov2_source'),'dinov2_vits14',source='local',pretrained=False).to(device).eval()
 model.load_state_dict(torch.load(ROOT/'checkpoints/fine_retrieval/dinov2_vits14_pretrain.pth',weights_only=True))
 from src.dataset import GeoDataset
 ds=GeoDataset(ROOT/'data/kerala_osv5m/kerala_test.csv',train=False);features=[]
 with torch.inference_mode():
  for images,_ in torch.utils.data.DataLoader(ds,batch_size=64,num_workers=2):
   with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device=='cuda'):z=model(images.to(device))
   features.append(z.float().cpu())
 assert ds.df.osv_id.astype(str).tolist()==test['ids']
 tq=combine(test['x'],torch.cat(features));tp=pred(tq);tm,errors=location_metrics(tp,test['coords']);result['compact_bank_test']=tm
 out=ROOT/'docs/results/fine_deployment_20261021';out.mkdir(parents=True,exist_ok=True)
 (out/'verification.json').write_text(json.dumps(result,indent=2));manifest.iloc[retained].to_csv(out/'reviewed_new_reference_candidates.csv',index=False)
 pd.DataFrame(dict(osv_id=test['ids'],pred_lat=tp[:,0],pred_lon=tp[:,1],error_km=errors)).to_csv(out/'compact_test_predictions.csv',index=False)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
