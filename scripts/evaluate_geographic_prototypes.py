"""Validation-only ground-prototype/interpolation experiment inspired by ETH.

No aerial training, new images or held-out labels used to fit prototypes.
The existing fine retrieval remains eligible and untouched on disk.
"""
import sys,json,math,time,copy
from pathlib import Path
import torch,numpy as np
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.geographic_prototypes import build_grid,local_xy,initialize_prototypes,prototype_loss
from scripts.evaluate_l8_fine_retrieval import location_metrics

def main():
 torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
 device='cuda' if torch.cuda.is_available() else 'cpu';out=ROOT/'checkpoints/geographic_prototypes';out.mkdir(exist_ok=True)
 protocol=dict(inspiration='Scaling Image Geo-Localization to Continent Level: ground prototypes, boundary interpolation, multi-similarity ground-prototype edge',differences='Frozen ResNet-DINO input; no encoder optimization, no aerial imagery/encoder or S2; local regular 2/5km grids; no full ETH reproduction',recipes={'spacing_km':[2.,5.],'epochs':[0,4,8],'rerank_weight':[0.,.1,.25]},training={'seed':42,'epochs':8,'optimizer':'AdamW','lr':.003,'weight_decay':0.,'batch':256,'positive':'bilinear normalized prototype mixture','negative':'nodes farther than2x spacing','loss_alpha':2.,'loss_beta':50.,'loss_margin':.5},selection='Apply all safety conditions to each candidate before ranking eligible candidates by within5, median and within25; current retrieval eligible',selection_amendment='Corrected candidate-level guardrail filtering after validation inspection and before test. Original highest-within5 candidate failed median guardrail. This is an exploratory amendment, not a pristine preregistration.',promotion='within5 improves >=.5pp; within25 no worse by >.5pp; median <=1.02x current',test='Only evaluate after promotion gate; benchmark reused, not independent confirmation',storage='reuse cached frozen descriptors; no raw image download')
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 source=ROOT/'checkpoints/l6_kerala_head_regularized';dino=ROOT/'checkpoints/l9_dinov2'
 train=torch.load(source/'features_train.pt',weights_only=False);val=torch.load(source/'features_val.pt',weights_only=False);dt=torch.load(dino/'features_train.pt',weights_only=False);dv=torch.load(dino/'features_val.pt',weights_only=False)
 assert train['ids']==dt['ids'] and val['ids']==dv['ids'] and not set(train['ids'])&set(val['ids'])
 def features(a,b):return torch.cat((F.normalize(a.float(),dim=1),F.normalize(b.float(),dim=1)),1)/math.sqrt(2)
 x=features(train['x'],dt['x']).to(device);vq=features(val['x'],dv['x']).to(device)
 xy=np.asarray(train['coords']);scores,near=(vq@x.T).topk(50,1)
 baseline_pred=xy[near[:,0].cpu().numpy()];baseline,_=location_metrics(baseline_pred,val['coords']);best=baseline;winner=None;best_state=None;records=[]
 for spacing in [2.,5.]:
  g=build_grid(xy,spacing);idx=g['indices'].to(device);weights=g['weights'].to(device)
  p=torch.nn.Parameter(initialize_prototypes(x,idx,weights,len(g['coords'])))
  opt=torch.optim.AdamW([p],lr=.003,weight_decay=0.)
  txy=torch.tensor(local_xy(xy),device=device,dtype=torch.float32);nxy=torch.tensor(g['xy'],device=device,dtype=torch.float32)
  def evaluate(epoch):
   nonlocal best,winner,best_state
   with torch.inference_mode():
    unit=F.normalize(p,dim=1);node_sim=vq@unit.T
    predictions=[('prototype_coordinate',g['coords'][node_sim.argmax(1).cpu().numpy()],0.)]
    # Learned regional appearance prior re-ranks the same50 original references.
    candidate_codes=F.normalize((unit[idx]*weights[:,:,None]).sum(1),dim=1)
    prior=(vq[:,None,:]*candidate_codes[near]).sum(2)
    for weight in [.1,.25]:
     choice=(scores+weight*prior).argmax(1);ref=near[torch.arange(len(vq),device=device),choice];predictions.append(('prototype_reranked_reference',xy[ref.cpu().numpy()],weight))
    for kind,pred,weight in predictions:
     metrics,_=location_metrics(pred,val['coords']);recipe=dict(spacing_km=spacing,epoch=epoch,kind=kind,rerank_weight=weight);records.append(dict(recipe=recipe,validation=metrics))
     print(json.dumps(records[-1]),flush=True)
     key=lambda m:(m['within_5km'],-m['median_km'],m['within_25km'])
     if metrics['within_5km']>=baseline['within_5km']+.5 and metrics['within_25km']>=baseline['within_25km']-.5 and metrics['median_km']<=baseline['median_km']*1.02 and key(metrics)>key(best):
      best=metrics;winner=recipe;best_state=dict(prototypes=p.detach().cpu().clone(),grid=g)
  evaluate(0)
  for epoch in range(1,9):
   start=time.monotonic();total=0.
   for b in torch.randperm(len(x),device=device).split(256):
    mask=torch.cdist(txy[b],nxy)>2*spacing
    opt.zero_grad();loss=prototype_loss(x[b],p,idx[b],weights[b],mask);loss.backward();opt.step();total+=float(loss.detach())
   print('epoch',spacing,epoch,'loss',total,'seconds',time.monotonic()-start,flush=True)
   if epoch in [4,8]:evaluate(epoch)
 gate=winner is not None and best['within_5km']>=baseline['within_5km']+.5 and best['within_25km']>=baseline['within_25km']-.5 and best['median_km']<=baseline['median_km']*1.02
 result=dict(protocol=protocol,baseline_validation=baseline,selected_recipe=winner,selected_validation=best,promotion_eligible=gate,all_validation=records,official_test_evaluated=False)
 (out/'selection_before_test.json').write_text(json.dumps(result,indent=2))
 if best_state:torch.save(best_state,out/'selected.pt')
 (out/'results.json').write_text(json.dumps(result,indent=2));print('FINAL',json.dumps(result),flush=True)
if __name__=='__main__':main()
