"""Validation-gated fine-coordinate retrieval, with a preserved L7 baseline."""
import sys,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.fine_retrieval import fine_predict
from src.retrieval import knn_prob
from src.model import build_model
from src.cells import load_cells_json
from src.eval import haversine_km_vec

def location_metrics(pred,truth):
    error=haversine_km_vec(pred[:,0],pred[:,1],truth[:,0],truth[:,1])
    return dict(n=len(error),within_1km=float((error<=1).mean()*100),within_5km=float((error<=5).mean()*100),within_25km=float((error<=25).mean()*100),within_200km=float((error<=200).mean()*100),median_km=float(np.median(error)),mean_km=float(error.mean()),p90_km=float(np.percentile(error,90))),error

def main():
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';out=ROOT/'checkpoints/l8_fine_retrieval';out.mkdir(exist_ok=True)
    recipes=[dict(name='nearest_training_position',k=1,radius_km=25.,prior_weight=0.),dict(name='local5',k=5,radius_km=25.,prior_weight=0.),dict(name='local20',k=20,radius_km=25.,prior_weight=0.),dict(name='local20_classifier_prior',k=20,radius_km=25.,prior_weight=.2)]
    protocol=dict(recipes=recipes,feature_source='Frozen L5/L6 ResNet-50 features',bank_split='15939 train only',validation_n=1000,test_n=357,selection='highest validation within25, then lowest median, then mean; L7 baseline eligible',gate='within25 +.5pp and median <=1.02 baseline, or median improves >=10% with within25 no worse by >.5pp',test_caveat='Repeated benchmark already inspected for L6/L7; exploratory follow-up, not fresh confirmation',coordinate_method='Spherical average only inside the highest-mass 25-km neighbor mode; k1 returns a training-photo coordinate')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    source=ROOT/'checkpoints/l6_kerala_head_regularized';train=torch.load(source/'features_train.pt',weights_only=False);val=torch.load(source/'features_val.pt',weights_only=False)
    assert len(train['ids'])==15939 and len(val['ids'])==1000 and not set(train['ids'])&set(val['ids'])
    model=build_model(num_cells=300,pretrained=False,freeze_backbone=False);ckpt=torch.load(ROOT/'checkpoints/l6_kerala_geographic/selected.pt',map_location='cpu',weights_only=True);model.load_state_dict(ckpt['model_state']);model.to(device).eval()
    cells=load_cells_json(ROOT/'checkpoints/l6_kerala_geographic/cells.json');centers=np.array([[c.centroid_lat,c.centroid_lon] for c in cells])
    bank=train['x'].to(device);xy=torch.tensor(train['coords'],dtype=torch.float32,device=device);labels=torch.tensor(train['labels'],device=device)
    def predict(data,recipe=None):
        predictions=[]
        with torch.inference_mode():
            for x in data['x'].to(device).split(128):
                probs=model.head(x).softmax(1)
                if recipe is None:
                    vote=knn_prob(x,bank,labels);idx=(.8*probs+.2*vote).argmax(1).cpu().numpy();predictions.append(centers[idx])
                else:
                    args={k:v for k,v in recipe.items() if k!='name'};p=fine_predict(x,bank,xy,labels,probs,**args);predictions.append(p['coords'].cpu().numpy())
        return np.concatenate(predictions)
    baseline,_=location_metrics(predict(val),val['coords']);scores={};best=baseline;chosen=None
    key=lambda m:(m['within_25km'],-m['median_km'],-m['mean_km'])
    for recipe in recipes:
        m,_=location_metrics(predict(val,recipe),val['coords']);scores[recipe['name']]=m;print(recipe['name'],m,flush=True)
        if key(m)>key(best):best=m;chosen=recipe
    gate=chosen is not None and ((best['within_25km']>=baseline['within_25km']+.5 and best['median_km']<=baseline['median_km']*1.02) or (best['median_km']<=baseline['median_km']*.9 and best['within_25km']>=baseline['within_25km']-.5))
    selection=dict(recipe=chosen,promotion_eligible=gate,baseline_validation=baseline,selected_validation=best,all_validation=scores)
    (out/'selection_before_test.json').write_text(json.dumps(selection,indent=2));print('SELECTION',json.dumps(selection),flush=True)
    result=dict(protocol=protocol,selection=selection,device=device)
    if gate:
        test=torch.load(source/'features_test.pt',weights_only=False);assert not set(test['ids']) & (set(train['ids'])|set(val['ids']))
        base_pred=predict(test);fine_pred=predict(test,chosen);bm,be=location_metrics(base_pred,test['coords']);fm,fe=location_metrics(fine_pred,test['coords']);result.update(baseline_test=bm,selected_test=fm)
        pd.DataFrame(dict(osv_id=test['ids'],lat=test['coords'][:,0],lon=test['coords'][:,1],baseline_lat=base_pred[:,0],baseline_lon=base_pred[:,1],fine_lat=fine_pred[:,0],fine_lon=fine_pred[:,1],baseline_error_km=be,fine_error_km=fe)).to_csv(out/'test_predictions.csv',index=False)
    (out/'results.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps(result),flush=True)
if __name__=='__main__':main()
