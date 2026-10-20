"""Fixed training-only cosine KNN diagnostic; validation gates any test access."""
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_l6_kerala_head import metrics
from src.cells import load_cells_json
from src.model import build_model

from src.retrieval import knn_prob

def main():
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    source=ROOT/'checkpoints/l6_kerala_head_regularized';out=ROOT/'checkpoints/l7_knn_diagnostic';out.mkdir(exist_ok=True)
    protocol=dict(k=5,similarity='cosine of L5/L6 frozen ResNet features',neighbor_weight_temperature=.07,blend_knn_weight=.2,
      bank='15,939 training images only; never validation/test',selection='Fixed 80% L6 softmax + 20% KNN vote; validation improvement >= .5pp within25 and median <= 1.02 baseline required',
      caveats=['Random validation is spatially optimistic.','Does not detect outside-region inputs.','Official test already reported in L6; any later reporting is a reused benchmark, not fresh confirmation.'])
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    train=torch.load(source/'features_train.pt',weights_only=False);val=torch.load(source/'features_val.pt',weights_only=False)
    assert not set(train['ids']) & set(val['ids'])
    bank=train['x'].to(device);labels=torch.tensor(train['labels'],device=device)
    ckpt=torch.load(ROOT/'checkpoints/l6_kerala_geographic/selected.pt',weights_only=True,map_location='cpu')
    model=build_model(num_cells=300,pretrained=False,freeze_backbone=False);model.load_state_dict(ckpt['model_state']);model.to(device).eval()
    cells=load_cells_json(ROOT/'checkpoints/l6_kerala_geographic/cells.json');centers=np.array([[c.centroid_lat,c.centroid_lon] for c in cells])
    def compare(data):
        with torch.inference_mode():
            x=data['x'].to(device);base=torch.softmax(model.head(x),dim=1)
            votes=torch.cat([knn_prob(q,bank,labels) for q in x.split(128)])
            return {name:metrics(p,data['coords'],data['labels'],centers)[0] for name,p in [('l6',base),('knn_only',votes),('hybrid',.8*base+.2*votes)]}
    validation=compare(val);a,b=validation['l6'],validation['hybrid']
    eligible=b['within_25km']>=a['within_25km']+.5 and b['median_km']<=a['median_km']*1.02
    result=dict(protocol=protocol,validation=validation,promotion_eligible=eligible,device=device)
    (out/'selection_before_test.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    if eligible:
        test=torch.load(source/'features_test.pt',weights_only=False)
        assert not set(test['ids']) & (set(train['ids']) | set(val['ids']))
        result['reused_official_test']=compare(test)
    (out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
