"""Bounded experiment: freeze L5 features, regularize its classification head.

Uses the original seed-42 15,939/1,000 split. Selection uses validation only.
The official test is evaluated once after the selected epoch is frozen.
No new images are claimed; restoring existing OSV-5M is not a new dataset.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, sys, time, os
from pathlib import Path
import numpy as np
import pandas as pd
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cells import assign_cells, load_cells_json
from src.dataset import GeoDataset
from src.eval import haversine_km_vec
from src.model import build_model


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def metrics(logits, coords, labels, centers):
    pred = logits.detach().cpu().argmax(1).numpy()
    xy = centers[pred]
    error = haversine_km_vec(xy[:, 0], xy[:, 1], coords[:, 0], coords[:, 1])
    return dict(n=len(pred), cell_top1_pct=float((pred == labels).mean()*100),
                within_1km=float((error <= 1).mean()*100),
                within_25km=float((error <= 25).mean()*100),
                within_200km=float((error <= 200).mean()*100),
                mean_km=float(error.mean()), median_km=float(np.median(error)),
                p90_km=float(np.percentile(error, 90))), error, pred


def geographic_targets(coords, centers, labels, smoothing=0.1, scale_km=15.0):
    """Mix hard labels with a training-coordinate geographic neighborhood.

    Adaptation of geographic boundary awareness; not the ETH paper's prototype
    interpolation or cross-view model. No validation/test coordinates are used.
    """
    if not 0 <= smoothing <= 1 or scale_km <= 0:
        raise ValueError('Require smoothing in [0,1] and positive scale')
    distances=haversine_km_vec(coords[:,0,None],coords[:,1,None],centers[None,:,0],centers[None,:,1])
    soft=torch.softmax(-torch.tensor(distances,dtype=torch.float32)/scale_km,dim=1)*smoothing
    soft[torch.arange(len(labels)),torch.as_tensor(labels)]+=1-smoothing
    return soft


def features(split, model, output, cells, batch_size):
    csv = ROOT/'data/kerala_osv5m'/f'kerala_{split}.csv'
    cached = output/f'features_{split}.pt'
    identity = dict(csv_sha256=sha(csv), checkpoint_sha256=sha(ARGS.checkpoint))
    if cached.exists():
        value=torch.load(cached, map_location='cpu', weights_only=False)
        if value['identity'] == identity:
            print(f'[features] reuse {split} {len(value["x"])}', flush=True)
            return value
    ds=GeoDataset(csv, train=False)
    loader=DataLoader(ds, batch_size=batch_size, num_workers=2, shuffle=False)
    xs=[]; coords=[]; t=time.monotonic()
    with torch.inference_mode():
        for i,(images,xy) in enumerate(loader):
            xs.append(model.backbone(images.to(DEVICE)).flatten(1).cpu().clone())
            coords.append(xy)
            if (i+1)%16 == 0:
                print(f'[features] {split} {min((i+1)*batch_size,len(ds))}/{len(ds)} {time.monotonic()-t:.1f}s',flush=True)
    xy=torch.cat(coords).numpy()
    value=dict(x=torch.cat(xs), coords=xy, labels=np.array(assign_cells(xy,cells)),
               ids=ds.df.osv_id.astype(str).tolist(), identity=identity)
    torch.save(value,cached)
    return value


def main():
    global DEVICE
    DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42)
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    out=Path(ARGS.output);out.mkdir(parents=True,exist_ok=True)
    checkpoint=torch.load(ARGS.checkpoint,map_location='cpu',weights_only=False)
    cells=load_cells_json(ROOT/'checkpoints/l5_kerala_kmeans300/cells.json')
    centers=np.array([[c.centroid_lat,c.centroid_lon] for c in cells])
    model=build_model(num_cells=len(cells),pretrained=False,freeze_backbone=False)
    model.load_state_dict(checkpoint['model_state'],strict=True);model.to(DEVICE).eval()
    protocol=dict(seed=42, source_checkpoint=str(ARGS.checkpoint), checkpoint_sha256=sha(ARGS.checkpoint),
        cells_sha256=sha(ROOT/'checkpoints/l5_kerala_kmeans300/cells.json'),
        train_n=15939, val_n=1000, test_n=357, device=str(DEVICE), torch_version=torch.__version__,
        gpu_name=torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else None,
        algorithm='Frozen L5 backbone; paired head-only smoothing recipes',
        recipes=['uniform_smoothing','geographic_smoothing'], geographic_scale_km=15.0,
        geometry_source='Training coordinates only; scale fixed before evaluation',
        lr=1e-4, weight_decay=0.01, label_smoothing=0.1, batch_size=256,
        epochs_per_recipe=12, selection='max validation within_25km, then min median_km; baseline eligible',
        promote_rule='validation within25 improves by >=0.5 percentage points, median no more than 2% worse',
        caveats=['Existing random holdout is spatially optimistic.',
                 'Backbone and cell geometry stay unchanged.',
                 'This restores the original dataset; no additional training data was added.',
                 'One seed and two prespecified regularization recipes; no statistical significance claim.'])
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    val=features('val',model,out,cells,ARGS.batch_size)
    baseline_val,_,_=metrics(model.head(val['x'].to(DEVICE)).detach(),val['coords'],val['labels'],centers)
    print('[baseline val]',baseline_val,flush=True)
    train=features('train',model,out,cells,ARGS.batch_size)
    assert len(set(train['ids']) & set(val['ids'])) == 0
    assert len(train['ids']) ==15939 and len(val['ids'])==1000
    original=copy.deepcopy(model.head.state_dict())
    selected=copy.deepcopy(original);best=baseline_val;best_epoch=0
    target=torch.tensor(train['labels'],dtype=torch.long)
    geo_target=geographic_targets(train['coords'],centers,train['labels'])
    history=[];start=time.monotonic();selected_recipe='baseline'
    for recipe in ['uniform_smoothing','geographic_smoothing']:
        torch.manual_seed(42)
        model.head.load_state_dict(original)
        optimizer=torch.optim.AdamW(model.head.parameters(),lr=1e-4,weight_decay=0.01)
        for epoch in range(1,13):
            model.head.train();order=torch.randperm(len(target));loss_sum=0
            for batch in order.split(256):
                optimizer.zero_grad(set_to_none=True)
                logits=model.head(train['x'][batch].to(DEVICE))
                if recipe=='uniform_smoothing':
                    loss=torch.nn.functional.cross_entropy(logits,target[batch].to(DEVICE),label_smoothing=0.1)
                else:
                    loss=torch.nn.functional.cross_entropy(logits,geo_target[batch].to(DEVICE))
                loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(batch)
            model.head.eval()
            with torch.inference_mode():
                score,_,_=metrics(model.head(val['x'].to(DEVICE)),val['coords'],val['labels'],centers)
            row=dict(recipe=recipe,epoch=epoch,train_loss=loss_sum/len(target),validation=score)
            history.append(row);print('[epoch]',json.dumps(row),flush=True)
            key=lambda m:(m['within_25km'],-m['median_km'])
            if key(score)>key(best):
                selected=copy.deepcopy(model.head.state_dict());best=score;best_epoch=epoch;selected_recipe=recipe
    training_seconds=time.monotonic()-start
    model.head.load_state_dict(selected)
    promoted=(best['within_25km']>=baseline_val['within_25km']+.5 and best['median_km']<=baseline_val['median_km']*1.02)
    frozen=dict(selected_recipe=selected_recipe,selected_epoch=best_epoch,validation=best,baseline_validation=baseline_val,promotion_eligible=promoted)
    (out/'selection_before_test.json').write_text(json.dumps(frozen,indent=2))
    candidate=dict(epoch=best_epoch,model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},num_cells=len(cells),
        cells=checkpoint['cells'],args={**checkpoint.get('args',{}),'l6_protocol':protocol},
        source_checkpoint_sha256=protocol['checkpoint_sha256'],selection=frozen)
    torch.save(candidate,out/'selected.pt')
    print('[selection frozen]',json.dumps(frozen),flush=True)
    test=features('test',model,out,cells,ARGS.batch_size)
    assert not (set(test['ids']) & (set(train['ids']) | set(val['ids'])))
    with torch.inference_mode():
        candidate_test,errors,pred=metrics(model.head(test['x'].to(DEVICE)),test['coords'],test['labels'],centers)
        model.head.load_state_dict(original)
        baseline_test,base_errors,base_pred=metrics(model.head(test['x'].to(DEVICE)),test['coords'],test['labels'],centers)
    result=dict(protocol=protocol,selection=frozen,history=history,baseline_test=baseline_test,
        selected_test=candidate_test,head_training_seconds=training_seconds,
        test_used_for='Final paired reporting after selection; not epoch/hyperparameter selection')
    (out/'results.json').write_text(json.dumps(result,indent=2))
    pd.DataFrame(dict(osv_id=test['ids'],lat=test['coords'][:,0],lon=test['coords'][:,1],
        baseline_cell=base_pred,selected_cell=pred,baseline_error_km=base_errors,
        selected_error_km=errors)).to_csv(out/'test_predictions.csv',index=False)
    print('[final]',json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/l5_kerala_kmeans300/last.pt')
    p.add_argument('--output',type=Path,default=ROOT/'checkpoints/l6_kerala_head_regularized')
    p.add_argument('--batch-size',type=int,default=32,help='Feature extraction batch; head uses 256')
    ARGS=p.parse_args();main()
