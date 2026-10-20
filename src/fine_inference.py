"""Offline-ready fine-coordinate retrieval using the validation-selected L10 descriptor."""
from pathlib import Path
import math
import torch
import torch.nn.functional as F

def load_fine_resources(root,device,checkpoint_sha256):
    folder=Path(root)/'checkpoints/fine_retrieval'
    bank=torch.load(folder/'bank.pt',map_location='cpu',weights_only=True)
    if bank['checkpoint_sha256']!=checkpoint_sha256:raise ValueError('Fine bank and ResNet checkpoint do not match')
    weights=folder/'dinov2_vits14_pretrain.pth'
    from src.demo_config import sha256_file
    if sha256_file(weights)!=bank['dino_weights_sha256']:raise ValueError('Fine DINO checkpoint fingerprint mismatch')
    if len(bank['ids'])!=len(bank['coords']) or bank['x'].shape!=(len(bank['ids']),2432):raise ValueError('Fine bank dimensions invalid')
    model=torch.hub.load(str(folder/'dinov2_source'),'dinov2_vits14',source='local',pretrained=False)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True);model.to(device).eval()
    return dict(model=model,x=F.normalize(bank['x'].to(device).float(),dim=1),coords=bank['coords'],ids=bank['ids'],resnet_weight=bank['resnet_weight'])

def predict_fine(image_tensor,resnet_features,resources):
    device=image_tensor.device
    with torch.inference_mode():
        with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device.type=='cuda'):
            dino=resources['model'](image_tensor)
        w=resources['resnet_weight']
        query=torch.cat((F.normalize(resnet_features.float(),dim=1)*math.sqrt(w),F.normalize(dino.float(),dim=1)*math.sqrt(1-w)),dim=1)
        similarity,idx=(F.normalize(query,dim=1) @ resources['x'].T).max(dim=1)
    i=int(idx[0]);lat,lon=map(float,resources['coords'][i])
    return dict(lat=lat,lon=lon,similarity=float(similarity[0]),reference_id=resources['ids'][i],reference_index=i)
