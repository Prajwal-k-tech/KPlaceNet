"""Fine-coordinate retrieval: local geographic consensus, never global coordinate averaging."""
import torch
import torch.nn.functional as F

def spherical_mean(coords,weights):
    radians=torch.deg2rad(coords);lat,lon=radians[...,0],radians[...,1]
    xyz=torch.stack((torch.cos(lat)*torch.cos(lon),torch.cos(lat)*torch.sin(lon),torch.sin(lat)),dim=-1)
    center=(xyz*weights[...,None]).sum(-2)
    return torch.rad2deg(torch.stack((torch.atan2(center[...,2],torch.linalg.vector_norm(center[...,:2],dim=-1)),torch.atan2(center[...,1],center[...,0])),dim=-1))

def fine_predict(query,bank,coords,labels=None,classifier_probs=None,k=20,radius_km=25.,prior_weight=0.):
    """Retrieve actual training positions, select a local mode, interpolate only its neighbors.

    Fixed cosine temperature .07. Output is a coordinate, not a class centroid.
    Similarity/consensus values are diagnostics, not calibrated confidence/OOD detection.
    """
    if not 1<=k<=len(bank) or radius_km<=0:raise ValueError('Invalid k or radius')
    sims,indices=(F.normalize(query,dim=1) @ F.normalize(bank,dim=1).T).topk(k,dim=1)
    score=sims/.07
    if prior_weight:
        if labels is None or classifier_probs is None:raise ValueError('Class prior needs labels and probabilities')
        score=score+prior_weight*torch.log(classifier_probs.gather(1,labels[indices]).clamp_min(1e-8))
    weights=score.softmax(dim=1);xy=coords[indices]
    radians=torch.deg2rad(xy);lat=radians[:,:,0];lon=radians[:,:,1]
    dlat=lat[:,:,None]-lat[:,None,:];dlon=lon[:,:,None]-lon[:,None,:]
    a=(torch.sin(dlat/2)**2+torch.cos(lat[:,:,None])*torch.cos(lat[:,None,:])*torch.sin(dlon/2)**2).clamp(0,1)
    distances=6371.*2*torch.atan2(a.sqrt(),(1-a).clamp_min(0).sqrt())
    local=distances<=radius_km
    masses=(local*weights[:,None,:]).sum(-1);anchor=masses.argmax(1)
    mask=local[torch.arange(len(query),device=query.device),anchor]
    selected=weights*mask;selected=selected/selected.sum(1,keepdim=True).clamp_min(1e-8)
    point=spherical_mean(xy,selected)
    return dict(coords=point,indices=indices,similarities=sims,weights=selected,
                consensus_mass=masses.max(1).values,neighbor_count=mask.sum(1))
