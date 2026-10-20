"""Regional bilinear geographic prototypes; ground-only adaptation, not ETH reproduction."""
import numpy as np
import torch
import torch.nn.functional as F

KM_PER_DEGREE=111.195
LON_SCALE=KM_PER_DEGREE*np.cos(np.radians(10.4))

def local_xy(coords):
    """Fixed regional equirectangular coordinates in approximate kilometers."""
    return (np.asarray(coords,dtype=np.float64)-[8.,74.5])*[KM_PER_DEGREE,LON_SCALE]

def build_grid(coords,spacing_km):
    """Four bilinear nodes per training position; only train-supported nodes exist."""
    if spacing_km<=0:raise ValueError('spacing_km must be positive')
    xy=local_xy(coords)/spacing_km;base=np.floor(xy).astype(np.int64);f=xy-base
    offsets=np.array([[0,0],[0,1],[1,0],[1,1]])
    nodes=base[:,None,:]+offsets[None,:,:]
    unique,inverse=np.unique(nodes.reshape(-1,2),axis=0,return_inverse=True)
    a,b=f[:,0],f[:,1]
    weights=np.stack(((1-a)*(1-b),(1-a)*b,a*(1-b),a*b),axis=1)
    centers=unique*spacing_km/[KM_PER_DEGREE,LON_SCALE]+[8.,74.5]
    return {'indices':torch.tensor(inverse.reshape(-1,4)), 'weights':torch.tensor(weights,dtype=torch.float32),'coords':centers,'xy':unique*spacing_km}

def initialize_prototypes(features,indices,weights,count):
    """Weighted train-feature means are only an initialization, not learned weights."""
    values=torch.zeros((count,features.shape[1]),device=features.device)
    for j in range(4):values.index_add_(0,indices[:,j],features*weights[:,j,None])
    return F.normalize(values,dim=1)

def prototype_loss(query,prototypes,indices,weights,negative_mask,alpha=2.,beta=50.,margin=.5):
    """One interpolated positive plus geographically separated prototype negatives.

    Ground-prototype edge only; aerial-ground/prototype edges and encoder training
    from Scaling Geo-Localization are deliberately absent in this adaptation.
    """
    p=F.normalize(prototypes,dim=1)
    positive=F.normalize((p[indices]*weights[:,:,None]).sum(1),dim=1)
    pos=F.softplus(-alpha*((query*positive).sum(1)-margin))/alpha
    negatives=(beta*(query@p.T-margin)).masked_fill(~negative_mask,-torch.inf)
    # log(1+sum(exp(scores))) in stable logsumexp form.
    neg=torch.logsumexp(torch.cat((negatives,torch.zeros((len(query),1),device=query.device)),1),1)/beta
    return (pos+neg).mean()

def rerank_references(query,bank,prototype_codes,weight=.1,k=50):
    """Top-k ground retrieval with an interpolated geographic appearance prior."""
    if not 0<=weight<=1 or k<1 or k>len(bank):raise ValueError('Invalid reranking weight or neighbor count')
    scores,indices=(F.normalize(query,dim=1)@F.normalize(bank,dim=1).T).topk(k,1)
    prior=(F.normalize(query,dim=1)[:,None,:]*F.normalize(prototype_codes,dim=1)[indices]).sum(2)
    chosen=(scores+weight*prior).argmax(1)
    return indices[torch.arange(len(query),device=query.device),chosen]
