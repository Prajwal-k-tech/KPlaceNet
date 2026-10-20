"""Training-only cosine retrieval in the frozen ResNet feature space."""
import torch
import torch.nn.functional as F

def knn_prob(query, bank, labels, k=5, temperature=.07, num_cells=300):
    if len(bank) < k or k < 1 or temperature <= 0:
        raise ValueError('Require a positive temperature and 1 <= k <= bank size')
    similarities=F.normalize(query,dim=1) @ F.normalize(bank,dim=1).T
    values,indices=similarities.topk(k,dim=1)
    result=torch.zeros((len(query),num_cells),device=query.device,dtype=query.dtype)
    result.scatter_add_(1,labels[indices],torch.softmax(values/temperature,dim=1))
    return result
