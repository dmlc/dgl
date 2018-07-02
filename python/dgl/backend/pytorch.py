from __future__ import absolute_import

import torch
import scipy.sparse

Tensor = torch.Tensor
SparseTensor = scipy.sparse.spmatrix

def asnumpy(a):
    return a.cpu().numpy()

def reduce_sum(a):
    if isinstance(a, list):
        return sum(a)
    elif isinstance(a, Tensor):
        return torch.sum(a, 0, keepdim=True)
    else:
        raise Exception("reduce_sum only supports input of type Tensor or list of Tensor")
