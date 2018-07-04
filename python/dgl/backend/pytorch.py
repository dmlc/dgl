from __future__ import absolute_import

import torch
import scipy.sparse

Tensor = torch.Tensor
SparseTensor = scipy.sparse.spmatrix

def asnumpy(a):
    return a.cpu().numpy()

def reduce_sum(a):
    return sum(a)

def reduce_max(a):
    a = torch.cat(a, 0)
    a, _ = torch.max(a, 0, keepdim=True)
    return a
