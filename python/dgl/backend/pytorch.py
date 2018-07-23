from __future__ import absolute_import

import torch
import scipy.sparse

Tensor = torch.Tensor
SparseTensor = scipy.sparse.spmatrix

cat = torch.cat
stack = torch.stack
sum = torch.sum

def asnumpy(a):
    return a.cpu().numpy()

def concatenate(tensors, axis=0):
    return th.concatenate(tensors, axis)

def reduce_sum(a):
    return sum(a)

def reduce_max(a):
    a = torch.cat(a, 0)
    a, _ = torch.max(a, 0, keepdim=True)
    return a

def packable(tensors):
    return all(isinstance(t, torch.Tensor) and \
               t.device == tensors[0].device and \
               t.dtype == tensors[0].dtype and \
               t.shape[1:] == tensors[0].shape[1:] for t in tensors)

def pack(tensors):
    return torch.cat(tensors)

def unpackable(t):
    return isinstance(t, torch.Tensor) and t.numel() > 0

def unpack(t):
    return th.split(t, 1)

def shape(t):
    return t.shape
