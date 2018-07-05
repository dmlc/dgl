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

def isbatchable(x_list, method):
    device = lambda x: x.device == x_list[0].device
    dtype = lambda x: x.dtype == x_list[0].dtype
    if method == cat:
        shape = lambda x: x.shape[1:] == x_list[0].shape[1:]
    elif method == stack:
        shape = lambda x: x.shape == x_list[0].shape
    else:
        raise Exception()

    return isinstance(x_list[0], Tensor) \
        and all(map(device, x_list)) \
        and all(map(dtype, x_list)) \
        and all(map(shape, x_list))

def reduce_sum(a):
    return sum(a)

def reduce_max(a):
    a = torch.cat(a, 0)
    a, _ = torch.max(a, 0, keepdim=True)
    return a
