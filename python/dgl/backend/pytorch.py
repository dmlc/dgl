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

def expand_dims(a, axis):
    return a.unsqueeze(axis)

def ndim(a):
    return a.dim()

def reduce_sum(a):
    return sum(a)

def reduce_max(a):
    a = torch.cat(a, 0)
    a, _ = torch.max(a, 0, keepdim=True)
    return a

def shape(a):
    return a.size()

def split(ary, indices_or_sections, axis=0):
    return torch.split(ary, indices_or_sections, axis)

class Batchable:
    def __init__(batchable, error=''):
        self.batchable = batchable
        if not batchable:
            self.error = error

    def __bool__(self):
        return self.batchable

def isbatchable(xs, method):
    # TODO(gaiyu): error message
    if not all(isinstance(x, Tensor) for x in xs):
        return Batchable(False)

    if not all(x.device == xs[0].device for x in xs):
        return Batchable(False)

    if not all(x.dtype == xs[0].dtype for x in xs):
        return Batchable(False)

    if method == cat:
        if not all(x.shape[1:] == xs[0].shape[1:] for x in xs):
            return Batchable(False)
    elif method == stack:
        if not all(x.shape == x_list[0].shape for x in xs):
            return Batchable(False)
    else:
        raise Exception()

    return Batchable(True)

class Unbatchable:
    """ Whether possible to unbatch.
    """
    def __init__(unbatchable, error=''):
        self.unbatchable = unbatchable
        if not unbatchable:
            self.error = error

    def __bool__(self):
        return self.unbatchable

def unbatchable(x):
    # TODO(gaiyu): error message
    if not isinstance(x, Tensor):
        return Unbatchable(False)

    if F.ndim(x) < 1:
        return Unbatchable(False)

    return Unbatchable(True)
