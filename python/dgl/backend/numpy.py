from __future__ import absolute_import

import numpy as np
import scipy as sp

Tensor = np.ndarray
SparseTensor = sp.sparse.spmatrix

def asnumpy(a):
    return a

def concatenate(arrays, axis=0):
    return np.concatenate(arrays, axis)

def packable(arrays):
    return all(isinstance(a, np.ndarray) for a in arrays) and \
           all(a.dtype == arrays[0].dtype for a in arrays) and \
           all(a.shape[1:] == arrays[0].shape[1:] for a in arrays)

def pack(arrays):
    return np.concatenate(arrays, axis=0)

def unpackable(a):
    return isinstance(a, np.ndarray) and a.size > 0

def unpack(a):
    return np.split(a, a.shape[0], axis=0)

def shape(a):
    return a.shape
