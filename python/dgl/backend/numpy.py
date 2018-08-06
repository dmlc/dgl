from __future__ import absolute_import

import numpy as np
import scipy as sp

Tensor = np.ndarray
SparseTensor = sp.sparse.spmatrix

def asnumpy(a):
    return a

def pack(arrays):
    return np.concatenate(arrays, axis=0)

def unpack(a):
    return np.split(a, a.shape[0], axis=0)

def shape(a):
    return a.shape
