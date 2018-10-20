from __future__ import absolute_import

import numpy as np
import scipy as sp

Tensor = np.ndarray
SparseTensor = sp.sparse.spmatrix

def asnumpy(a):
    return a

def pack(arrays):
    return np.concatenate(arrays, axis=0)

def unpack(a, split_size_or_sections=None):
    if split_size_or_sections is None:
        indices_or_sections = a.shape[0]
    else:
        # convert split size to split indices by cumsum
        indices_or_sections = np.cumsum(split_size_or_sections)[:-1]
    return np.split(a, indices_or_sections, axis=0)

def shape(a):
    return a.shape

def nonzero_1d(a):
    assert a.ndim == 2
    return np.nonzero(a)[0]
