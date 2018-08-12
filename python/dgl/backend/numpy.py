from __future__ import absolute_import

import numpy as np
import scipy as sp

Tensor = np.ndarray
SparseTensor = sp.sparse.spmatrix

def asnumpy(a):
    return a

def pack(arrays):
    return np.concatenate(arrays, axis=0)

def unpack(a, indices_or_sections=None):
    if indices_or_sections is None:
        indices_or_sections = a.shape[0]
    return np.split(a, indices_or_sections, axis=0)

def shape(a):
    return a.shape
