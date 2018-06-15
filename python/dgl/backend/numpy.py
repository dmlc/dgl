from __future__ import absolute_import

import numpy as np
import scipy as sp

Tensor = np.ndarray
SparseTensor = sp.sparse.spmatrix

def asnumpy(a):
    return a
