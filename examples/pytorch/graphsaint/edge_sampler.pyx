# cython: language_level=3
# distutils: language=c++

from libc.stdlib cimport rand
from libcpp.algorithm cimport sort, unique, lower_bound
from libcpp.vector cimport vector
cimport numpy as np
import numpy as np
cdef extern from "stdlib.h":
    int RAND_MAX

cdef inline void npy2vec_float(np.ndarray[float,ndim=1,mode='c'] nda, vector[float]& vec):
    cdef int size = nda.size
    cdef float* vec_c = &(nda[0])
    vec.assign(vec_c,vec_c+size)

def sample_edge(unsigned int edge_budget, np.ndarray[float,ndim=1,mode='c'] p_cumsum_nda):
    cdef:
        int e = 0, i = 0
        float p_cumsum = p_cumsum_nda[-1]
        float ran = 0.
        vector[float] p_cumsum_vec
        vector[int] sampled_edges
    sampled_edges.reserve(edge_budget)
    npy2vec_float(p_cumsum_nda, p_cumsum_vec)
    while i < edge_budget:
        ran = (<float> rand()) / RAND_MAX * p_cumsum
        e = lower_bound(p_cumsum_vec.begin(), p_cumsum_vec.end(),
                        ran) - p_cumsum_vec.begin()
        sampled_edges.push_back(e)
        i += 1
    return sampled_edges



