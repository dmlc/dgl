from __future__ import absolute_import

import numpy as np
import scipy.sparse as sp
import warnings

warnings.warn('Detect using numpy backend. Please be aware that numpy does not support autograd!')

def data_type_dict():
    return {'float16' : np.float16,
            'float32' : np.float32,
            'float64' : np.float64,
            'uint8'   : np.uint8,
            'int8'    : np.int8,
            'int16'   : np.int16,
            'int32'   : np.int32,
            'int64'   : np.int64}

def cpu():
    return 'cpu'

def tensor(data, dtype=None):
    return np.array(data, dtype)

def sparse_matrix(data, index, shape, force_format=False):
    fmt = index[0]
    if fmt == 'coo':
        i = index[1][0,:]
        j = index[1][1,:]
        return sp.coo_matrix((data, (i, j)), shape=shape)
    elif fmt == 'csr':
        indices = index[1]
        indptr = index[2]
        return sp.csr_matrix((data, indices, indptr), shape=shape)
    else:
        raise TypeError('Invalid format: %s.' % fmt)

def sparse_matrix_indices(spmat):
    if spmat.format == 'coo':
        return ('coo', np.stack(spmat.row, spmat.col))
    elif spmat.format == 'csr':
        return ('csr', spmat.indices, spmat.indptr)
    else:
        raise TypeError('Invalid format: %s.' % spmat.format)

def is_tensor(obj):
    return isinstance(obj, np.ndarray)

def shape(input):
    return input.shape

def dtype(input):
    return input.dtype

def context(input):
    return 'cpu'

def astype(input, ty):
    return input.astype(ty)

def asnumpy(input):
    return input

def copy_to(input, ctx):
    return input

def sum(input, dim):
    return np.sum(input, axis=dim)

def max(input, dim):
    return np.max(input, axis=dim)

def cat(seq, dim):
    return np.concatenate(seq, axis=dim)

def split(input, sizes_or_sections, dim):
    dimsize = input.shape[dim]
    if isinstance(sizes_or_sections, int):
        if dimsize % sizes_or_sections != 0:
            raise ValueError('Require dimension %d to be equally splitted'
                             ' to %d pieces, but got %d.' % (dim, sizes_or_sections, dimsize))
        idx = np.arange(sizes_or_sections, dimsize, sizes_or_sections)
    else:
        idx = np.cumsum(sizes_or_sections)[0:-1]
    return np.split(input, idx, axis=dim)

def gather_row(data, row_index):
    return data[row_index]

def scatter_row(data, row_index, value):
    # NOTE: inplace instead of out-place
    data[row_index] = value
    return data

def scatter_row_inplace(data, row_index, value):
    data[row_index] = value

def squeeze(input, dim):
    return np.squeeze(input, dim)

def unsqueeze(input, dim):
    return np.unsqueeze(input, dim)

def reshape(input, shape):
    return np.reshape(input ,shape)

def zeros(shape, dtype):
    return np.zeros(shape, dtype=dtype)

def ones(shape, dtype):
    return np.ones(shape, dtype=dtype)

def spmm(x, y):
    return x.dot(y)

def unique(input):
    return np.unique(input)

def full_1d(length, fill_value):
    return np.full((length,), fill_value)

def nonzero_1d(input):
    return np.nonzero(input)[0]

def sort_1d(input):
    return np.sort(input), np.argsort(input)

def arange(start, stop):
    return np.arange(start, stop, dtype=np.int64)

def rand_shuffle(arr):
    copy = np.copy(arr)
    np.random.shuffle(copy)
    return copy

# zerocopy_to_dlpack not enabled

# zerocopy_from_dlpack not enabled

def zerocopy_to_numpy(input):
    return input

def zerocopy_from_numpy(np_array):
    return np_array

# create_immutable_graph_index not enabled
