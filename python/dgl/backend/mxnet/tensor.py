from __future__ import absolute_import

import numpy as np
import mxnet as mx
import mxnet.ndarray as nd

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
    return mx.cpu()

def tensor(data, dtype=None):
    return nd.array(data, dtype=dtype)

def sparse_matrix(data, index, shape, force_format=False):
    fmt = index[0]
    if fmt == 'coo':
        if force_format:
            raise TypeError('MXNet backend only supports CSR format,'
                            ' but COO format is forced.')
        coord = index[1]
        # generate convert idx
        # FIXME: cannot use int64
        tmp_data = nd.arange(len(coord[0]), dtype=data.dtype, ctx=coord[0].context)
        tmp_spmat = nd.sparse.csr_matrix((tmp_data, (coord[0], coord[1])),
                tuple(shape), ctx=data.context)
        convert_idx = nd.cast(tmp_spmat.data, dtype='int64')
        # shuffle the data
        data = data[convert_idx]
        spmat = nd.sparse.csr_matrix((data, tmp_spmat.indices, tmp_spmat.indptr),
                tuple(shape), ctx=data.context)
        return spmat, convert_idx
    elif fmt == 'csr':
        indices = index[1]
        indptr = index[2]
        spmat = nd.sparse.csr_matrix((data, indices, indptr),
                tuple(shape), ctx=data.context)
        # No conversion is required.
        return spmat, None
    else:
        raise TypeError('Invalid format: %s.' % fmt)

def sparse_matrix_indices(spmat):
    return ('csr', spmat.indices, spmat.indptr)

def is_tensor(obj):
    return isinstance(obj, nd.NDArray)

def shape(input):
    # NOTE: the input cannot be a symbol
    return input.shape

def dtype(input):
    # NOTE: the input cannot be a symbol
    return input.dtype

def ndim(input):
    return input.ndim

def context(input):
    return input.context

def astype(input, ty):
    return nd.cast(input, ty)

def asnumpy(input):
    return input.asnumpy()

def copy_to(input, ctx):
    return input.as_in_context(ctx)

def sum(input, dim):
    return nd.sum(input, axis=dim)

def mean(input, dim):
    return nd.mean(input, axis=dim)

def max(input, dim):
    return nd.max(input, axis=dim)

def cat(seq, dim):
    return nd.concat(*seq, dim=dim)

def stack(seq, dim):
    return nd.stack(*seq, dim=dim)

def split(x, sizes_or_sections, dim):
    if isinstance(sizes_or_sections, list) or isinstance(sizes_or_sections, np.ndarray):
        # TODO: fallback to numpy is unfortunate
        np_arr = x.asnumpy()
        indices = np.cumsum(sizes_or_sections)[:-1]
        res = np.split(np_arr, indices, axis=dim)
        return [tensor(arr, dtype=x.dtype) for arr in res]
    else:
        return nd.split(x, sizes_or_sections, axis=dim)

def gather_row(data, row_index):
    if isinstance(row_index, nd.NDArray):
        return nd.take(data, row_index)
    else:
        return data[row_index,]

def narrow_row(data, start, stop):
    return nd.slice(data, begin=start, end=stop)

def scatter_row(data, row_index, value):
    return mx.nd.contrib.index_copy(data, row_index, value)

def scatter_row_inplace(data, row_index, value):
    data[row_index] = value

def squeeze(input, dim):
    return nd.squeeze(input, axis=dim)

def unsqueeze(input, dim):
    return nd.expand_dims(input, axis=dim)

def reshape(input, shape):
    # NOTE: the input cannot be a symbol
    return nd.reshape(input ,shape)

def zeros(shape, dtype, ctx):
    return nd.zeros(shape, dtype=dtype, ctx=ctx)

def ones(shape, dtype, ctx):
    return nd.ones(shape, dtype=dtype, ctx=ctx)

def spmm(x, y):
    return nd.dot(x, y)

def unsorted_1d_segment_sum(input, seg_id, n_segs, dim):
    # TODO: support other dimensions
    assert dim == 0, 'MXNet only supports segment sum on first dimension'

    # Use SPMV to simulate segment sum
    ctx = input.context
    n_inputs = input.shape[0]
    input_shape_suffix = input.shape[1:]
    input = input.reshape(n_inputs, -1)
    n_range = nd.arange(n_inputs, dtype='int64').as_in_context(input.context)
    w_nnz = nd.ones(n_inputs).as_in_context(input.context)
    w_nid = nd.stack(seg_id, n_range, axis=0)
    w = nd.sparse.csr_matrix((w_nnz, (seg_id, n_range)), (n_segs, n_inputs))
    w = w.as_in_context(input.context)
    y = nd.dot(w, input)
    y = nd.reshape(y, (n_segs,) + input_shape_suffix)
    return y

def unsorted_1d_segment_mean(input, seg_id, n_segs, dim):
    # TODO: support other dimensions
    assert dim == 0, 'MXNet only supports segment mean on first dimension'

    n_ones = nd.ones_like(seg_id).astype(input.dtype)
    w = unsorted_1d_segment_sum(n_ones, seg_id, n_segs, 0)
    w = nd.clip(w, a_min=1, a_max=np.inf)
    y = unsorted_1d_segment_sum(input, seg_id, n_segs, dim)
    y /= w.reshape((-1,) + (1,) * (y.ndim - 1))
    return y

def unique(input):
    # TODO: fallback to numpy is unfortunate
    tmp = input.asnumpy()
    tmp = np.unique(tmp)
    return nd.array(tmp, ctx=input.context, dtype=input.dtype)

def full_1d(length, fill_value):
    return nd.full((length,), fill_value)

def nonzero_1d(input):
    # TODO: fallback to numpy is unfortunate
    tmp = input.asnumpy()
    tmp = np.nonzero(tmp)[0]
    return nd.array(tmp, ctx=input.context, dtype=input.dtype)

def sort_1d(input):
    # TODO: this isn't an ideal implementation.
    val = nd.sort(input, axis=None, is_ascend=True)
    idx = nd.argsort(input, is_ascend=True)
    idx = nd.cast(idx, dtype='int64')
    return val, idx

def arange(start, stop):
    return nd.arange(start, stop, dtype=np.int64)

def rand_shuffle(arr):
    return mx.nd.random.shuffle(arr)

def zerocopy_to_dlpack(arr):
    return arr.to_dlpack_for_read()

def zerocopy_from_dlpack(dlpack_arr):
    return nd.from_dlpack(dlpack_arr)

def zerocopy_to_numpy(arr):
    # NOTE: not zerocopy
    return arr.asnumpy()

def zerocopy_from_numpy(np_data):
    # NOTE: not zerocopy
    return nd.array(np_data, dtype=np_data.dtype)
