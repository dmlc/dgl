from __future__ import absolute_import

import numpy as np
import mxnet as mx
print(mx.__version__)
import mxnet.ndarray as nd
import numbers
from ... import ndarray as dglnd
from ... import kernel as K

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
    # MXNet always returns a float tensor regardless of type inside data.
    # This is a workaround.
    if dtype is None:
        if isinstance(data[0], numbers.Integral):
            dtype = np.int64
        else:
            dtype = np.float32
    return nd.array(data, dtype=dtype)

def get_preferred_sparse_format():
    """Get the preferred sparse matrix format supported by the backend.

    Different backends have their preferred backend. This info is useful when
    constructing a sparse matrix.
    """
    return "csr"

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

def device_type(ctx):
    return ctx.device_type

def device_id(ctx):
    return ctx.device_id

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
    return nd.stack(*seq, axis=dim)

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
    # MXNet workaround for empty row index
    if len(row_index) == 0:
        return data[0:0]

    if isinstance(row_index, nd.NDArray):
        return nd.take(data, row_index)
    else:
        return data[row_index,]

def narrow_row(data, start, stop):
    return data[start:stop]

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

def zeros_like(input):
    return nd.zeros_like(input)

def ones(shape, dtype, ctx):
    return nd.ones(shape, dtype=dtype, ctx=ctx)

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
    y = y / w.reshape((-1,) + (1,) * (y.ndim - 1))
    return y

def boolean_mask(input, mask):
    return mx.contrib.nd.boolean_mask(input, mask)

def equal(x, y):
    return x == y

def logical_not(input):
    return nd.logical_not(input)

def unique(input):
    # TODO: fallback to numpy is unfortunate
    tmp = input.asnumpy()
    tmp = np.unique(tmp)
    return nd.array(tmp, ctx=input.context, dtype=input.dtype)

def full_1d(length, fill_value, dtype, ctx):
    return nd.full((length,), fill_value, dtype=dtype, ctx=ctx)

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

def zerocopy_to_dgl_ndarray(arr):
    return dglnd.from_dlpack(arr.to_dlpack_for_read())

def zerocopy_to_dgl_ndarray_for_write(arr):
    return dglnd.from_dlpack(arr.to_dlpack_for_write())

def zerocopy_from_dgl_ndarray(arr):
    return nd.from_dlpack(arr.to_dlpack())


class SrcOpEdgeReduce(mx.autograd.Function):
    def __init__(self, reducer, binary_op, spmat, out_size, src_map, edge_map,
                 out_map):
        super(SrcOpEdgeReduce, self).__init__()
        self.reducer = reducer
        self.binary_op = binary_op
        self.spmat = spmat
        self.out_size = out_size
        self.src_map = src_map
        self.edge_map = edge_map
        if reducer == "none":
            self.forward_out_map = out_map[0]
            self.backward_out_map = out_map[1]
        else:
            self.forward_out_map = out_map
            self.backward_out_map = out_map

    def forward(self, src_data, edge_data):
        src_data_nd = zerocopy_to_dgl_ndarray(src_data)
        edge_data_nd = zerocopy_to_dgl_ndarray(edge_data)
        feat_shape = K.infer_binary_feature_shape(src_data_nd, edge_data_nd)
        out_data = nd.empty((self.out_size,) + feat_shape,
                            ctx=src_data.context, dtype=src_data.dtype)
        out_data_nd = zerocopy_to_dgl_ndarray_for_write(out_data)
        K.src_op_edge_reduce(
            self.reducer, self.binary_op, self.spmat[0], self.spmat[1],
            self.spmat[2], self.spmat[3], self.src_map, self.edge_map[0],
            src_data_nd, edge_data_nd, self.forward_out_map, out_data_nd)
        self.save_for_backward(src_data_nd, edge_data_nd, out_data_nd,
                               feat_shape)
        return out_data

    def backward(self, grad_out):
        src_data_nd, edge_data_nd, out_data_nd, feat_shape = self.saved_tensors
        grad_out_nd = zerocopy_to_dgl_ndarray(grad_out)
        grad_src = nd.empty((src_data_nd.shape[0],) + feat_shape,
                            ctx=grad_out.context, dtype=grad_out.dtype)
        K.backward_lhs_src_mul_edge_reduce(
            self.reducer, self.binary_op, self.spmat[0], self.spmat[1],
            self.spmat[2], self.spmat[3], self.src_map, self.edge_map[1],
            self.backward_out_map, src_data_nd, edge_data_nd, out_data_nd,
            grad_out_nd, zerocopy_to_dgl_ndarray_for_write(grad_src))
        grad_src = _reduce_grad(grad_src, src_data_nd.shape)
        grad_edge = nd.empty((edge_data_nd.shape[0],) + feat_shape,
                             ctx=grad_out.context, dtype=grad_out.dtype)
        K.backward_rhs_src_mul_edge_reduce(
            self.reducer, self.binary_op, self.spmat[0], self.spmat[1],
            self.spmat[2], self.spmat[3], self.src_map, self.edge_map[1],
            self.backward_out_map, src_data_nd, edge_data_nd, out_data_nd,
            grad_out_nd, zerocopy_to_dgl_ndarray_for_write(grad_edge))
        grad_edge = _reduce_grad(grad_edge, edge_data_nd.shape)
        return grad_src, grad_edge


def src_op_edge_reduce(reducer, binary_op, spmat, src_data, edge_data,
                       out_size, src_map, edge_map, out_map):
    func = SrcOpEdgeReduce(reducer, binary_op, spmat, out_size, src_map,
                           edge_map, out_map)
    return func(src_data, edge_data)


class SrcOpDstReduce(mx.autograd.Function):
    def forward(self, reducer, binary_op, spmat, src_data, dst_data, out_size,
                src_map, dst_map, out_map):
        src_data_nd = zerocopy_to_dgl_ndarray(src_data)
        dst_data_nd = zerocopy_to_dgl_ndarray(dst_data)
        feat_shape = K.infer_binary_feature_shape(src_data_nd, dst_data_nd)
        out_data = nd.empty((out_size,) + feat_shape, ctx=src_data.context,
                            dtype=src_data.dtype)
        out_data_nd = zerocopy_to_dgl_ndarray_for_write(out_data)
        if reducer == "none":
            forward_out_map = out_map[0]
            backward_out_map = out_map[1]
        else:
            forward_out_map = out_map
            backward_out_map = out_map
        K.src_op_dst_reduce(
            reducer, binary_op, spmat[0], spmat[1], spmat[2], spmat[3],
            src_map, dst_map, src_data, dst_data, forward_out_map, out_data,
            feat_shape)
        self.backward_cache = (reducer, binary_op, spmat, src_map, dst_map,
                               backward_out_map, src_data_nd, dst_data_nd,
                               out_data_nd, feat_shape)
        return zerocopy_from_dgl_ndarray(out_data)

    def backward(self, grad_out):
        reducer, binary_op, spmat, src_map, dst_map, backward_out_map, \
            src_data_nd, dst_data_nd, out_data_nd, feat_shape \
            = self.backward_cache
        self.backward_cache = None
        grad_src = None
        grad_dst = None
        grad_out_nd = zerocopy_to_dgl_ndarray(grad_out)
        if self.needs_input_grad[3]:
            grad_src = nd.empty((src_data_nd.shape[0],) + feat_shape,
                                ctx=grad_out.context, dtype=grad_out.dtype)
            K.backward_lhs_src_mul_dst_reduce(
                reducer, binary_op, spmat[0], spmat[1], spmat[2], spmat[3],
                src_map, dst_map, backward_out_map, src_data_nd, dst_data_nd,
                out_data_nd, grad_out_nd, grad_src)
            grad_src = zerocopy_from_dgl_ndarray(grad_src)
            grad_src = _reduce_grad(grad_src, src_data_nd.shape)
        if self.needs_input_grad[4]:
            grad_dst = nd.empty((dst_data_nd.shape[0],) + feat_shape,
                                ctx=grad_out.context, dtype=grad_out.dtype)
            K.backward_rhs_src_mul_dst_reduce(
                reducer, binary_op, spmat[0], spmat[1], spmat[2], spmat[3],
                src_map, dst_map, backward_out_map, src_data_nd, dst_data_nd,
                out_data_nd, grad_out_nd, grad_dst)
            grad_dst = zerocopy_from_dgl_ndarray(grad_dst)
            grad_dst = _reduce_grad(grad_dst, dst_data_nd.shape)
        return None, None, None, grad_src, grad_dst, None, None, None, None


def src_op_dst_reduce(reducer, binary_op, spmat, src_data, dst_data, out_size,
                      src_map, dst_map, out_map):
    func = SrcOpDstReduce()
    return func(reducer, binary_op, spmat, src_data, dst_data, out_size,
                src_map, dst_map, out_map)


class CopySrcReduce(mx.autograd.Function):
    def __init__(self, reducer, spmat, out_size, src_map, out_map):
        super(CopySrcReduce, self).__init__()
        self.reducer = reducer
        self.spmat = spmat
        self.out_size = out_size
        self.src_map = src_map
        if reducer == "none":
            self.forward_out_map = out_map[0]
            self.backward_out_map = out_map[1]
        else:
            self.forward_out_map = out_map
            self.backward_out_map = out_map

    def forward(self, src_data):
        feat_shape = src_data.shape[1:]
        out_data = nd.empty((self.out_size,) + feat_shape,
                            ctx=src_data.context, dtype=src_data.dtype)
        src_data_nd = zerocopy_to_dgl_ndarray(src_data)
        out_data_nd = zerocopy_to_dgl_ndarray_for_write(out_data)
        K.copy_src_reduce(
            self.reducer, self.spmat[0], self.spmat[1], self.spmat[2],
            self.spmat[3], self.src_map, src_data_nd, self.forward_out_map,
            out_data_nd)
        self.save_for_backward(src_data_nd, out_data_nd)
        return out_data

    def backward(self, grad_out):
        src_data_nd, out_data_nd = self.saved_tensors
        grad_out_nd = zerocopy_to_dgl_ndarray(grad_out)
        grad_src = nd.empty(src_data_nd.shape, ctx=grad_out.context,
                            dtype=grad_out.dtype)
        K.backward_copy_src_reduce(
            self.reducer, self.spmat[0], self.spmat[1], self.spmat[2],
            self.spmat[3], self.src_map, self.backward_out_map,
            src_data_nd, out_data_nd, grad_out_nd,
            zerocopy_to_dgl_ndarray_for_write(grad_src))
        return grad_src


def copy_src_reduce(reducer, spmat, src_data, out_size, src_map, out_map):
    func = CopySrcReduce(reducer, spmat, out_size, src_map, out_map)
    return func(src_data)


class CopyEdgeReduce(mx.autograd.Function):
    def __init__(self, reducer, spmat, out_size, edge_map, out_map):
        super(CopyEdgeReduce, self).__init__()
        self.reducer = reducer
        self.spmat = spmat
        self.out_size = out_size
        self.edge_map = edge_map
        if reducer == "none":
            self.forward_out_map = out_map[0]
            self.backward_out_map = out_map[1]
        else:
            self.forward_out_map = out_map
            self.backward_out_map = out_map

    def forward(self, edge_data):
        feat_shape = edge_data.shape[1:]
        out_data = nd.empty((self.out_size,) + feat_shape,
                            ctx=edge_data.context, dtype=edge_data.dtype)
        edge_data_nd = zerocopy_to_dgl_ndarray(edge_data)
        out_data_nd = zerocopy_to_dgl_ndarray_for_write(out_data)
        K.copy_edge_reduce(
            self.reducer, self.spmat[0], self.spmat[1], self.spmat[2],
            self.spmat[3], self.edge_map[0], edge_data_nd,
            self.forward_out_map, out_data_nd)
        self.save_for_backward(edge_data_nd, out_data_nd)
        return out_data

    def backward(self, grad_out):
        edge_data_nd, out_data_nd = self.saved_tensors
        grad_out_nd = zerocopy_to_dgl_ndarray(grad_out)
        grad_edge = nd.empty(edge_data_nd.shape, ctx=grad_out.context,
                             dtype=grad_out.dtype)
        K.backward_copy_edge_reduce(
            self.reducer, self.spmat[0], self.spmat[1], self.spmat[2],
            self.spmat[3], self.edge_map[1], self.backward_out_map,
            edge_data_nd, out_data_nd, grad_out_nd,
            zerocopy_to_dgl_ndarray_for_write(grad_edge))
        return grad_edge


def copy_edge_reduce(reducer, spmat, edge_data, out_size, edge_map, out_map):
    func = CopyEdgeReduce(reducer, spmat, out_size, edge_map, out_map)
    return func(edge_data)


def _reduce_grad(grad, shape):
    grad_shape = grad.shape[1:]
    src_shape = shape[1:]
    if src_shape == grad_shape:
        # no need to reduce
        return grad
    num_to_squeeze = len(grad_shape) - len(src_shape)
    # pad src_shape
    src_shape = (1,) * num_to_squeeze + src_shape
    reduce_idx = np.nonzero(np.array(grad_shape) - np.array(src_shape))[0]
    reduce_idx += 1  # skip batch dim
    grad = grad.sum(axis=tuple(reduce_idx), keepdims=True)
    return grad.reshape(shape)
