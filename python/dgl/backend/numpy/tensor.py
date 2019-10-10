from __future__ import absolute_import

import numpy as onp
from mxnet import numpy as np
import mxnet as mx
import scipy.sparse as sp
import warnings
from ... import ndarray as dglnd
from ... import kernel as K

def data_type_dict():
    return {'float16' : onp.float16,
            'float32' : onp.float32,
            'float64' : onp.float64,
            'uint8'   : onp.uint8,
            'int8'    : onp.int8,
            'int16'   : onp.int16,
            'int32'   : onp.int32,
            'int64'   : onp.int64}

def cpu():
    return mx.cpu()

def tensor(data, dtype=None):
    return np.array(data, dtype)

def as_scalar(data):
    if data.dim() > 1:
        raise ValueError('The data must have shape (1,).')
    return data[0]

def get_preferred_sparse_format():
    """Get the preferred sparse matrix format supported by the backend.

    Different backends have their preferred backend. This info is useful when
    constructing a sparse matrix.
    """
    return "csr"

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
    return input.context

def device_type(ctx):
    return ctx.device_type

def device_id(ctx):
    return ctx.device_id

def astype(input, ty):
    return input.astype(ty)

def asnumpy(input):
    return input.asnumpy()

def copy_to(input, ctx):
    return input.as_in_context(ctx)

def sum(input, dim):
    return np.sum(input, axis=dim)

def reduce_sum(input):
    dtype = input.dtype
    return np.array(input.sum(), dtype=dtype)

def mean(input, dim):
    return np.mean(input, axis=dim)

def reduce_mean(input):
    dtype = input.dtype
    return np.array(input.mean(), dtype=dtype)

def max(input, dim):
    return np.max(input, axis=dim)

def reduce_max(input):
    dtype = input.dtype
    return np.array(input.max(), dtype=dtype)

def min(input, dim):
    return np.min(input, axis=dim)

def reduce_min(input):
    dtype = input.dtype
    return np.array(input.min(), dtype=dtype)

def argsort(input, dim, descending):
    if descending:
        return np.argsort(-input, axis=dim)
    return np.argsort(input, axis=dim)

def exp(input):
    return np.exp(input)

def softmax(input, dim=-1):
    max_val = input.max(axis=dim)
    minus_max = input - np.expand_dims(max_val, axis=dim)
    exp_val = np.exp(minus_max)
    sum_val = np.sum(exp_val, axis=dim)
    return exp_val / np.expand_dims(sum_val, axis=dim)

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
        idx = np.cumsum(np.array(sizes_or_sections))[0:-1]
    return np.split(input, idx, axis=dim)

def repeat(input, repeats, dim):
    return np.repeat(input, repeats, axis=dim)

def gather_row(data, row_index):
    return data[row_index]

def slice_axis(data, axis, begin, end):
    if begin >= end:
        raise IndexError("Begin index ({}) equals or greater than end index ({})".format(begin, end))
    return np.take(data, np.arange(begin, end), axis=axis)

def take(data, indices, dim):
    return np.take(data, indices, axis=dim)

def narrow_row(data, start, stop):
    return data[start:stop]

def scatter_row(data, row_index, value):
    data = data.as_nd_ndarray()
    row_index = row_index.as_nd_ndarray()
    value = value.as_nd_ndarray()
    ret = mx.nd.contrib.index_copy(data, row_index, value)
    return ret.as_np_ndarray()

def scatter_row_inplace(data, row_index, value):
    data[row_index] = value

def squeeze(input, dim):
    return np.squeeze(input, dim)

def unsqueeze(input, dim):
    return np.unsqueeze(input, dim)

def reshape(input, shape):
    return np.reshape(input ,shape)

def zeros(shape, dtype, ctx):
    return np.zeros(shape, dtype=dtype, ctx=ctx)

def ones(shape, dtype, ctx):
    return np.ones(shape, dtype=dtype, ctx=ctx)

def unsorted_1d_segment_sum(input, seg_id, n_segs, dim):
    #TODO
    pass

def unsorted_1d_segment_mean(input, seg_id, n_segs, dim):
    #TODO
    pass

def boolean_mask(input, mask):
    input = input.as_nd_ndarray()
    mask = mask.as_nd_ndarray()
    ret = mx.contrib.nd.boolean_mask(input, mask)
    return ret.as_np_ndarray()

def equal(x, y):
    return x == y

def logical_not(input):
    return np.logical_not(input)

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

def zerocopy_to_dlpack(arr):
    return arr.to_dlpack_for_read()

def zerocopy_from_dlpack(dlpack_arr):
    return mx.nd.from_dlpack(dlpack_arr).as_np_ndarray()

def zerocopy_to_numpy(arr):
    # NOTE: not zerocopy
    return arr.asnumpy()

def zerocopy_from_numpy(np_data):
    return mx.nd.from_numpy(np_data, zero_copy=True).as_np_ndarray()

def one_hot(t, num_classes=-1):
    if num_classes == -1:
        num_classes = np.max(t) + 1

    t = t.as_nd_ndarray()
    ret = mx.nd.one_hot(t, num_classes)
    return ret.as_np_ndarray()

def zerocopy_to_dgl_ndarray(arr):
    return dglnd.from_dlpack(arr.to_dlpack_for_read())

def zerocopy_to_dgl_ndarray_for_write(arr):
    return dglnd.from_dlpack(arr.to_dlpack_for_write())

def zerocopy_from_dgl_ndarray(arr):
    return np.from_dlpack(arr.to_dlpack())

class BinaryReduce(mx.autograd.Function):
    def __init__(self, reducer, binary_op, graph, lhs, rhs, out_size, lhs_map,
                 rhs_map, out_map):
        super(BinaryReduce, self).__init__()
        self.reducer = reducer
        self.binary_op = binary_op
        self.graph = graph
        self.lhs = lhs
        self.rhs = rhs
        self.out_size = out_size
        self.lhs_map = lhs_map
        self.rhs_map = rhs_map
        self.out_map = out_map

    def forward(self, lhs_data, rhs_data):
        lhs_data_nd = zerocopy_to_dgl_ndarray(lhs_data)
        rhs_data_nd = zerocopy_to_dgl_ndarray(rhs_data)
        feat_shape = K.infer_binary_feature_shape(lhs_data_nd, rhs_data_nd)
        out_data = np.empty((self.out_size,) + feat_shape,
                            ctx=lhs_data.context, dtype=lhs_data.dtype)
        out_data_nd = zerocopy_to_dgl_ndarray_for_write(out_data)
        K.binary_op_reduce(
            self.reducer, self.binary_op, self.graph, self.lhs, self.rhs,
            lhs_data_nd, rhs_data_nd, out_data_nd, self.lhs_map[0],
            self.rhs_map[0], self.out_map[0])
        self.save_for_backward(lhs_data_nd, rhs_data_nd, out_data_nd,
                               feat_shape)
        return out_data

    def backward(self, grad_out):
        lhs_data_nd, rhs_data_nd, out_data_nd, feat_shape = self.saved_tensors
        grad_out_nd = zerocopy_to_dgl_ndarray(grad_out)
        grad_lhs = np.empty((lhs_data_nd.shape[0],) + feat_shape,
                            ctx=grad_out.context, dtype=grad_out.dtype)
        K.backward_lhs_binary_op_reduce(
            self.reducer, self.binary_op, self.graph, self.lhs, self.rhs,
            lhs_data_nd, rhs_data_nd, out_data_nd, grad_out_nd,
            zerocopy_to_dgl_ndarray_for_write(grad_lhs), self.lhs_map[1],
            self.rhs_map[1], self.out_map[1])
        grad_lhs = _reduce_grad(grad_lhs, lhs_data_nd.shape)
        grad_rhs = np.empty((rhs_data_nd.shape[0],) + feat_shape,
                             ctx=grad_out.context, dtype=grad_out.dtype)
        K.backward_rhs_binary_op_reduce(
            self.reducer, self.binary_op, self.graph, self.lhs, self.rhs,
            lhs_data_nd, rhs_data_nd, out_data_nd, grad_out_nd,
            zerocopy_to_dgl_ndarray_for_write(grad_rhs), self.lhs_map[1],
            self.rhs_map[1], self.out_map[1])
        grad_rhs = _reduce_grad(grad_rhs, rhs_data_nd.shape)
        # clear saved tensors explicitly
        self.saved_tensors = None
        return grad_lhs, grad_rhs


def binary_reduce(reducer, binary_op, graph, lhs, rhs, lhs_data, rhs_data,
                  out_size, lhs_map, rhs_map, out_map):
    func = BinaryReduce(reducer, binary_op, graph, lhs, rhs, out_size, lhs_map,
                        rhs_map, out_map)
    return func(lhs_data, rhs_data)


class CopyReduce(mx.autograd.Function):
    def __init__(self, reducer, graph, target, out_size, in_map, out_map):
        super(CopyReduce, self).__init__()
        self.reducer = reducer
        self.graph = graph
        self.target = target
        self.out_size = out_size
        self.in_map = in_map
        self.out_map = out_map

    def forward(self, in_data):
        feat_shape = in_data.shape[1:]
        out_data = np.empty((self.out_size,) + feat_shape,
                            ctx=in_data.context, dtype=in_data.dtype)
        in_data_nd = zerocopy_to_dgl_ndarray(in_data)
        out_data_nd = zerocopy_to_dgl_ndarray_for_write(out_data)
        K.copy_reduce(
            self.reducer, self.graph, self.target, in_data_nd, out_data_nd,
            self.in_map[0], self.out_map[0])
        self.save_for_backward(in_data_nd, out_data_nd)
        return out_data

    def backward(self, grad_out):
        in_data_nd, out_data_nd = self.saved_tensors
        grad_out_nd = zerocopy_to_dgl_ndarray(grad_out)
        grad_in = np.empty(in_data_nd.shape, ctx=grad_out.context,
                            dtype=grad_out.dtype)
        K.backward_copy_reduce(
            self.reducer, self.graph, self.target, in_data_nd, out_data_nd,
            grad_out_nd, zerocopy_to_dgl_ndarray_for_write(grad_in),
            self.in_map[1], self.out_map[1])
        # clear saved tensors explicitly
        self.saved_tensors = None
        return grad_in


def copy_reduce(reducer, graph, target, in_data, out_size, in_map, out_map):
    func = CopyReduce(reducer, graph, target, out_size, in_map, out_map)
    return func(in_data)


def _reduce_grad(grad, shape):
    """Reduce gradient on the broadcast dimension

    If there is broadcast in forward pass, gradients need to be reduced on
    broadcast dimension. This function checks the input tensor shape and
    gradient shape and perform the reduction.

    Parameters
    ----------
    grad: Tensor
        Gradient tensor
    shape: tuple
        Shape of input tensor

    Returns
    -------
    Tensor
    """
    grad_shape = grad.shape[1:]
    in_shape = shape[1:]
    if in_shape == grad_shape:
        # no need to reduce
        return grad
    num_to_squeeze = len(grad_shape) - len(in_shape)
    # pad in_shape
    in_shape = (1,) * num_to_squeeze + in_shape
    reduce_idx = np.nonzero(np.array(grad_shape) - np.array(in_shape))[0]
    reduce_idx += 1  # skip batch dim
    grad = grad.sum(axis=tuple(reduce_idx), keepdims=True)
    return grad.reshape(shape)

def sync():
    """Synchronize computation.

    In DL frameworks such as MXNet and TensorFlow, the computation in operators
    are done asynchronously. This is to synchronize computation and makes sure
    that all computation is complete after this function call.
    """
    mx.nd.waitall()
