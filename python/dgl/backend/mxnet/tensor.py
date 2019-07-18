from __future__ import absolute_import

from distutils.version import LooseVersion

import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
import numbers
from ... import ndarray as dglnd
from ... import kernel as K

MX_VERSION = LooseVersion(mx.__version__)
# After MXNet 1.5, empty tensors aren't supprted by default.
# after we turn on the numpy compatible flag, MXNet supports empty NDArray.
if MX_VERSION.version[0] == 1 and MX_VERSION.version[1] >= 5:
    mx.set_np_shape(True)

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
    if isinstance(sizes_or_sections, list) and len(sizes_or_sections) == 1:
        assert len(x) == sizes_or_sections[0]
        return [x]

    if MX_VERSION.version[0] == 1 and MX_VERSION.version[1] >= 5:
        if isinstance(sizes_or_sections, (np.ndarray, list)):
            sizes_or_sections1 = tuple(np.cumsum(sizes_or_sections)[:-1])
        return nd.split_v2(x, sizes_or_sections1, axis=dim)

    if isinstance(sizes_or_sections, list) or isinstance(sizes_or_sections, np.ndarray):
        # Old MXNet doesn't support split with different section sizes.
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
    return mx.nd.from_numpy(np_data, zero_copy=True)

def zerocopy_to_dgl_ndarray(arr):
    return dglnd.from_dlpack(arr.to_dlpack_for_read())

def zerocopy_to_dgl_ndarray_for_write(arr):
    return dglnd.from_dlpack(arr.to_dlpack_for_write())

def zerocopy_from_dgl_ndarray(arr):
    return nd.from_dlpack(arr.to_dlpack())


class BinaryReduce(mx.autograd.Function):
    # pylint: disable=invalid-name
    def __init__(self, reducer, op, G, A_target, B_target, out_size, A_rows,
                 B_rows, out_rows):
        super(BinaryReduce, self).__init__()
        self.reducer = reducer
        self.op = op
        self.G = G
        self.A_target = A_target
        self.B_target = B_target
        self.out_size = out_size
        self.A_rows = A_rows
        self.B_rows = B_rows
        self.out_rows = out_rows

    # pylint: disable=invalid-name
    def forward(self, A, B):
        A_nd = zerocopy_to_dgl_ndarray(A)
        B_nd = zerocopy_to_dgl_ndarray(B)
        feat_shape = K.infer_binary_feature_shape(A_nd, B_nd)
        out = nd.empty((self.out_size,) + feat_shape,
                       ctx=A.context, dtype=A.dtype)
        out_nd = zerocopy_to_dgl_ndarray_for_write(out)
        K.binary_op_reduce(
            self.reducer, self.op, self.G, self.A_target, self.B_target,
            A_nd, B_nd, out_nd, self.A_rows[0],
            self.B_rows[0], self.out_rows[0])
        self.save_for_backward(A_nd, B_nd, out_nd,
                               feat_shape)
        return out

    def backward(self, grad_out):
        A_nd, B_nd, out_nd, feat_shape = self.saved_tensors
        grad_out_nd = zerocopy_to_dgl_ndarray(grad_out)
        grad_A = nd.empty((A_nd.shape[0],) + feat_shape,
                          ctx=grad_out.context, dtype=grad_out.dtype)
        K.backward_lhs_binary_op_reduce(
            self.reducer, self.op, self.G, self.A_target, self.B_target,
            A_nd, B_nd, out_nd, grad_out_nd,
            zerocopy_to_dgl_ndarray_for_write(grad_A), self.A_rows[1],
            self.B_rows[1], self.out_rows[1])
        grad_A = _reduce_grad(grad_A, A_nd.shape)
        grad_B = nd.empty((B_nd.shape[0],) + feat_shape,
                          ctx=grad_out.context, dtype=grad_out.dtype)
        K.backward_rhs_binary_op_reduce(
            self.reducer, self.op, self.G, self.A_target, self.B_target,
            A_nd, B_nd, out_nd, grad_out_nd,
            zerocopy_to_dgl_ndarray_for_write(grad_B), self.A_rows[1],
            self.B_rows[1], self.out_rows[1])
        grad_B = _reduce_grad(grad_B, B_nd.shape)
        # clear saved tensors explicitly
        self.saved_tensors = None
        return grad_A, grad_B


# pylint: disable=invalid-name
def binary_reduce(reducer, op, G, A_target, B_target, A, B,
                  out_size, A_rows, B_rows, out_rows):
    func = BinaryReduce(reducer, op, G, A_target, B_target, out_size, A_rows,
                        B_rows, out_rows)
    return func(A, B)


class CopyReduce(mx.autograd.Function):
    # pylint: disable=invalid-name
    def __init__(self, reducer, G, target, out_size, X_rows, out_rows):
        super(CopyReduce, self).__init__()
        self.reducer = reducer
        self.G = G
        self.target = target
        self.out_size = out_size
        self.X_rows = X_rows
        self.out_rows = out_rows

    # pylint: disable=invalid-name
    def forward(self, X):
        feat_shape = X.shape[1:]
        out = nd.empty((self.out_size,) + feat_shape,
                       ctx=X.context, dtype=X.dtype)
        X_nd = zerocopy_to_dgl_ndarray(X)
        out_nd = zerocopy_to_dgl_ndarray_for_write(out)
        K.copy_reduce(
            self.reducer, self.G, self.target, X_nd, out_nd,
            self.X_rows[0], self.out_rows[0])
        self.save_for_backward(X_nd, out_nd)
        return out

    def backward(self, grad_out):
        X_nd, out_nd = self.saved_tensors
        grad_out_nd = zerocopy_to_dgl_ndarray(grad_out)
        grad_X = nd.empty(X_nd.shape, ctx=grad_out.context,
                          dtype=grad_out.dtype)
        K.backward_copy_reduce(
            self.reducer, self.G, self.target, X_nd, out_nd,
            grad_out_nd, zerocopy_to_dgl_ndarray_for_write(grad_X),
            self.X_rows[1], self.out_rows[1])
        # clear saved tensors explicitly
        self.saved_tensors = None
        return grad_X


# pylint: disable=invalid-name
def copy_reduce(reducer, G, target, X, out_size, X_rows, out_rows):
    func = CopyReduce(reducer, G, target, out_size, X_rows, out_rows)
    return func(X)


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
