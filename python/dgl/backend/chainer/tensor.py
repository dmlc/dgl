from __future__ import absolute_import

import chainer
import chainer.functions as F
import numpy as np
import scipy.sparse as ssp
from ... import ndarray as nd
from ... import kernel as K

from .utils import *
from chainer.backend import CpuDevice, GpuDevice

try:
    import cupy
    import cupy.sparse as cussp
except ImportError:
    cupy = None
    cussp = None

def data_type_dict():
    return {'float16' : np.dtype('float16'),
            'float32' : np.dtype('float32'),
            'float64' : np.dtype('float64'),
            'uint8'   : np.dtype('uint8'),
            'int8'    : np.dtype('int8'),
            'int16'   : np.dtype('int16'),
            'int32'   : np.dtype('int32'),
            'int64'   : np.dtype('int64')}

def cpu():
    return '@numpy'

def tensor(data, dtype=None):
    if dtype is not None or isinstance(data, np.ndarray):
        return chainer.as_variable(np.asarray(data, dtype=dtype))
    else:
        # Chainer defaults to int32 and float64 which is troubling
        data = np.asarray(data)
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype('int64')
        else:
            data = data.astype('float32')
        return chainer.as_variable(data)

def get_preferred_sparse_format():
    return "coo"

def sparse_matrix(data, index, shape, force_format=False):
    fmt = index[0]
    data = data.data
    coord = index[1].data
    if fmt != 'coo':
        raise TypeError('Invalid format: %s.' % fmt)
    spmat = chainer.utils.CooMatrix(data, coord[0], coord[1], shape)
    return spmat, None

def sparse_matrix_indices(spmat):
    return ('csr', spmat.indices, spmat.indptr)

def is_tensor(obj):
    return isinstance(obj, chainer.Variable)

def shape(input):
    return input.shape

def dtype(input):
    return input.dtype

def ndim(input):
    return input.ndim

def context(input):
    return str(input.device)

def _context_of_array(input_array):
    return str(chainer.backend.get_device_from_array(input_array))

def device_type(ctx):
    if is_cpu(ctx):
        return 'cpu'
    elif is_cuda(ctx):
        return 'cuda'
    else:
        raise TypeError('Unknown device: %s.' % ctx)

def device_id(ctx):
    if is_cpu(ctx):
        return 0
    elif is_cuda(ctx):
        return chainer.get_device(ctx).device.id
    else:
        raise TypeError('Unknown device: %s.' % ctx)

def _get_dgl_context(ctx):
    if is_cpu(ctx):
        return nd.cpu()
    elif is_cuda(ctx):
        return nd.gpu(device_id(ctx))
    else:
        raise TypeError('Unknown device: %s.' % ctx)

def astype(input, ty):
    return F.cast(input, ty)

def asnumpy(input):
    input = F.copy(input, -1)
    return input.data

def copy_to(input, ctx):
    if is_cpu(ctx):
        return F.copy(input, -1)
    else:
        return F.copy(input, device_id(ctx))

def sum(input, dim):
    return F.sum(input, axis=dim)

def mean(input, dim):
    return F.mean(input, axis=dim)

def max(input, dim):
    return F.max(input, axis=dim)

def cat(input, dim):
    return F.concat(input, axis=dim)

def stack(input, dim):
    return F.stack(input, axis=dim)

def split(input, sizes_or_sections, dim):
    n_elements = input.shape[dim]
    if isinstance(sizes_or_sections, int):
        return list(F.split_axis(input, sizes_or_sections, dim))
    else:
        indices = tuple(np.cumsum(sizes_or_sections)[:-1])
        return list(F.split_axis(input, indices, dim))

def gather_row(data, row_index):
    # Chainer does not support indexing with tensors?
    return data[row_index.data]

def narrow_row(data, start, stop):
    return data[start:stop]

def scatter_row(data, row_index, value):
    row_index = row_index.data
    return F.scatter_add(data, row_index, value - data[row_index])

def scatter_row_inplace(data, row_index, value):
    if isinstance(value, chainer.Variable):
        data.data[row_index.data] = value.data
    else:
        data.data[row_index.data] = value


def squeeze(input, dim):
    return F.squeeze(input, axis=dim)

def unsqueeze(input, dim):
    return F.expand_dims(input, axis=dim)

def reshape(input, shape):
    return F.reshape(input, shape)

def zeros(shape, dtype, ctx):
    v = chainer.as_variable(
        get_context_module(ctx).zeros(shape, dtype=dtype))
    return v

def zeros_like(input):
    v = chainer.as_variable(
        get_array_module(input).zeros_like(input.data))
    return v

def ones(shape, dtype, ctx):
    v = chainer.as_variable(
        get_context_module(ctx).ones(shape, dtype=dtype))
    return v

def unsorted_1d_segment_sum(input, seg_id, n_segs, dim):
    y = zeros((n_segs,) + input.shape[1:], input.dtype, context(input))
    if dim != 0:
        # Transpose the input so that @dim goes to the first dimension
        axes = list(range(input.ndim))
        axes[dim] = 0
        axes[0] = dim
        input = F.transpose(input, axes)
        y = F.transpose(y, axes)

    y = F.scatter_add(y, seg_id.data, input)

    if dim != 0:
        y = F.transpose(y, axes)

    return y

def unsorted_1d_segment_mean(input, seg_id, n_segs, dim):
    n_ones = chainer.as_variable(
        get_array_module(seg_id).ones_like(seg_id.data).astype(input.dtype))

    w = unsorted_1d_segment_sum(n_ones, seg_id, n_segs, 0)
    w = F.clip(w, 1., np.inf)
    y = unsorted_1d_segment_sum(input, seg_id, n_segs, dim)

    expand_dims = [1] * y.ndim
    expand_dims[dim] = -1
    y = y / F.reshape(w, expand_dims)

    return y

def boolean_mask(input, mask):
    return input[mask.data]

def equal(x, y):
    return chainer.as_variable(x.data == y.data)

def logical_not(input):
    v = chainer.as_variable(
        get_array_module(input).logical_not(input.data))
    return v

def unique(input):
    v = chainer.as_variable(get_array_module(input).unique(input.data))
    return v

def full_1d(length, fill_value, dtype, ctx):
    v = chainer.as_variable(
        get_context_module(ctx).full((length,), fill_value, dtype=dtype))
    return v

def nonzero_1d(input):
    v = chainer.as_variable(
        get_array_module(input).nonzero(input.data)[0])
    return v

def sort_1d(input):
    idx = get_array_module(input).argsort(input.data)
    val = input[idx]
    return val, idx

def arange(start, stop):
    return chainer.as_variable(np.arange(start, stop).astype('int64'))

def rand_shuffle(arr):
    idx = np.random.permutation(arr.shape[0])
    return arr[idx]

# zerocopy_to_dlpack and zerocopy_from_dlpack disabled

def zerocopy_to_numpy(input):
    # NOTE: zerocopy for CPU but not zerocopy for GPU
    return asnumpy(input)

def zerocopy_from_numpy(input):
    return chainer.as_variable(input)

def _zerocopy_to_dgl_ndarray(input_array):
    if isinstance(input_array, np.ndarray):
        return nd.array(input_array)
    elif isinstance(input_array, cupy.ndarray):
        return nd.from_dlpack(input_array.toDlpack())
    else:
        raise TypeError('Unknown type %s.' % type(input_array))

def _zerocopy_from_dgl_ndarray(input):
    if input.ctx.device_type == 1:      # cpu
        return input.asnumpy()
    elif input.ctx.device_type == 2:    # gpu
        return cupy.fromDlpack(input.to_dlpack())
    else:
        raise TypeError('Unsupported device %d' % input.ctx.device_type)

def zerocopy_to_dgl_ndarray(input):
    return _zerocopy_to_dgl_ndarray(input.data)

def zerocopy_from_dgl_ndarray(input):
    return chainer.as_variable(_zerocopy_from_dgl_ndarray(input))


# TODO(quan):
# Currently I could not unify the NumPy-DLPack and CuPy-DLPack interaction
# logic, so I resorted to old-style Chainer functions.
# The new array backend ChainerX currently does not support DLPack.
class BinaryReduce(chainer.Function):
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

    def forward_cpu(self, inputs):
        A, B = inputs

        A_nd = _zerocopy_to_dgl_ndarray(A)
        B_nd = _zerocopy_to_dgl_ndarray(B)
        feat_shape = K.infer_binary_feature_shape(A_nd, B_nd)
        out_nd = nd.empty(
            (self.out_size,) + feat_shape,
            A.dtype,
            _get_dgl_context(_context_of_array(A)))

        K.binary_op_reduce(
            self.reducer, self.op, self.G, self.A_target, self.B_target,
            A_nd, B_nd, out_nd, self.A_rows[0], self.B_rows[0],
            self.out_rows[0])
        # TODO(minjie): (ZEROCOPY_BETWEEN_NUMPY_AND_DGL)
        # For cpu tensors, the DGL NDArrays are **NOT** zero-copied to
        # numpy arrays.  I (Quan) couldn't get zero-copying from numpy
        # to DGL tensors to work.
        out = _zerocopy_from_dgl_ndarray(out_nd)

        self.retain_outputs((0,))
        self.feat_shape = feat_shape

        return out,

    def forward_gpu(self, inputs):
        A, B = inputs

        A_nd = _zerocopy_to_dgl_ndarray(A)
        B_nd = _zerocopy_to_dgl_ndarray(B)
        feat_shape = K.infer_binary_feature_shape(A_nd, B_nd)
        out = cupy.empty((self.out_size,) + feat_shape, dtype=A.dtype)
        out_nd = _zerocopy_to_dgl_ndarray(out)

        K.binary_op_reduce(
            self.reducer, self.op, self.G, self.A_target, self.B_target,
            A_nd, B_nd, out_nd, self.A_rows[0], self.B_rows[0],
            self.out_rows[0])

        self.retain_outputs((0,))
        self.feat_shape = feat_shape

        return out,

    def backward_cpu(self, inputs, grad_outputs):
        A, B = inputs
        out, = self.output_data
        feat_shape = self.feat_shape
        grad_out, = grad_outputs

        A_nd = _zerocopy_to_dgl_ndarray(A)
        B_nd = _zerocopy_to_dgl_ndarray(B)
        out_nd = _zerocopy_to_dgl_ndarray(out)
        grad_out_nd = _zerocopy_to_dgl_ndarray(grad_out)

        grad_A_nd = nd.empty((A_nd.shape[0],) + feat_shape, A.dtype, nd.cpu())
        K.backward_lhs_binary_op_reduce(
            self.reducer, self.op, self.G, self.A_target, self.B_target,
            A_nd, B_nd, out_nd, grad_out_nd, grad_A_nd,
            self.A_rows[1], self.B_rows[1], self.out_rows[1])
        # TODO (ZEROCOPY_BETWEEN_NUMPY_AND_DGL)
        grad_A = _zerocopy_from_dgl_ndarray(grad_A_nd)
        grad_A = _reduce_grad(grad_A, A_nd.shape)

        grad_B_nd = nd.empty((B_nd.shape[0],) + feat_shape, B.dtype, nd.cpu())
        K.backward_rhs_binary_op_reduce(
            self.reducer, self.op, self.G, self.A_target, self.B_target,
            A_nd, B_nd, out_nd, grad_out_nd, grad_B_nd,
            self.A_rows[1], self.B_rows[1], self.out_rows[1])
        # TODO (ZEROCOPY_BETWEEN_NUMPY_AND_DGL)
        grad_B = _zerocopy_from_dgl_ndarray(grad_B_nd)
        grad_B = _reduce_grad(grad_B, B_nd.shape)

        return grad_A, grad_B

    def backward_gpu(self, inputs, grad_outputs):
        A, B = inputs
        out, = self.output_data
        feat_shape = self.feat_shape
        grad_out, = grad_outputs
        # TODO (CUPY_COPY)
        # The gradient CuPy array behaves weird with DLPack and DGL currently:
        # without the copy, printing grad_out_nd (zerocopy of grad_out) will
        # throw a CUDA invalid argument error, and the gradient computation
        # will be incorrect.  I'm leaving the inspection for future.
        grad_out = cupy.copy(grad_out)

        A_nd = _zerocopy_to_dgl_ndarray(A)
        B_nd = _zerocopy_to_dgl_ndarray(B)
        out_nd = _zerocopy_to_dgl_ndarray(out)
        grad_out_nd = _zerocopy_to_dgl_ndarray(grad_out)

        grad_A = cupy.empty((A_nd.shape[0],) + feat_shape, dtype=A.dtype)
        grad_A_nd = _zerocopy_to_dgl_ndarray(grad_A)
        K.backward_lhs_binary_op_reduce(
            self.reducer, self.op, self.G, self.A_target, self.B_target,
            A_nd, B_nd, out_nd, grad_out_nd, grad_A_nd,
            self.A_rows[1], self.B_rows[1], self.out_rows[1])
        grad_A = _reduce_grad(grad_A, A_nd.shape)

        grad_B = cupy.empty((B_nd.shape[0],) + feat_shape, dtype=B.dtype)
        grad_B_nd = _zerocopy_to_dgl_ndarray(grad_B)
        K.backward_rhs_binary_op_reduce(
            self.reducer, self.op, self.G, self.A_target, self.B_target,
            A_nd, B_nd, out_nd, grad_out_nd, grad_B_nd,
            self.A_rows[1], self.B_rows[1], self.out_rows[1])
        grad_B = _reduce_grad(grad_B, B_nd.shape)

        return grad_A, grad_B


# pylint: disable=invalid-name
def binary_reduce(reducer, op, G, A_target, B_target, A, B,
                  out_size, A_rows, B_rows, out_rows):
    func = BinaryReduce(reducer, op, G, A_target, B_target, out_size, A_rows,
                        B_rows, out_rows)
    return func(A, B)


class CopyReduce(chainer.Function):
    # pylint: disable=invalid-name
    def __init__(self, reducer, G, target, out_size, X_rows, out_rows):
        super(CopyReduce, self).__init__()
        self.reducer = reducer
        self.G = G
        self.target = target
        self.out_size = out_size
        self.X_rows = X_rows
        self.out_rows = out_rows

    def forward_cpu(self, inputs):
        X, = inputs
        feat_shape = X.shape[1:]

        X_nd = _zerocopy_to_dgl_ndarray(X)
        out_nd = nd.empty((self.out_size,) + feat_shape, X.dtype, nd.cpu())

        K.copy_reduce(
            self.reducer, self.G, self.target, X_nd, out_nd,
            self.X_rows[0], self.out_rows[0])
        # TODO (ZEROCOPY_BETWEEN_NUMPY_AND_DGL)
        out = _zerocopy_from_dgl_ndarray(out_nd)

        self.retain_outputs((0,))

        return out,

    def forward_gpu(self, inputs):
        X, = inputs
        feat_shape = X.shape[1:]

        X_nd = _zerocopy_to_dgl_ndarray(X)
        out = cupy.empty((self.out_size,) + feat_shape, dtype=X.dtype)
        out_nd = _zerocopy_to_dgl_ndarray(out)

        K.copy_reduce(
            self.reducer, self.G, self.target, X_nd, out_nd,
            self.X_rows[0], self.out_rows[0])

        self.retain_outputs((0,))

        return out,

    def backward_cpu(self, inputs, grad_outputs):
        X, = inputs
        out, = self.output_data
        grad_out, = grad_outputs

        X_nd = _zerocopy_to_dgl_ndarray(X)
        out_nd = _zerocopy_to_dgl_ndarray(out)
        grad_out_nd = _zerocopy_to_dgl_ndarray(grad_out)

        grad_X_nd = nd.empty(X.shape, X.dtype, nd.cpu())
        K.backward_copy_reduce(
            self.reducer, self.G, self.target, X_nd, out_nd,
            grad_out_nd, grad_X_nd,
            self.X_rows[1], self.out_rows[1])
        # TODO (ZEROCOPY_BETWEEN_NUMPY_AND_DGL)
        grad_X = _zerocopy_from_dgl_ndarray(grad_X_nd)

        return grad_X,

    def backward_gpu(self, inputs, grad_outputs):
        X, = inputs
        out, = self.output_data
        grad_out, = grad_outputs
        # TODO (CUPY_COPY)
        grad_out = cupy.copy(grad_out)

        X_nd = _zerocopy_to_dgl_ndarray(X)
        out_nd = _zerocopy_to_dgl_ndarray(out)
        grad_out_nd = _zerocopy_to_dgl_ndarray(grad_out)

        grad_X = cupy.empty(X.shape, dtype=X.dtype)
        grad_X_nd = _zerocopy_to_dgl_ndarray(grad_X)
        K.backward_copy_reduce(
            self.reducer, self.G, self.target, X_nd, out_nd,
            grad_out_nd, grad_X_nd,
            self.X_rows[1], self.out_rows[1])

        return grad_X,


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
    grad: numpy.ndarray or cupy.ndarray
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
    reduce_idx += 1 # skip batch dim
    grad = grad.sum(axis=tuple(reduce_idx), keepdims=True)
    return grad.reshape(shape)


def sync():
    # Chainer performs computation synchronously.
    pass
