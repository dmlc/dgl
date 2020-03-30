"""Tensorflow backend implementation"""
from __future__ import absolute_import

from distutils.version import LooseVersion

import tensorflow as tf
from tensorflow.python.eager import context
import builtins
import numpy as np
import os

from ... import ndarray as nd
from ... import kernel as K
from ...function.base import TargetCode

if os.getenv("USE_OFFICIAL_TFDLPACK", False):
    if LooseVersion(tf.__version__) < LooseVersion("2.2.0"):
        raise RuntimeError("DGL requires tensorflow>=2.2.0 for the official DLPack support.")

    def zerocopy_to_dlpack(input):
        return tf.experimental.dlpack.to_dlpack(input)

    def zerocopy_from_dlpack(dlpack_tensor):
        # TODO(Jinjing): Tensorflow requires memory to be 64-bit aligned. We check the
        #   alignment and make a copy if needed. The functionality is better in TF's main repo.
        aligned = nd.from_dlpack(dlpack_tensor).to_dlpack(64)
        return tf.experimental.dlpack.from_dlpack(aligned)
else:
    # Use our own DLPack solution
    try:
        import tfdlpack
    except ImportError:
        raise ImportError('Cannot find tfdlpack, which is required by the Tensorflow backend. '
                          'Please follow https://github.com/VoVAllen/tf-dlpack for installation.')

    if LooseVersion(tf.__version__) < LooseVersion("2.1.0"):
        raise RuntimeError("DGL requires tensorflow>=2.1.0.")

    def zerocopy_to_dlpack(input):
        return tfdlpack.to_dlpack(input)

    def zerocopy_from_dlpack(input):
        return tfdlpack.from_dlpack(input)

def data_type_dict():
    return {'float16': tf.float16,
            'float32': tf.float32,
            'float64': tf.float64,
            'uint8': tf.uint8,
            'int8': tf.int8,
            'int16': tf.int16,
            'int32': tf.int32,
            'int64': tf.int64}

def cpu():
    return "/cpu:0"

def tensor(data, dtype=None):
    return tf.convert_to_tensor(data, dtype=dtype)


def as_scalar(data):
    return data.numpy().asscalar()


def get_preferred_sparse_format():
    """Get the preferred sparse matrix format supported by the backend.

    Different backends have their preferred backend. This info is useful when
    constructing a sparse matrix.
    """
    return "coo"


def sparse_matrix(data, index, shape, force_format=False):
    fmt = index[0]
    if fmt != 'coo':
        raise TypeError(
            'Tensorflow backend only supports COO format. But got %s.' % fmt)
    spmat = tf.SparseTensor(indices=tf.transpose(
        index[1], (1, 0)), values=data, dense_shape=shape)
    return spmat, None


def sparse_matrix_indices(spmat):
    return ('coo', spmat.indices)


def is_tensor(obj):
    return isinstance(obj, tf.Tensor)


def shape(input):
    return input.shape


def dtype(input):
    return input.dtype


def ndim(input):
    return input.ndim


def context(input):
    return input.device


def device_type(ctx):
    return tf.DeviceSpec.from_string(ctx).device_type.lower()


def device_id(ctx):
    return tf.DeviceSpec.from_string(ctx).device_index


def astype(input, ty):
    return tf.cast(input, dtype=ty)


def asnumpy(input):
    if isinstance(input, tf.SparseTensor):
        # tf.sparse.to_dense assume sorted indices, need to turn off validate_indices in our cases
        return tf.sparse.to_dense(input, validate_indices=False).numpy()
    else:
        return input.numpy()


def copy_to(input, ctx):
    with tf.device(ctx):
        new_tensor = tf.identity(input)
    return new_tensor


def sum(input, dim, keepdims=False):
    return tf.reduce_sum(input, axis=dim, keepdims=keepdims)


def reduce_sum(input):
    return tf.reduce_sum(input)


def mean(input, dim):
    return tf.reduce_mean(input, axis=dim)


def reduce_mean(input):
    return th.reduce_mean(input)


def max(input, dim):
    return tf.reduce_max(input, axis=dim)


def reduce_max(input):
    return tf.reduce_max(input)


def min(input, dim):
    return tf.reduce_min(input, axis=dim)


def reduce_min(input):
    return tf.reduce_min(input)


def argsort(input, dim, descending):
    if descending:
        return tf.cast(tf.argsort(input, axis=dim, direction="DESCENDING"), dtype=tf.int64)
    else:
        return tf.cast(tf.argsort(input, axis=dim, direction="ASCENDING"), dtype=tf.int64)


def topk(input, k, dim, descending=True):
    if not descending:
        input = -input
    shape = np.arange(input.ndim)
    shape[dim], shape[-1] = shape[-1], shape[dim]
    out1 = tf.transpose(input, perm=shape)
    out2 = tf.math.top_k(out1, k=k, sorted=True)
    out = tf.transpose(out2[0], shape)
    if not descending:
        out = -out
    return out


def argtopk(input, k, dim, descending=True):
    if not descending:
        input = -input
    shape = np.arange(input.ndim)
    shape[dim], shape[-1] = shape[-1], shape[dim]
    out1 = tf.transpose(input, perm=shape)
    out2 = tf.math.top_k(out1, k=k, sorted=True)
    out = tf.transpose(out2[1], shape)
    if not descending:
        out = -out
    return out


def exp(input):
    return tf.exp(input)


def softmax(input, dim=-1):
    return tf.math.softmax(input, axis=dim)


def cat(seq, dim):
    return tf.concat(seq, axis=dim)


def stack(seq, dim):
    return tf.stack(seq, axis=dim)


def split(input, sizes_or_sections, dim):
    return tf.split(input, sizes_or_sections, axis=dim)


def repeat(input, repeats, dim):
    return tf.keras.backend.repeat_elements(input, repeats, dim)


def gather_row(data, row_index):
    return tf.gather(data, row_index)


def slice_axis(data, axis, begin, end):
    # assert axis == 0
    # tf doesn't behave well with negative
    s = [slice(None) for i in range(data.ndim)]
    if end == 0:
        end = data.shape[axis]
    s[axis] = slice(begin, end, None)
    return data[tuple(s)]


def take(data, indices, dim):
    return tf.gather_nd(data, indices, dim)


def narrow_row(x, start, stop):
    return x[start:stop]


def scatter_row(data, row_index, value):
    row_index = tf.expand_dims(row_index, 1)
    return tf.tensor_scatter_nd_update(data, row_index, value)


def scatter_row_inplace(data, row_index, value):
    raise NotImplementedError("Tensorflow doesn't support inplace update")


def squeeze(input, dim):
    return tf.squeeze(input, axis=dim)


def unsqueeze(input, dim):
    return tf.expand_dims(input, axis=dim)


def reshape(input, shape):
    return tf.reshape(input, shape)


def swapaxes(input, axis1, axis2):
    return tf.transpose(input, perm=[axis1, axis2])


def zeros(shape, dtype, ctx):
    with tf.device(ctx):
        t = tf.zeros(shape, dtype=dtype)
    return t


def zeros_like(input):
    return tf.zeros_like(input)


def ones(shape, dtype, ctx):
    with tf.device(ctx):
        t = tf.ones(shape, dtype=dtype)
    return t


def uniform(shape, dtype, ctx, low, high):
    with tf.device(ctx):
        t = tf.random.uniform(shape, dtype=dtype, minval=low, maxval=high)
    return t


def pad_packed_tensor(input, lengths, value, l_min=None):
    old_shape = input.shape
    if isinstance(lengths, tf.Tensor):
        max_len = as_scalar(lengths.max())
    else:
        max_len = builtins.max(lengths)

    if l_min is not None:
        max_len = builtins.max(max_len, l_min)

    batch_size = len(lengths)
    ndim = input.ndim
    tensor_list = []
    cum_row = 0
    pad_nparray = np.zeros((ndim, 2), dtype=np.int32)
    for l in lengths:
        t = input[cum_row:cum_row+l]
        pad_nparray[0, 1] = max_len - l
        t = tf.pad(t, tf.constant(pad_nparray),
                   mode='CONSTANT', constant_values=value)
        tensor_list.append(t)
        cum_row += l
    return tf.stack(tensor_list, axis=0)


def pack_padded_tensor(input, lengths):
    out_list = []
    for i, l in enumerate(lengths):
        t = input[i]
        out = t[:l]
        out_list.append(out)
    return tf.concat(out_list, axis=0)


def unsorted_1d_segment_sum(input, seg_id, n_segs, dim):
    assert dim == 0  # Why we need dim for 1d?
    return tf.math.unsorted_segment_sum(input, seg_id, n_segs)


def unsorted_1d_segment_mean(input, seg_id, n_segs, dim):
    assert dim == 0  # Why we need dim for 1d?
    return tf.math.unsorted_segment_mean(input, seg_id, n_segs)

# TODO: TF has unsorted_segment_max, which can accelerate _max_on on batched graph


def boolean_mask(input, mask):
    return tf.boolean_mask(input, mask)


def equal(x, y):
    return x == y


def logical_not(input):
    return ~input


def unique(input):
    return tf.unique(input).y


def full_1d(length, fill_value, dtype, ctx):
    with tf.device(ctx):
        t = tf.fill([length], value=fill_value)
        t = tf.cast(t, dtype=dtype)
    return t


def nonzero_1d(input):
    nonzero_bool = (input != False)
    return tf.reshape(tf.where(nonzero_bool), (-1, ))


def sort_1d(input):
    return tf.sort(input), tf.cast(tf.argsort(input), dtype=tf.int64)


def arange(start, stop):
    with tf.device("/cpu:0"):
        t = tf.range(start, stop, dtype=tf.int64)
    return t


def rand_shuffle(arr):
    return tf.random.shuffle(arr)


def zerocopy_to_numpy(input):
    return np.asarray(memoryview(input))


def zerocopy_from_numpy(np_array):
    # NOTE: not zerocopy
    # This assumes tensor should be on cpu
    with tf.device("/cpu:0"):
        t = tf.convert_to_tensor(np_array)
    return t


def zerocopy_to_dgl_ndarray(input):
    return nd.from_dlpack(zerocopy_to_dlpack(input))


def zerocopy_from_dgl_ndarray(input):
    return zerocopy_from_dlpack(input.to_dlpack())


def binary_reduce(reducer, binary_op, graph, lhs, rhs, lhs_data, rhs_data,
                  out_size, lhs_map=(None, None), rhs_map=(None, None), out_map=(None, None)):

    @tf.custom_gradient
    def _lambda(lhs_data, rhs_data):
        return binary_reduce_real(reducer, binary_op, graph, lhs, rhs, lhs_data, rhs_data,
                                  out_size, lhs_map, rhs_map, out_map)
    return _lambda(lhs_data, rhs_data)


def binary_reduce_real(reducer, binary_op, graph, lhs, rhs, lhs_data, rhs_data,
                       out_size, lhs_map, rhs_map, out_map):
    lhs_data_nd = zerocopy_to_dgl_ndarray(lhs_data)
    rhs_data_nd = zerocopy_to_dgl_ndarray(rhs_data)
    feat_shape = K.infer_binary_feature_shape(
        binary_op, lhs_data_nd, rhs_data_nd)
    out_shape = feat_shape
    if binary_op == 'dot':
        out_shape = feat_shape[:-1]
    # out_data = lhs_data.new_empty((out_size,) + out_shape)
    out_data = tf.zeros((out_size,) + out_shape, dtype=lhs_data.dtype)
    out_data_nd = zerocopy_to_dgl_ndarray(out_data)
    K.binary_op_reduce(
        reducer if reducer != 'mean' else 'sum',
        binary_op, graph, lhs, rhs, lhs_data_nd, rhs_data_nd,
        out_data_nd, lhs_map[0], rhs_map[0], out_map[0])
    # normalize if mean reducer
    # NOTE(zihao): this is a temporary hack and we should have better solution in the future.
    if reducer == 'mean':
        # degs = lhs_data.new_empty((out_data.shape[0],))
        degs = tf.zeros((out_data.shape[0],), dtype=lhs_data.dtype)
        degs_nd = zerocopy_to_dgl_ndarray(degs)
        if lhs != TargetCode.DST:  # src or edge
            target = lhs
            n = lhs_data.shape[0]
            in_map = lhs_map[0]
        else:  # rhs != TargetCode.DST
            target = rhs
            n = rhs_data.shape[0]
            in_map = rhs_map[0]
        # in_ones = lhs_data.new_ones((n,))
        in_ones = tf.ones((n,), dtype=lhs_data.dtype)
        in_ones_nd = zerocopy_to_dgl_ndarray(in_ones)
        K.copy_reduce(
            'sum', graph, target, in_ones_nd, degs_nd, in_map, out_map[0])
        # reshape
        degs = tf.reshape(degs,
                          (out_data.shape[0],) + (1,) * (out_data.ndim - 1))
        degs = tf.clip_by_value(degs, clip_value_min=1,
                                clip_value_max=np.inf)  # ???
        out_data = out_data / degs
    else:
        degs = None

    def grad(grad_out):
        grad_lhs = None
        grad_rhs = None
        if reducer == 'mean':
            grad_out = grad_out / degs
        grad_out_nd = zerocopy_to_dgl_ndarray(grad_out)
        if True:
            # grad_lhs = grad_out.new_empty((lhs_data_nd.shape[0],) + feat_shape)
            grad_lhs = tf.zeros((lhs_data_nd.shape[0],) + feat_shape)
            K.backward_lhs_binary_op_reduce(
                reducer if reducer != 'mean' else 'sum',
                binary_op, graph, lhs, rhs, lhs_data_nd, rhs_data_nd,
                out_data_nd, grad_out_nd, zerocopy_to_dgl_ndarray(grad_lhs),
                lhs_map[1], rhs_map[1], out_map[1])
            grad_lhs = _reduce_grad(grad_lhs, lhs_data_nd.shape)
        if True:
            # grad_rhs = grad_out.new_empty((rhs_data_nd.shape[0],) + feat_shape)
            grad_rhs = tf.zeros((rhs_data_nd.shape[0],) + feat_shape)
            K.backward_rhs_binary_op_reduce(
                reducer if reducer != 'mean' else 'sum',
                binary_op, graph, lhs, rhs, lhs_data_nd, rhs_data_nd,
                out_data_nd, grad_out_nd, zerocopy_to_dgl_ndarray(grad_rhs),
                lhs_map[1], rhs_map[1], out_map[1])
            grad_rhs = _reduce_grad(grad_rhs, rhs_data_nd.shape)

        return grad_lhs, grad_rhs
    return out_data, grad


def copy_reduce(reducer, graph, target, in_data, out_size, in_map=(None, None),
                out_map=(None, None)):
    @tf.custom_gradient
    def _lambda(in_data):
        return copy_reduce_real(reducer, graph, target, in_data, out_size, in_map,
                                out_map)
    return _lambda(in_data)


def copy_reduce_real(reducer, graph, target, in_data, out_size, in_map,
                     out_map):
    out_data = tf.zeros(
        (out_size,) + tuple(in_data.shape[1:]), dtype=in_data.dtype)
    in_data_nd = zerocopy_to_dgl_ndarray(in_data)
    out_data_nd = zerocopy_to_dgl_ndarray(out_data)
    K.copy_reduce(
        reducer if reducer != 'mean' else 'sum',
        graph, target, in_data_nd, out_data_nd, in_map[0], out_map[0])
    # normalize if mean reducer
    # NOTE(zihao): this is a temporary hack and we should have better solution in the future.
    if reducer == 'mean':
        # in_ones = in_data.new_ones((in_data.shape[0],))
        in_ones = tf.ones(in_data.shape[0], dtype=in_data.dtype)
        # degs = in_data.new_empty((out_data.shape[0],))
        degs = tf.zeros(out_data.shape[0], dtype=in_data.dtype)
        in_ones_nd = zerocopy_to_dgl_ndarray(in_ones)
        degs_nd = zerocopy_to_dgl_ndarray(degs)
        K.copy_reduce(
            'sum', graph, target, in_ones_nd, degs_nd, in_map[0], out_map[0])
        # reshape
        degs = tf.reshape(degs,
                          (out_data.shape[0],) + (1,) * (out_data.ndim - 1))
        degs = tf.clip_by_value(degs, clip_value_min=1,
                                clip_value_max=np.inf)  # TODO: ???
        out_data = out_data / degs
    else:
        degs = None
    # save_for_backward can only save variables

    def grad(grad_out):
        if reducer == 'mean':
            grad_out = grad_out / degs
        grad_out_nd = zerocopy_to_dgl_ndarray(grad_out)
        # if ctx.needs_input_grad[3]:
        if True:
            # grad_in = grad_out.new_empty(in_data_nd.shape)
            grad_in = tf.zeros(in_data_nd.shape)
            K.backward_copy_reduce(
                reducer if reducer != 'mean' else 'sum',
                graph, target, in_data_nd, out_data_nd, grad_out_nd,
                zerocopy_to_dgl_ndarray(grad_in), in_map[1], out_map[1])
        return grad_in
    return out_data, grad


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
    # pad inshape
    in_shape = (1,) * num_to_squeeze + in_shape
    reduce_idx = np.asarray(np.nonzero(np.asarray(grad_shape) - np.asarray(in_shape)))
    reduce_idx += 1  # skip batch dim
    reduce_idx_tensor = tf.constant(tuple(
        reduce_idx.flatten().tolist()))
    grad = tf.reduce_sum(grad, axis=reduce_idx_tensor, keepdims=True)
    return tf.reshape(grad, shape)


def sync():
    context = context().context()
    context.async_wait()
