"""Tensorflow backend implementation"""
from __future__ import absolute_import

import builtins
import numbers

import numpy as np
import tensorflow as tf

from ... import ndarray as nd
from ...function.base import TargetCode
from ...utils import version

if version.parse(tf.__version__) < version.parse("2.3.0"):
    raise RuntimeError(
        "DGL requires TensorFlow>=2.3.0 for the official DLPack support."
    )


def zerocopy_to_dlpack(data):
    return tf.experimental.dlpack.to_dlpack(data)


def zerocopy_from_dlpack(dlpack_tensor):
    # TODO(Jinjing): Tensorflow requires memory to be 64-bytes aligned. We check the
    #   alignment and make a copy if needed. The functionality is better in TF's main repo.
    aligned = nd.from_dlpack(dlpack_tensor).to_dlpack(64)
    return tf.experimental.dlpack.from_dlpack(aligned)


def data_type_dict():
    return {
        "bfloat16": tf.bfloat16,
        "float16": tf.float16,
        "float32": tf.float32,
        "float64": tf.float64,
        "uint8": tf.uint8,
        "int8": tf.int8,
        "int16": tf.int16,
        "int32": tf.int32,
        "int64": tf.int64,
        "bool": tf.bool,
    }


def cpu():
    return "/cpu:0"


def tensor(data, dtype=None):
    if isinstance(data, tf.Tensor):
        if dtype is None or data.dtype == dtype:
            return data
        else:
            return tf.cast(data, dtype=dtype)
    else:
        if isinstance(data, numbers.Number):
            data = [data]
        return tf.convert_to_tensor(data, dtype=dtype)


def initialize_context():
    tf.zeros(1)


def as_scalar(data):
    data = data.numpy()
    return data if np.isscalar(data) else data.item()


def get_preferred_sparse_format():
    """Get the preferred sparse matrix format supported by the backend.

    Different backends have their preferred backend. This info is useful when
    constructing a sparse matrix.
    """
    return "coo"


def sparse_matrix(data, index, shape, force_format=False):
    fmt = index[0]
    if fmt != "coo":
        raise TypeError(
            "Tensorflow backend only supports COO format. But got %s." % fmt
        )
    # tf.SparseTensor only supports int64 indexing,
    # therefore manually casting to int64 when input in int32
    spmat = tf.SparseTensor(
        indices=tf.cast(tf.transpose(index[1], (1, 0)), tf.int64),
        values=data,
        dense_shape=shape,
    )
    return spmat, None


def sparse_matrix_indices(spmat):
    return ("coo", spmat.indices)


def is_tensor(obj):
    return isinstance(obj, tf.Tensor)


def shape(input):
    return input.shape


def dtype(input):
    return input.dtype


def ndim(input):
    return input.ndim


def context(input):
    spec = tf.DeviceSpec.from_string(input.device)
    return "/{}:{}".format(spec.device_type.lower(), spec.device_index)


def device_type(ctx):
    return tf.DeviceSpec.from_string(ctx).device_type.lower()


def device_id(ctx):
    return tf.DeviceSpec.from_string(ctx).device_index


def to_backend_ctx(dglctx):
    dev_type = dglctx.device_type
    if dev_type == 1:
        return "/cpu:0"
    elif dev_type == 2:
        return "/gpu:%d" % (dglctx.device_id)
    else:
        raise ValueError("Unsupported DGL device context:", dglctx)


def astype(input, ty):
    with tf.device(input.device):
        return tf.cast(input, dtype=ty)


def asnumpy(input):
    if isinstance(input, tf.SparseTensor):
        # tf.sparse.to_dense assume sorted indices, need to turn off validate_indices in our cases
        return tf.sparse.to_dense(input, validate_indices=False).numpy()
    else:
        return input.numpy()


def copy_to(input, ctx, **kwargs):
    with tf.device(ctx):
        new_tensor = tf.identity(input)
    return new_tensor


def is_pinned(input):
    return False  # not sure how to do this


def sum(input, dim, keepdims=False):
    if input.dtype == tf.bool:
        input = tf.cast(input, tf.int32)
    return tf.reduce_sum(input, axis=dim, keepdims=keepdims)


def floor_div(in1, in2):
    return astype(in1 / in2, dtype(in1))


def reduce_sum(input):
    if input.dtype == tf.bool:
        input = tf.cast(input, tf.int32)
    return tf.reduce_sum(input)


def cumsum(input, dim):
    if input.dtype == tf.bool:
        input = tf.cast(input, tf.int32)
    return tf.cumsum(input, axis=dim)


def mean(input, dim):
    return tf.reduce_mean(input, axis=dim)


def reduce_mean(input):
    return tf.reduce_mean(input)


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
        return tf.cast(
            tf.argsort(input, axis=dim, direction="DESCENDING"), dtype=tf.int64
        )
    else:
        return tf.cast(
            tf.argsort(input, axis=dim, direction="ASCENDING"), dtype=tf.int64
        )


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


def inverse(input):
    return tf.linalg.inv(input)


def sqrt(input):
    return tf.sqrt(input)


def softmax(input, dim=-1):
    return tf.math.softmax(input, axis=dim)


def cat(seq, dim):
    return tf.concat(seq, axis=dim)


def stack(seq, dim):
    return tf.stack(seq, axis=dim)


def split(input, sizes_or_sections, dim):
    return [
        copy_to(_, input.device)
        for _ in tf.split(input, sizes_or_sections, axis=dim)
    ]


def repeat(input, repeats, dim):
    return tf.repeat(input, repeats, dim)


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
    # XXX(minjie): Normally, the copy_to here is unnecessary. However, TF has this
    #   notorious legacy issue that int32 type data is always on CPU, which will
    #   crash the program since DGL requires feature data to be on the same device
    #   as graph structure.
    return copy_to(
        tf.tensor_scatter_nd_update(data, row_index, value), data.device
    )


def index_add_inplace(data, row_idx, value):
    raise NotImplementedError("Tensorflow doesn't support inplace index_add")


def scatter_row_inplace(data, row_index, value):
    raise NotImplementedError("Tensorflow doesn't support inplace update")


def squeeze(input, dim):
    return tf.squeeze(input, axis=dim)


def unsqueeze(input, dim):
    return tf.expand_dims(input, axis=dim)


def reshape(input, shape):
    return tf.reshape(input, shape)


def swapaxes(input, axis1, axis2):
    ndim = input.ndim
    t = list(range(ndim))
    t[axis1], t[axis2] = axis2 % ndim, axis1 % ndim
    return tf.transpose(input, perm=t)


def empty(shape, dtype, ctx):
    # tf doesn't have tf.empty(), use zeros() as a workaround
    return zeros(shape, dtype, ctx)


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


def randint(shape, dtype, ctx, low, high):
    with tf.device(ctx):
        t = tf.random.uniform(shape, dtype=dtype, minval=low, maxval=high)
    return t


def pad_packed_tensor(input, lengths, value, l_min=None):
    old_shape = input.shape
    if isinstance(lengths, tf.Tensor):
        max_len = as_scalar(tf.reduce_max(lengths))
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
        t = input[cum_row : cum_row + l]
        pad_nparray[0, 1] = max_len - l
        t = tf.pad(
            t, tf.constant(pad_nparray), mode="CONSTANT", constant_values=value
        )
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


def boolean_mask(input, mask):
    return tf.boolean_mask(input, mask)


def equal(x, y):
    return x == y


def allclose(x, y, rtol=1e-4, atol=1e-4):
    return np.allclose(
        tf.convert_to_tensor(x).numpy(),
        tf.convert_to_tensor(y).numpy(),
        rtol=rtol,
        atol=atol,
    )


def logical_not(input):
    return ~input


def logical_and(input1, input2):
    return tf.math.logical_and(input1, input2)


def clone(input):
    # TF tensor is always immutable so returning the input is safe.
    return input


def clamp(data, min_val, max_val):
    return tf.clip_by_value(data, min_val, max_val)


def replace_inf_with_zero(x):
    return tf.where(tf.abs(x) == np.inf, 0, x)


def count_nonzero(input):
    return int(tf.math.count_nonzero(input))


def unique(input, return_inverse=False, return_counts=False):
    if return_inverse and return_counts:
        return tf.unique_with_counts(input)
    elif return_counts:
        result = tf.unique_with_counts(input)
        return result.y, result.count
    elif return_inverse:
        return tf.unique(input)
    else:
        return tf.unique(input).y


def full_1d(length, fill_value, dtype, ctx):
    with tf.device(ctx):
        t = tf.fill([length], value=fill_value)
        t = tf.cast(t, dtype=dtype)
    return t


def nonzero_1d(input):
    nonzero_bool = tf.cast(input, tf.bool)
    return tf.reshape(tf.where(nonzero_bool), (-1,))


def sort_1d(input):
    return tf.sort(input), tf.cast(tf.argsort(input), dtype=tf.int64)


def arange(start, stop, dtype=tf.int64, ctx=None):
    if not ctx:
        ctx = "/cpu:0"
    with tf.device(ctx):
        t = tf.range(start, stop, dtype=dtype)
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


def zerocopy_to_dgl_ndarray(data):
    if device_type(data.device) == "gpu" and data.dtype in (tf.int32, tf.int64):
        # NOTE: TF doesn't keep signed tensors on GPU due to legacy issues with
        #   shape inference. Convert it to unsigned and cast it back afterwards.
        if data.dtype == tf.int32:
            data = tf.cast(data, tf.uint32)
        elif data.dtype == tf.int64:
            data = tf.cast(data, tf.uint64)
        return nd.cast_to_signed(nd.from_dlpack(zerocopy_to_dlpack(data)))
    else:
        return nd.from_dlpack(zerocopy_to_dlpack(data))


def zerocopy_to_dgl_ndarray_for_write(input):
    return zerocopy_to_dgl_ndarray(input)


def zerocopy_from_dgl_ndarray(input):
    return zerocopy_from_dlpack(input.to_dlpack())


def sync():
    context = context().context()
    context.async_wait()


class GradContext:
    def __init__(self):
        self.tensor_for_grad = []
        self.grad_list = []
        self.tape = None

    def set_tape(self, tape):
        self.tape = tape

    def add_tensor(self, x):
        idx_pop = []
        for idx, ele in enumerate(self.tensor_for_grad):
            if ele._id == x._id:
                idx_pop.append(idx)
        if len(idx_pop) > 0:
            self.tensor_for_grad.pop(idx_pop[0])
        if self.tape is not None:
            self.tape.watch(x)
        self.tensor_for_grad.append(x)

    def backward(self, x, head_gradient=None):
        if head_gradient is not None:
            x = x * head_gradient
        self.grad_list = self.tape.gradient(x, self.tensor_for_grad)

    def is_no_grad(self, x):
        idx_pop = []
        for idx, ele in enumerate(self.tensor_for_grad):
            if ele._id == x._id:
                idx_pop.append(idx)
        if len(idx_pop) == 0:
            return True
        else:
            return self.grad_list[idx_pop[0]] is None

    def grad(self, x):
        idx_pop = []
        for idx, ele in enumerate(self.tensor_for_grad):
            if ele._id == x._id:
                idx_pop.append(idx)
        assert len(idx_pop) == 1
        t = self.grad_list[idx_pop[0]]
        return tf.convert_to_tensor(t)


cgrad = GradContext()


def get_cgrad():
    return cgrad


class record_grad:
    def __init__(self):
        self.tape = tf.GradientTape()

    def __enter__(self):
        cgrad.set_tape(self.tape)
        self.tape.__enter__()
        for x in cgrad.tensor_for_grad:
            self.tape.watch(x)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # pass
        self.tape.__exit__(exc_type, exc_value, exc_traceback)
        cgrad.tape = None


def attach_grad(x):
    cgrad.add_tensor(x)
    return x


def backward(x, head_gradient=None):
    cgrad.backward(x, head_gradient)


def grad(x):
    return cgrad.grad(x)


def is_no_grad(x):
    return cgrad.is_no_grad(x)


def is_recording():
    raise NotImplementedError("Tensorflow doesn't support is_recording")


no_grad = None

initialize_context()
