from __future__ import absolute_import

import builtins
import numbers
import os

import mxnet as mx
import mxnet.ndarray as nd
import numpy as np

from ... import ndarray as dglnd
from ...function.base import TargetCode
from ...utils import version

if version.parse(mx.__version__) < version.parse("1.6.0"):
    raise RuntimeError("DGL requires MXNet >= 1.6")

# After MXNet 1.5, empty tensors aren't supprted by default.
# After we turn on the numpy compatible flag, MXNet supports empty NDArray.
mx.set_np_shape(bool(os.environ.get("DGL_MXNET_SET_NP_SHAPE", True)))


def data_type_dict():
    return {
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
        "uint8": np.uint8,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "bool": np.bool_,
    }  # mxnet does not support bool


def cpu():
    return mx.cpu()


def tensor(data, dtype=None):
    if dtype == np.bool_:
        # mxnet doesn't support bool
        dtype = np.int32
    if isinstance(data, nd.NDArray):
        if dtype is None or data.dtype == dtype:
            return data
        else:
            return data.astype(dtype)
    else:
        if isinstance(data, numbers.Number):
            data = [data]
        if dtype is None:
            if isinstance(data, np.ndarray):
                dtype = np.int32 if data.dtype == np.bool_ else data.dtype
            elif len(data) == 0:
                dtype = np.int64
            else:
                dtype = (
                    np.int64
                    if isinstance(data[0], numbers.Integral)
                    else np.float32
                )
        return nd.array(data, dtype=dtype)


def as_scalar(data):
    if data.size != 1:
        raise ValueError("The current array is not a scalar")
    if data.shape != (1,):
        data = data.expand_dims(axis=0)
    return data.asscalar()


def get_preferred_sparse_format():
    """Get the preferred sparse matrix format supported by the backend.

    Different backends have their preferred backend. This info is useful when
    constructing a sparse matrix.
    """
    return "csr"


def sparse_matrix(data, index, shape, force_format=False):
    fmt = index[0]
    if fmt == "coo":
        if force_format:
            raise TypeError(
                "MXNet backend only supports CSR format,"
                " but COO format is forced."
            )
        coord = index[1]
        # generate convert idx
        # FIXME: cannot use int64
        tmp_data = nd.arange(
            len(coord[0]), dtype=data.dtype, ctx=coord[0].context
        )
        tmp_spmat = nd.sparse.csr_matrix(
            (tmp_data, (coord[0], coord[1])), tuple(shape), ctx=data.context
        )
        convert_idx = nd.cast(tmp_spmat.data, dtype="int64")
        # shuffle the data
        data = data[convert_idx]
        spmat = nd.sparse.csr_matrix(
            (data, tmp_spmat.indices, tmp_spmat.indptr),
            tuple(shape),
            ctx=data.context,
        )
        return spmat, convert_idx
    elif fmt == "csr":
        indices = index[1]
        indptr = index[2]
        spmat = nd.sparse.csr_matrix(
            (data, indices, indptr), tuple(shape), ctx=data.context
        )
        # No conversion is required.
        return spmat, None
    else:
        raise TypeError("Invalid format: %s." % fmt)


def sparse_matrix_indices(spmat):
    return ("csr", spmat.indices, spmat.indptr)


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


def to_backend_ctx(dglctx):
    dev_type = dglctx.device_type
    if dev_type == 1:
        return mx.cpu()
    elif dev_type == 2:
        return mx.gpu(dglctx.device_id)
    else:
        raise ValueError("Unsupported DGL device context:", dglctx)


def astype(input, ty):
    if ty == np.bool_:
        ty = np.int32
    return input.astype(ty)


def asnumpy(input):
    return input.asnumpy()


def copy_to(input, ctx, **kwargs):
    return input.as_in_context(ctx)


def is_pinned(input):
    return input.context == mx.cpu_pinned()


def sum(input, dim, keepdims=False):
    if len(input) == 0:
        return nd.array([0.0], dtype=input.dtype, ctx=input.context)
    return nd.sum(input, axis=dim, keepdims=keepdims)


def floor_div(in1, in2):
    return in1 / in2


def reduce_sum(input):
    return input.sum()


def cumsum(input, dim):
    return nd.cumsum(input, axis=dim)


def mean(input, dim):
    return nd.mean(input, axis=dim)


def reduce_mean(input):
    return input.mean()


def max(input, dim):
    return nd.max(input, axis=dim)


def reduce_max(input):
    return input.max()


def min(input, dim):
    return nd.min(input, axis=dim)


def reduce_min(input):
    return input.min()


def topk(input, k, dim, descending=True):
    return nd.topk(
        input, axis=dim, k=k, ret_typ="value", is_ascend=not descending
    )


def argtopk(input, k, dim, descending=True):
    idx = nd.argsort(input, dim, is_ascend=not descending)
    return nd.slice_axis(input, dim, 0, k)


def argsort(input, dim, descending):
    idx = nd.argsort(input, dim, is_ascend=not descending)
    idx = nd.cast(idx, dtype="int64")
    return idx


def exp(input):
    return nd.exp(input)


def inverse(input):
    return nd.linalg_inverse(input)


def sqrt(input):
    return nd.sqrt(input)


def softmax(input, dim=-1):
    return nd.softmax(input, axis=dim)


def cat(seq, dim):
    return nd.concat(*seq, dim=dim)


def stack(seq, dim):
    return nd.stack(*seq, axis=dim)


def split(x, sizes_or_sections, dim):
    if isinstance(sizes_or_sections, list) and len(sizes_or_sections) == 1:
        assert len(x) == sizes_or_sections[0]
        return [x]

    if isinstance(sizes_or_sections, (np.ndarray, list)):
        sizes_or_sections1 = tuple(np.cumsum(sizes_or_sections)[:-1])
    return nd.split_v2(x, sizes_or_sections1, axis=dim)


def repeat(input, repeats, dim):
    if isinstance(repeats, nd.NDArray):
        return nd.array(
            np.repeat(input.asnumpy(), repeats.asnumpy(), axis=dim),
            ctx=input.context,
            dtype=input.dtype,
        )
    else:
        return nd.repeat(input, repeats, axis=dim)


def gather_row(data, row_index):
    # MXNet workaround for empty row index
    if len(row_index) == 0:
        if data.shape[0] == 0:
            return data
        else:
            return data[0:0]

    if isinstance(row_index, nd.NDArray):
        return nd.take(data, row_index)
    else:
        return data[
            row_index,
        ]


def slice_axis(data, axis, begin, end):
    dim = data.shape[axis]
    if begin < 0:
        begin += dim
    if end <= 0:
        end += dim
    return nd.slice_axis(data, axis, begin, end)


def take(data, indices, dim):
    return nd.take(data, indices, dim)


def narrow_row(data, start, stop):
    return data[start:stop]


def index_add_inplace(data, row_idx, value):
    raise NotImplementedError("MXNet doesn't support inplace index_add")


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
    return nd.reshape(input, shape)


def swapaxes(input, axis1, axis2):
    return nd.swapaxes(input, axis1, axis2)


def empty(shape, dtype, ctx):
    return nd.empty(shape, dtype=dtype, ctx=ctx)


def zeros(shape, dtype, ctx):
    return nd.zeros(shape, dtype=dtype, ctx=ctx)


def zeros_like(input):
    return nd.zeros_like(input)


def ones(shape, dtype, ctx):
    return nd.ones(shape, dtype=dtype, ctx=ctx)


def uniform(shape, dtype, ctx, low, high):
    return nd.random.uniform(low, high, ctx=ctx, dtype=dtype, shape=shape)


def randint(shape, dtype, ctx, low, high):
    return nd.random.randint(low, high, ctx=ctx, dtype=dtype, shape=shape)


def pad_packed_tensor(input, lengths, value, l_min=None):
    old_shape = input.shape
    if isinstance(lengths, nd.NDArray):
        lengths = list(lengths.asnumpy())
    max_len = builtins.max(lengths)

    if l_min is not None:
        max_len = builtins.max(max_len, l_min)

    batch_size = len(lengths)
    ctx = input.context
    dtype = input.dtype
    x = nd.full(
        (batch_size * max_len, *old_shape[1:]), value, ctx=ctx, dtype=dtype
    )
    index = []
    for i, l in enumerate(lengths):
        index.extend(range(i * max_len, i * max_len + l))
    index = nd.array(index, ctx=ctx)
    return scatter_row(x, index, input).reshape(
        batch_size, max_len, *old_shape[1:]
    )


def pack_padded_tensor(input, lengths):
    batch_size, max_len = input.shape[:2]
    ctx = input.context
    index = []
    for i, l in enumerate(lengths):
        index.extend(range(i * max_len, i * max_len + l))
    index = nd.array(index, ctx=ctx)
    return gather_row(input.reshape(batch_size * max_len, -1), index)


def boolean_mask(input, mask):
    return mx.contrib.nd.boolean_mask(input, mask)


def equal(x, y):
    return x == y


def allclose(x, y, rtol=1e-4, atol=1e-4):
    return np.allclose(x.asnumpy(), y.asnumpy(), rtol=rtol, atol=atol)


def logical_not(input):
    return nd.logical_not(input)


def logical_and(input1, input2):
    return nd.logical_and(input1, input2)


def clone(input):
    return input.copy()


def clamp(data, min_val, max_val):
    return nd.clip(data, min_val, max_val)


def replace_inf_with_zero(x):
    return nd.where(nd.abs(x) == np.inf, nd.zeros_like(x), x)


def count_nonzero(input):
    # TODO: fallback to numpy is unfortunate
    tmp = input.asnumpy()
    return np.count_nonzero(tmp)


def unique(input, return_inverse=False, return_counts=False):
    # TODO: fallback to numpy is unfortunate
    tmp = input.asnumpy()
    if return_inverse and return_counts:
        tmp, inv, count = np.unique(
            tmp, return_inverse=True, return_counts=True
        )
        tmp = nd.array(tmp, ctx=input.context, dtype=input.dtype)
        inv = nd.array(inv, ctx=input.context)
        count = nd.array(count, ctx=input.context)
        return tmp, inv, count
    elif return_inverse or return_counts:
        tmp, tmp2 = np.unique(
            tmp, return_inverse=return_inverse, return_counts=return_counts
        )
        tmp = nd.array(tmp, ctx=input.context, dtype=input.dtype)
        tmp2 = nd.array(tmp2, ctx=input.context)
        return tmp, tmp2
    else:
        tmp = np.unique(tmp)
        return nd.array(tmp, ctx=input.context, dtype=input.dtype)


def full_1d(length, fill_value, dtype, ctx):
    return nd.full((length,), fill_value, dtype=dtype, ctx=ctx)


def nonzero_1d(input):
    # TODO: fallback to numpy is unfortunate
    tmp = input.asnumpy()
    tmp = np.nonzero(tmp)[0]
    r = nd.array(tmp, ctx=input.context, dtype=tmp.dtype)
    return r


def sort_1d(input):
    # TODO: this isn't an ideal implementation.
    val = nd.sort(input, axis=None, is_ascend=True)
    idx = nd.argsort(input, is_ascend=True)
    idx = nd.cast(idx, dtype="int64")
    return val, idx


def arange(start, stop, dtype=np.int64, ctx=None):
    if start >= stop:
        return nd.array([], dtype=dtype, ctx=ctx)
    else:
        return nd.arange(start, stop, dtype=dtype, ctx=ctx)


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
    np_data = np.asarray(np_data, order="C")
    return mx.nd.from_numpy(np_data, zero_copy=True)


def zerocopy_to_dgl_ndarray(arr):
    arr.to_dlpack_for_read()
    return dglnd.from_dlpack(arr.to_dlpack_for_read())


def zerocopy_to_dgl_ndarray_for_write(arr):
    return dglnd.from_dlpack(arr.to_dlpack_for_write())


def zerocopy_from_dgl_ndarray(arr):
    return nd.from_dlpack(arr.to_dlpack())


def sync():
    """Synchronize computation.

    In DL frameworks such as MXNet and TensorFlow, the computation in operators
    are done asynchronously. This is to synchronize computation and makes sure
    that all computation is complete after this function call.
    """
    mx.nd.waitall()


def attach_grad(tensor):
    tensor.attach_grad()
    return tensor


def backward(x, head_gradient=None):
    x.backward(head_gradient)


def grad(x):
    return x.grad


def is_no_grad(x):
    return (x != 0).sum() == 0


def is_recording():
    return mx.autograd.is_recording()


record_grad = mx.autograd.record


class no_grad(object):
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass
