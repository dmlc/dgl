from __future__ import absolute_import

import builtins
import numbers

import numpy as np
import scipy  # Weird bug in new pytorch when import scipy after import torch
import torch as th
from torch.utils import dlpack

from ... import ndarray as nd
from ...function.base import TargetCode
from ...utils import version

if version.parse(th.__version__) < version.parse("2.1.0"):
    raise RuntimeError("DGL requires PyTorch >= 2.1.0")


def data_type_dict():
    return {
        "bfloat16": th.bfloat16,
        "float16": th.float16,
        "float32": th.float32,
        "float64": th.float64,
        "uint8": th.uint8,
        "int8": th.int8,
        "int16": th.int16,
        "int32": th.int32,
        "int64": th.int64,
        "bool": th.bool,
    }


def cpu():
    return th.device("cpu")


def tensor(data, dtype=None):
    if isinstance(data, numbers.Number):
        data = [data]
    if (
        isinstance(data, list)
        and len(data) > 0
        and isinstance(data[0], th.Tensor)
    ):
        # prevent GPU->CPU->GPU copies
        if data[0].ndim == 0:
            # zero dimenion scalar tensors
            return th.stack(data)
    if isinstance(data, th.Tensor):
        return th.as_tensor(data, dtype=dtype, device=data.device)
    else:
        return th.as_tensor(data, dtype=dtype)


def as_scalar(data):
    return data.item()


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
            "Pytorch backend only supports COO format. But got %s." % fmt
        )
    spmat = th.sparse_coo_tensor(index[1], data, shape)
    return spmat, None


def sparse_matrix_indices(spmat):
    return ("coo", spmat._indices())


def is_tensor(obj):
    return isinstance(obj, th.Tensor)


def shape(input):
    return input.shape


def dtype(input):
    return input.dtype


def ndim(input):
    return input.dim()


def context(input):
    return input.device


def device_type(ctx):
    return th.device(ctx).type


def device_id(ctx):
    ctx = th.device(ctx)
    if ctx.index is None:
        return 0 if ctx.type == "cpu" else th.cuda.current_device()
    else:
        return ctx.index


def to_backend_ctx(dglctx):
    dev_type = dglctx.device_type
    if dev_type == 1:
        return th.device("cpu")
    elif dev_type == 2:
        return th.device("cuda", dglctx.device_id)
    else:
        raise ValueError("Unsupported DGL device context:", dglctx)


def astype(input, ty):
    return input.type(ty)


def asnumpy(input):
    if isinstance(input, th.sparse.FloatTensor):
        return input.to_dense().cpu().detach().numpy()
    else:
        return input.cpu().detach().numpy()


def copy_to(input, ctx, **kwargs):
    ctx = th.device(ctx)
    if ctx.type == "cpu":
        return input.cpu()
    elif ctx.type == "cuda":
        if ctx.index is not None:
            th.cuda.set_device(ctx.index)
        return input.cuda(**kwargs)
    else:
        raise RuntimeError("Invalid context", ctx)


def is_pinned(input):
    return input.is_pinned()


def sum(input, dim, keepdims=False):
    return th.sum(input, dim=dim, keepdim=keepdims)


def floor_div(in1, in2):
    return in1 // in2


def reduce_sum(input):
    return input.sum()


def cumsum(input, dim):
    return th.cumsum(input, dim=dim)


def mean(input, dim):
    return th.mean(input, dim=dim)


def reduce_mean(input):
    return input.mean()


def max(input, dim):
    # NOTE: the second argmax array is not returned
    return th.max(input, dim=dim)[0]


def reduce_max(input):
    return input.max()


def min(input, dim):
    # NOTE: the second argmin array is not returned
    return th.min(input, dim=dim)[0]


def reduce_min(input):
    return input.min()


def argsort(input, dim, descending):
    return th.argsort(input, dim=dim, descending=descending)


def topk(input, k, dim, descending=True):
    return th.topk(input, k, dim, largest=descending)[0]


def argtopk(input, k, dim, descending=True):
    return th.topk(input, k, dim, largest=descending)[1]


def exp(input):
    return th.exp(input)


def inverse(input):
    return th.inverse(input)


def sqrt(input):
    return th.sqrt(input)


def softmax(input, dim=-1):
    return th.softmax(input, dim=dim)


def cat(seq, dim):
    return th.cat(seq, dim=dim)


def stack(seq, dim):
    return th.stack(seq, dim=dim)


def split(input, sizes_or_sections, dim):
    return th.split(input, sizes_or_sections, dim)


def repeat(input, repeats, dim):
    return th.repeat_interleave(input, repeats, dim)  # PyTorch 1.1


def gather_row(data, row_index):
    return th.index_select(data, 0, row_index.long())


def slice_axis(data, axis, begin, end):
    return th.narrow(data, axis, begin, end - begin)


def take(data, indices, dim):
    new_shape = data.shape[:dim] + indices.shape + data.shape[dim + 1 :]
    return th.index_select(data, dim, indices.view(-1)).view(new_shape)


def narrow_row(x, start, stop):
    return x[start:stop]


def index_add_inplace(data, row_idx, value):
    data.index_add_(0, row_idx, value)


def scatter_row(data, row_index, value):
    return data.index_copy(0, row_index.long(), value)


def scatter_row_inplace(data, row_index, value):
    data[row_index.long()] = value


def squeeze(input, dim):
    return th.squeeze(input, dim)


def unsqueeze(input, dim):
    return th.unsqueeze(input, dim)


def reshape(input, shape):
    return th.reshape(input, shape)


def swapaxes(input, axis1, axis2):
    return th.transpose(input, axis1, axis2)


def empty(shape, dtype, ctx):
    return th.empty(shape, dtype=dtype, device=ctx)


def zeros(shape, dtype, ctx):
    return th.zeros(shape, dtype=dtype, device=ctx)


def zeros_like(input):
    return th.zeros_like(input)


def ones(shape, dtype, ctx):
    return th.ones(shape, dtype=dtype, device=ctx)


def uniform(shape, dtype, ctx, low, high):
    return th.empty(shape, dtype=dtype, device=ctx).uniform_(low, high)


def randint(shape, dtype, ctx, low, high):
    return th.randint(low, high, shape, dtype=dtype, device=ctx)


def pad_packed_tensor(input, lengths, value, l_min=None):
    old_shape = input.shape
    device = input.device
    if not is_tensor(lengths):
        lengths = th.tensor(lengths, dtype=th.int64, device=device)
    else:
        lengths = lengths.to(device)
    max_len = as_scalar(lengths.max())

    if l_min is not None:
        max_len = builtins.max(max_len, l_min)

    batch_size = len(lengths)
    x = input.new(batch_size * max_len, *old_shape[1:])
    x.fill_(value)
    index = th.ones(len(input), dtype=th.int64, device=device)
    cum_lengths = th.cumsum(lengths, 0)
    index[cum_lengths[:-1]] += max_len - lengths[:-1]
    index = th.cumsum(index, 0) - 1
    x[index] = input
    return x.view(batch_size, max_len, *old_shape[1:])


def pack_padded_tensor(input, lengths):
    max_len = input.shape[1]
    device = input.device
    if not is_tensor(lengths):
        lengths = th.tensor(lengths, dtype=th.int64, device=device)
    else:
        lengths = lengths.to(device)
    input = input.view(-1, *input.shape[2:])
    out_len = lengths.sum().item()
    index = th.ones(out_len, dtype=th.int64, device=device)
    cum_lengths = th.cumsum(lengths, 0)
    index[cum_lengths[:-1]] += max_len - lengths[:-1]
    index = th.cumsum(index, 0) - 1
    return input[index]


def boolean_mask(input, mask):
    if "bool" not in str(mask.dtype):
        mask = th.as_tensor(mask, dtype=th.bool)
    return input[mask]


def equal(x, y):
    return x == y


def allclose(x, y, rtol=1e-4, atol=1e-4):
    return th.allclose(x, y, rtol=rtol, atol=atol)


def logical_not(input):
    return ~input


def logical_and(input1, input2):
    return input1 & input2


def clone(input):
    return input.clone()


def clamp(data, min_val, max_val):
    return th.clamp(data, min_val, max_val)


def replace_inf_with_zero(x):
    return th.masked_fill(x, th.isinf(x), 0)


def count_nonzero(input):
    # TODO: fallback to numpy for backward compatibility
    return np.count_nonzero(input)


def unique(input, return_inverse=False, return_counts=False):
    if input.dtype == th.bool:
        input = input.type(th.int8)
    return th.unique(
        input, return_inverse=return_inverse, return_counts=return_counts
    )


def full_1d(length, fill_value, dtype, ctx):
    return th.full((length,), fill_value, dtype=dtype, device=ctx)


def nonzero_1d(input):
    x = th.nonzero(input, as_tuple=False).squeeze()
    return x if x.dim() == 1 else x.view(-1)


def sort_1d(input):
    return th.sort(input)


def arange(start, stop, dtype=th.int64, ctx=None):
    return th.arange(start, stop, dtype=dtype, device=ctx)


def rand_shuffle(arr):
    idx = th.randperm(len(arr))
    return arr[idx]


def zerocopy_to_dlpack(input):
    return dlpack.to_dlpack(input.contiguous())


def zerocopy_from_dlpack(dlpack_tensor):
    return dlpack.from_dlpack(dlpack_tensor)


def zerocopy_to_numpy(input):
    # NOTE: not zerocopy
    return asnumpy(input)


def zerocopy_from_numpy(np_array):
    return th.as_tensor(np_array)


def zerocopy_to_dgl_ndarray(data):
    if data.dtype == th.bool:
        data = data.byte()
    return nd.from_dlpack(dlpack.to_dlpack(data.contiguous()))


# NGC PyTorch containers are shipping alpha version PyTorch.
if version.parse(th.__version__) >= version.parse("2.0.0a0"):

    def check_is_view(input):
        assert (
            input.data_ptr() == input.untyped_storage().data_ptr()
        ), "Cannot convert view tensors to dgl ndarray for write."

else:

    def check_is_view(input):
        assert (
            input.data_ptr() == input._storage().data_ptr()
        ), "Cannot convert view tensors to dgl ndarray for write."


def zerocopy_to_dgl_ndarray_for_write(input):
    if input.numel() > 0:
        # only check non-empty tensors
        assert input.is_contiguous(), (
            "Cannot convert non-contiguous tensors "
            "to dgl ndarray for write. Call .to_contiguous() first."
        )
        check_is_view(input)
    return zerocopy_to_dgl_ndarray(input)


def zerocopy_from_dgl_ndarray(data):
    if data.shape == (0,):
        # NOTE: PyTorch v1.5 does not accept DLPack object representing empty CUDA tensor.
        #  Related issue: https://github.com/pytorch/pytorch/issues/41182
        #  The issue will be fixed in v1.6 and later.
        return th.tensor(
            [], dtype=getattr(th, data.dtype), device=to_backend_ctx(data.ctx)
        )
    elif len(data.shape) == 0 or builtins.min(data.shape) == 0:
        # Workaround the same issue as above, but preserve the shape of the
        # empty tensor. This is needed by the sparse optimizer when one of
        # processors may receive no gradients to update, but we want to keep
        # the dimension of the embedding.
        return th.empty(
            data.shape,
            dtype=getattr(th, data.dtype),
            device=to_backend_ctx(data.ctx),
        )
    else:
        return dlpack.from_dlpack(data.to_dlpack())


def sync():
    # Pytorch performs computation synchronously, so no need for synchronization.
    pass


def attach_grad(x):
    if x.grad is not None:
        x.grad.zero_()
        return x
    else:
        return x.requires_grad_()


def backward(x, head_gradient=None):
    if (
        head_gradient is not None
        and head_gradient.shape[0] == 1
        and len(head_gradient.shape) == 1
    ):
        # Fix for torch 1.3.1
        head_gradient = th.tensor(head_gradient.item()).to(head_gradient.device)
    x.backward(head_gradient)


def grad(x):
    x.retain_grad()
    return x.grad


def is_no_grad(x):
    return x.grad is None or (x.grad == 0).all()


def is_recording():
    return th.is_grad_enabled()


class record_grad(object):
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass


no_grad = th.no_grad
