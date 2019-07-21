from __future__ import absolute_import

from distutils.version import LooseVersion

import torch as th
from torch.utils import dlpack

from ... import ndarray as nd
from ... import kernel as K

TH_VERSION = LooseVersion(th.__version__)

def data_type_dict():
    return {'float16' : th.float16,
            'float32' : th.float32,
            'float64' : th.float64,
            'uint8'   : th.uint8,
            'int8'    : th.int8,
            'int16'   : th.int16,
            'int32'   : th.int32,
            'int64'   : th.int64}

def cpu():
    return th.device('cpu')

def tensor(data, dtype=None):
    return th.tensor(data, dtype=dtype)

def get_preferred_sparse_format():
    """Get the preferred sparse matrix format supported by the backend.

    Different backends have their preferred backend. This info is useful when
    constructing a sparse matrix.
    """
    return "coo"

def sparse_matrix(data, index, shape, force_format=False):
    fmt = index[0]
    if fmt != 'coo':
        raise TypeError('Pytorch backend only supports COO format. But got %s.' % fmt)
    spmat = th.sparse_coo_tensor(index[1], data, shape)
    return spmat, None

def sparse_matrix_indices(spmat):
    return ('coo', spmat._indices())

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
    return ctx.type

def device_id(ctx):
    if ctx.index is None:
        return 0
    else:
        return ctx.index

def astype(input, ty):
    return input.type(ty)

def asnumpy(input):
    if isinstance(input, th.sparse.FloatTensor):
        return input.to_dense().detach().cpu().numpy()
    else:
        return input.detach().cpu().numpy()

def copy_to(input, ctx):
    if ctx.type == 'cpu':
        return input.cpu()
    elif ctx.type == 'cuda':
        return input.cuda(ctx.index)
    else:
        raise RuntimeError('Invalid context', ctx)

def sum(input, dim):
    return th.sum(input, dim=dim)

def mean(input, dim):
    return th.mean(input, dim=dim)

def max(input, dim):
    # NOTE: the second argmax array is not returned
    return th.max(input, dim=dim)[0]

def cat(seq, dim):
    return th.cat(seq, dim=dim)

def stack(seq, dim):
    return th.stack(seq, dim=dim)

def split(input, sizes_or_sections, dim):
    return th.split(input, sizes_or_sections, dim)

def gather_row(data, row_index):
    return th.index_select(data, 0, row_index)

def narrow_row(x, start, stop):
    return x[start:stop]

def scatter_row(data, row_index, value):
    return data.index_copy(0, row_index, value)

def scatter_row_inplace(data, row_index, value):
    data[row_index] = value

def squeeze(input, dim):
    return th.squeeze(input, dim)

def unsqueeze(input, dim):
    return th.unsqueeze(input, dim)

def reshape(input, shape):
    return th.reshape(input ,shape)

def zeros(shape, dtype, ctx):
    return th.zeros(shape, dtype=dtype, device=ctx)

def zeros_like(input):
    return th.zeros_like(input)

def ones(shape, dtype, ctx):
    return th.ones(shape, dtype=dtype, device=ctx)

def unsorted_1d_segment_sum(input, seg_id, n_segs, dim):
    y = th.zeros(n_segs, *input.shape[1:]).to(input)
    seg_id = seg_id.view((-1,) + (1,) * (input.dim() - 1)).expand_as(input)
    y = y.scatter_add_(dim, seg_id, input)
    return y

def unsorted_1d_segment_mean(input, seg_id, n_segs, dim):
    w = unsorted_1d_segment_sum(th.ones_like(seg_id), seg_id, n_segs, 0).to(input)
    w = w.clamp(min=1)   # remove 0 entries
    y = unsorted_1d_segment_sum(input, seg_id, n_segs, dim)
    y = y / w.view((-1,) + (1,) * (y.dim() - 1))
    return y

def boolean_mask(input, mask):
    return input[mask]

def equal(x, y):
    return x == y

def logical_not(input):
    return ~input

def unique(input):
    return th.unique(input)

def full_1d(length, fill_value, dtype, ctx):
    return th.full((length,), fill_value, dtype=dtype, device=ctx)

def nonzero_1d(input):
    x = th.nonzero(input).squeeze()
    return x if x.dim() == 1 else x.view(-1)

def sort_1d(input):
    return th.sort(input)

def arange(start, stop):
    return th.arange(start, stop, dtype=th.int64)

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
    return th.from_numpy(np_array)

def zerocopy_to_dgl_ndarray(input):
    return nd.from_dlpack(dlpack.to_dlpack(input.contiguous()))

def zerocopy_from_dgl_ndarray(input):
    return dlpack.from_dlpack(input.to_dlpack())


class BinaryReduce(th.autograd.Function):
    # pylint: disable=invalid-name
    @staticmethod
    def forward(ctx, reducer, op, G, A_target, B_target, A, B,
                out_size, A_rows, B_rows, out_rows):
        A_nd = zerocopy_to_dgl_ndarray(A)
        B_nd = zerocopy_to_dgl_ndarray(B)
        feat_shape = K.infer_binary_feature_shape(A_nd, B_nd)
        out = A.new_empty((out_size,) + feat_shape)
        out_nd = zerocopy_to_dgl_ndarray(out)
        K.binary_op_reduce(
            reducer, op, G, A_target, B_target, A_nd, B_nd,
            out_nd, A_rows[0], B_rows[0], out_rows[0])
        # save_for_backward can only save variables
        ctx.backward_cache = (reducer, op, G, A_target, B_target, A_rows,
                              B_rows, out_rows, A_nd, B_nd,
                              out_nd, feat_shape)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        reducer, op, G, A_target, B_target, A_rows, B_rows, out_rows, \
            A_nd, B_nd, out_nd, feat_shape \
            = ctx.backward_cache
        ctx.backward_cache = None
        grad_A = None
        grad_B = None
        grad_out_nd = zerocopy_to_dgl_ndarray(grad_out)
        if ctx.needs_input_grad[5]:
            grad_A = grad_out.new_empty((A_nd.shape[0],) + feat_shape)
            K.backward_lhs_binary_op_reduce(
                reducer, op, G, A_target, B_target, A_nd, B_nd,
                out_nd, grad_out_nd, zerocopy_to_dgl_ndarray(grad_A),
                A_rows[1], B_rows[1], out_rows[1])
            grad_A = _reduce_grad(grad_A, A_nd.shape)
        if ctx.needs_input_grad[6]:
            grad_B = grad_out.new_empty((B_nd.shape[0],) + feat_shape)
            K.backward_rhs_binary_op_reduce(
                reducer, op, G, A_target, B_target, A_nd, B_nd,
                out_nd, grad_out_nd, zerocopy_to_dgl_ndarray(grad_B),
                A_rows[1], B_rows[1], out_rows[1])
            grad_B = _reduce_grad(grad_B, B_nd.shape)

        return None, None, None, None, None, grad_A, grad_B, None, None, \
            None, None


class CopyReduce(th.autograd.Function):
    # pylint: disable=invalid-name
    @staticmethod
    def forward(ctx, reducer, G, target, X, out_size, X_rows,
                out_rows):
        out = X.new_empty((out_size,) + X.shape[1:])
        X_nd = zerocopy_to_dgl_ndarray(X)
        out_nd = zerocopy_to_dgl_ndarray(out)
        K.copy_reduce(
            reducer, G, target, X_nd, out_nd, X_rows[0],
            out_rows[0])
        # save_for_backward can only save variables
        ctx.backward_cache = (reducer, G, target, X_rows, out_rows,
                              X_nd, out_nd)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        reducer, G, target, X_rows, out_rows, X_nd, out_nd \
            = ctx.backward_cache
        ctx.backward_cache = None
        grad_X = None
        grad_out_nd = zerocopy_to_dgl_ndarray(grad_out)
        if ctx.needs_input_grad[3]:
            grad_X = grad_out.new_empty(X_nd.shape)
            K.backward_copy_reduce(
                reducer, G, target, X_nd, out_nd, grad_out_nd,
                zerocopy_to_dgl_ndarray(grad_X), X_rows[1], out_rows[1])
        return None, None, None, grad_X, None, None, None


binary_reduce = BinaryReduce.apply
copy_reduce = CopyReduce.apply


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
    reduce_idx = th.nonzero(th.tensor(grad_shape) - th.tensor(in_shape))
    reduce_idx += 1  # skip batch dim
    grad = grad.sum(dim=tuple(reduce_idx), keepdim=True)
    return grad.view(shape)

def sync():
    # Pytorch performs computation synchronously, so no need for synchronization.
    pass
