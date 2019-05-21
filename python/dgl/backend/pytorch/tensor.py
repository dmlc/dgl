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
        return input.to_dense().cpu().numpy()
    else:
        return input.cpu().numpy()

def copy_to(input, ctx):
    if ctx.type == 'cpu':
        return input.cpu()
    elif ctx.type == 'cuda':
        th.cuda.set_device(ctx.index)
        return input.cuda()
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


class SrcOpEdgeReduce(th.autograd.Function):
    @staticmethod
    def forward(ctx, reducer, binary_op, spmat, src_data, edge_data, out_size,
                src_map, edge_map, out_map):
        src_data_nd = zerocopy_to_dgl_ndarray(src_data)
        edge_data_nd = zerocopy_to_dgl_ndarray(edge_data)
        feat_shape = K.infer_binary_feature_shape(src_data_nd, edge_data_nd)
        out_data = src_data.new_empty((out_size,) + feat_shape)
        out_data_nd = zerocopy_to_dgl_ndarray(out_data)
        if reducer == "none":
            forward_out_map = out_map[0]
            backward_out_map = out_map[1]
        else:
            forward_out_map = out_map
            backward_out_map = out_map
        K.src_op_edge_reduce(
            reducer, binary_op, spmat[0], spmat[1], spmat[2], spmat[3],
            src_map, edge_map[0], src_data_nd, edge_data_nd, forward_out_map,
            out_data_nd)
        # save_for_backward can only save variables
        ctx.backward_cache = (reducer, binary_op, spmat, src_map, edge_map,
                              backward_out_map, src_data_nd, edge_data_nd,
                              out_data_nd, feat_shape)
        return out_data

    @staticmethod
    def backward(ctx, grad_out):
        reducer, binary_op, spmat, src_map, edge_map, backward_out_map, \
            src_data_nd, edge_data_nd, out_data_nd, feat_shape = ctx.backward_cache
        ctx.backward_cache = None
        grad_src = None
        grad_edge = None
        grad_out_nd = zerocopy_to_dgl_ndarray(grad_out)
        if ctx.needs_input_grad[3]:
            grad_src = grad_out.new_empty((src_data_nd.shape[0],) + feat_shape)
            K.backward_lhs_src_mul_edge_reduce(
                reducer, binary_op, spmat[0], spmat[1], spmat[2], spmat[3],
                src_map, edge_map[1], backward_out_map, src_data_nd,
                edge_data_nd, out_data_nd, grad_out_nd,
                zerocopy_to_dgl_ndarray(grad_src))
            grad_src = _reduce_grad(grad_src, src_data_nd.shape)
        if ctx.needs_input_grad[4]:
            grad_edge = grad_out.new_empty((edge_data_nd.shape[0],) + feat_shape)
            K.backward_rhs_src_mul_edge_reduce(
                reducer, binary_op, spmat[0], spmat[1], spmat[2], spmat[3],
                src_map, edge_map[1], backward_out_map, src_data_nd,
                edge_data_nd, out_data_nd, grad_out_nd,
                zerocopy_to_dgl_ndarray(grad_edge))
            grad_edge = _reduce_grad(grad_edge, edge_data_nd.shape)
            #grad_edge = grad_edge.sum(dim=2, keepdim=True)

        return None, None, None, grad_src, grad_edge, None, None, None, None


class SrcOpDstReduce(th.autograd.Function):
    @staticmethod
    def forward(ctx, reducer, binary_op, spmat, src_data, dst_data, out_size,
                src_map, dst_map, out_map):
        src_data = zerocopy_to_dgl_ndarray(src_data)
        dst_data = zerocopy_to_dgl_ndarray(dst_data)
        feat_shape = K.infer_binary_feature_shape(src_data, dst_data)
        out_data = src_data.new_empty((out_size,) + feat_shape)
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
        # save_for_backward can only save variables
        ctx.backward_cache = (reducer, binary_op, spmat, src_map, dst_map,
                              backward_out_map, src_data, dst_data, out_data)
        return zerocopy_from_dgl_ndarray(out_data)

    @staticmethod
    def backward(ctx, grad_out):
        reducer, binary_op, spmat, src_map, dst_map, backward_out_map, \
            src_data, dst_data, out_data, feat_shape = ctx.backward_cache
        ctx.backward_cache = None
        grad_src = None
        grad_dst = None
        grad_out = zerocopy_to_dgl_ndarray(grad_out)
        if ctx.needs_input_grad[3]:
            grad_src = src_data.new_empty((src_data.shape[0]) + feat_shape)
            K.backward_lhs_src_mul_dst_reduce(
                reducer, binary_op, spmat[0], spmat[1], spmat[2], spmat[3],
                src_map, dst_map, backward_out_map, src_data, dst_data,
                out_data, grad_out, grad_src)
            grad_src = zerocopy_from_dgl_ndarray(grad_src)
            grad_src = _reduce_grad(grad_src, src_data.shape)
        if ctx.needs_input_grad[4]:
            grad_dst = dst_data.new_empty((dst_data.shape[1],) + feat_shape)
            K.backward_rhs_src_mul_dst_reduce(
                reducer, binary_op, spmat[0], spmat[1], spmat[2], spmat[3],
                src_map, dst_map, backward_out_map, src_data, dst_data,
                out_data, grad_out, grad_dst)
            grad_dst = zerocopy_from_dgl_ndarray(grad_dst)
            grad_dst = _reduce_grad(grad_dst, dst_data.shape)
        return None, None, None, grad_src, grad_dst, None, None, None, None


class CopySrcReduce(th.autograd.Function):
    @staticmethod
    def forward(ctx, reducer, spmat, src_data, out_size, src_map, out_map):
        if reducer == "none":
            forward_out_map = out_map[0]
            backward_out_map = out_map[1]
        else:
            forward_out_map = out_map
            backward_out_map = out_map
        out_data = src_data.new_empty((out_size,) + src_data.shape[1:])
        src_data_nd = zerocopy_to_dgl_ndarray(src_data)
        out_data_nd = zerocopy_to_dgl_ndarray(out_data)
        K.copy_src_reduce(
            reducer, spmat[0], spmat[1], spmat[2], spmat[3], src_map,
            src_data_nd, forward_out_map, out_data_nd)
        # save_for_backward can only save variables
        ctx.backward_cache = (reducer, spmat, src_map, backward_out_map,
                              src_data_nd, out_data_nd)
        return out_data

    @staticmethod
    def backward(ctx, grad_out):
        reducer, spmat, src_map, backward_out_map, src_data_nd, out_data_nd \
            = ctx.backward_cache
        ctx.backward_cache = None
        grad_src = None
        grad_out_nd = zerocopy_to_dgl_ndarray(grad_out)
        if ctx.needs_input_grad[2]:
            grad_src = grad_out.new_empty(src_data_nd.shape)
            K.backward_copy_src_reduce(
                reducer, spmat[0], spmat[1], spmat[2], spmat[3], src_map,
                backward_out_map, src_data_nd, out_data_nd, grad_out_nd,
                zerocopy_to_dgl_ndarray(grad_src))
        return None, None, grad_src, None, None, None


class CopyEdgeReduce(th.autograd.Function):
    @staticmethod
    def forward(ctx, reducer, spmat, edge_data, out_size, edge_map, out_map):
        out_data = edge_data.new_empty((out_size,) + edge_data.shape[1:])
        edge_data_nd = zerocopy_to_dgl_ndarray(edge_data)
        out_data_nd = zerocopy_to_dgl_ndarray(out_data)
        if reducer == "none":
            forward_out_map = out_map[0]
            backward_out_map = out_map[1]
        else:
            forward_out_map = out_map
            backward_out_map = out_map
        K.copy_edge_reduce(
            reducer, spmat[0], spmat[1], spmat[2], spmat[3], edge_map[0],
            edge_data_nd, forward_out_map, out_data_nd)
        # save_for_backward can only save variables
        ctx.backward_cache = (reducer, spmat, edge_map, backward_out_map,
                              edge_data_nd, out_data_nd)
        return out_data

    @staticmethod
    def backward(ctx, grad_out):
        reducer, spmat, edge_map, backward_out_map, edge_data, out_data \
            = ctx.backward_cache
        ctx.backward_cache = None
        grad_edge = None
        grad_out_nd = zerocopy_to_dgl_ndarray(grad_out)
        if ctx.needs_input_grad[2]:
            grad_edge = grad_out.new_empty(edge_data.shape)
            K.backward_copy_edge_reduce(
                reducer, spmat[0], spmat[1], spmat[2], spmat[3], edge_map[1],
                backward_out_map, edge_data, out_data, grad_out_nd,
                zerocopy_to_dgl_ndarray(grad_edge))
        return None, None, grad_edge, None, None, None


src_op_edge_reduce = SrcOpEdgeReduce.apply
src_op_dst_reduce = SrcOpDstReduce.apply
copy_src_reduce = CopySrcReduce.apply
copy_edge_reduce = CopyEdgeReduce.apply

def _reduce_grad(grad, src_shape):
    grad_shape = grad.shape[1:]
    src_shape = src_shape[1:]
    if src_shape == grad_shape:
        # no need to reduce
        return grad
    num_to_squeeze = len(grad_shape) - len(src_shape)
    # pad src_shape
    src_shape = (1,) * num_to_squeeze + src_shape
    reduce_idx = th.nonzero(th.tensor(grad_shape) - th.tensor(src_shape))
    reduce_idx += 1  # skip batch dim
    grad = grad.sum(dim=tuple(reduce_idx), keepdim=True)
    for dim in range(1, num_to_squeeze + 1):
        grad = grad.squeeze(dim)
    return grad
