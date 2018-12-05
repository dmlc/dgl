from __future__ import absolute_import

import torch as th
from torch.utils import dlpack

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

def sparse_matrix(data, index, shape, force_format=False):
    fmt = index[0]
    if fmt != 'coo':
        raise TypeError('Pytorch backend only supports COO format. But got %s.' % fmt)
    # NOTE: use _sparse_coo_tensor_unsafe to avoid unnecessary boundary check
    spmat = th._sparse_coo_tensor_unsafe(index[1], data, shape)
    # No conversion is required.
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

def astype(input, ty):
    return input.type(ty)

def asnumpy(input):
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

def ones(shape, dtype, ctx):
    return th.ones(shape, dtype=dtype, device=ctx)

def spmm(x, y):
    return th.spmm(x, y)

def unsorted_1d_segment_sum(input, seg_id, n_segs, dim):
    y = th.zeros(n_segs, *input.shape[1:]).to(input)
    seg_id = seg_id.view((-1,) + (1,) * (input.dim() - 1)).expand_as(input)
    y = y.scatter_add_(dim, seg_id, input)
    return y

def unsorted_1d_segment_mean(input, seg_id, n_segs, dim):
    w = unsorted_1d_segment_sum(th.ones_like(seg_id), seg_id, n_segs, 0).to(input)
    w = w.clamp(min=1)   # remove 0 entries
    y = unsorted_1d_segment_sum(input, seg_id, n_segs, dim)
    y /= w.view((-1,) + (1,) * (y.dim() - 1))
    return y

def unique(input):
    return th.unique(input)

def full_1d(length, fill_value):
    return th.full((length,), fill_value)

def nonzero_1d(input):
    return th.nonzero(input).squeeze()

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

# create_immutable_graph_index not enabled
