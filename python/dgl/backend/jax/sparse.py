import numpy as onp
import jax
import jax.numpy as jnp
from functools import partial
from distutils.version import LooseVersion
from ...base import is_all, ALL
from ...sparse import infer_broadcast_shape
from ...sparse import _gspmm, _gspmm_hetero, _gsddmm, _gsddmm_hetero, _segment_reduce, _bwd_segment_cmp
from ...sparse import _csrmm, _csrsum, _csrmask, _scatter_add, _update_grad_minmax_hetero
from ...sparse import _gather_mm, _gather_mm_scatter, _segment_mm, _segment_mm_backward_B
from ...sparse import _gspmm, _gspmm_hetero, _gsddmm, _gsddmm_hetero, _segment_reduce, _bwd_segment_cmp, _edge_softmax_forward, _edge_softmax_backward
from ...sparse import _csrmm, _csrsum, _csrmask, _scatter_add, _update_grad_minmax_hetero
from ...heterograph_index import create_unitgraph_from_csr
from jax.experimental import host_callback as hcb

# __all__ = ['gspmm', 'gsddmm', 'gspmm_hetero', 'gsddmm_hetero', 'edge_softmax', 'edge_softmax_hetero',
#            'segment_reduce', 'scatter_add', 'csrmm', 'csrsum', 'csrmask', 'gather_mm', 'segment_mm']

__all__ = ['gspmm', 'gsddmm', 'edge_softmax', 'segment_reduce', 'scatter_add',
           'csrmm', 'csrsum', 'csrmask']

from ... import utils as dgl_utils
from ... import heterograph_index
from ...heterograph_index import HeteroGraphIndex
#
def _flatten_HeteroGraphIndex(gidx):
    adj, _ = gidx.adjacency_matrix(
        etype=0,
        transpose=False,
        ctx=jax.devices('cpu')[0],
    )

    srctype, dsttype = gidx.metagraph.find_edge(0)
    num_src = gidx.number_of_nodes(srctype)
    num_dst = gidx.number_of_nodes(dsttype)

    num_ntypes = 2 - int(srctype == dsttype)

    idx = adj.index

    u = idx[0, :]
    v = idx[1, :]
    return ((u, v, num_src, num_dst, num_ntypes), None)

def _unflatten_HeteroGraphIndex(_, children):
    u, v, num_src, num_dst, num_ntypes = children
    if u is None or v is None:
        return None

    return heterograph_index.create_unitgraph_from_coo(
        num_ntypes, num_src, num_dst, u, v, ["coo", "csr"],
    )

jax.tree_util.register_pytree_node(
    HeteroGraphIndex,
    _flatten_HeteroGraphIndex,
    _unflatten_HeteroGraphIndex,
)

@jax.jit
def scatter_add(x, index, source):
    pointers = jnp.meshgrid(*[jnp.arange(axis) for axis in x.shape], sparse=True, indexing="ij")
    pointers[0] = index
    x = x.at[tuple(pointers)].add(source)
    return x

@jax.jit
def _gather(x, index):
    pointers = jnp.meshgrid(*[jnp.arange(axis) for axis in x.shape], sparse=True, indexing="ij")
    pointers[0] = index
    return x[tuple(pointers)]

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
    reduce_idx = jnp.nonzero(jnp.array(grad_shape) - jnp.array(in_shape))[0]
    reduce_idx += 1  # skip batch dim
    if len(reduce_idx) > 0:
        grad = grad.sum(axis=tuple(reduce_idx), keepdims=True)
    return grad.reshape((-1, *shape[1:]))


def _need_reduce_last_dim(ufeat, efeat):
    """Indicates whether to reduce the last dimension on edges
    in the backward pass of spmm,
    if so, use dot instead of mul."""
    if ufeat is None or efeat is None:
        return False
    ushp = ufeat.shape
    eshp = efeat.shape
    return ushp[1:-1] == eshp[1:-1] and eshp[-1] == 1 and ushp[-1] > 1


def _expand(x, shape):
    # return x.expand(-1, *shape)
    return jnp.broadcast_to(x, (x.shape[0], *shape))

def spmm_cache_X(binary_op, reduce_op, req_grad_X, req_grad_Y):
    """Rules to identify whether to cache X in SpMM forward stage."""
    if binary_op != 'copy_lhs' and req_grad_Y:
        if reduce_op == 'sum':
            return True
        else:
            if binary_op == 'mul':
                return True
    return False


def spmm_cache_Y(binary_op, reduce_op, req_grad_X, req_grad_Y):
    """Rules to identify whether to cache Y in SpMM forward stage."""
    if binary_op != 'copy_rhs' and req_grad_X:
        if reduce_op == 'sum':
            if binary_op in ['mul', 'add']:
                return True
        else:
            if binary_op == 'mul':
                return True
    return False


def spmm_cache_argX(binary_op, reduce_op, req_grad_X, req_grad_Y):
    """Rules to identify whether to cache argX in SpMM forward stage."""
    if req_grad_X or req_grad_Y:
        if reduce_op in ['min', 'max']:
            return True
    return False


def spmm_cache_argY(binary_op, reduce_op, req_grad_X, req_grad_Y):
    """Rules to identify whether to cache argY in SpMM forward stage."""
    if req_grad_X or req_grad_Y:
        if reduce_op in ['min', 'max']:
            return True
    return False

@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def GSpMM(gidx, op, reduce_op, X, Y):
    out, (argX, argY) = _gspmm(gidx, op, reduce_op, X, Y)
    return out

def GSpMM_fwd(gidx, op, reduce_op, X, Y):
    out, (argX, argY) = _gspmm(gidx, op, reduce_op, X, Y)
    reduce_last = _need_reduce_last_dim(X, Y)
    X_shape = X.shape if X is not None else None
    Y_shape = Y.shape if Y is not None else None
    req_grad_X = True # X.requires_grad if X is not None else False
    req_grad_Y = True # Y.requires_grad if Y is not None else False
    if not spmm_cache_X(op, reduce_op, req_grad_X, req_grad_Y):
        X = None
    if not spmm_cache_Y(op, reduce_op, req_grad_X, req_grad_Y):
        Y = None
    if not spmm_cache_argX(op, reduce_op, req_grad_X, req_grad_Y):
        argX = None
    if not spmm_cache_argY(op, reduce_op, req_grad_X, req_grad_Y):
        argY = None
    cache = (X, Y, argX, argY, X_shape, Y_shape)
    return out, cache

def GSpMM_bwd(gidx, op, reduce_op, cache, dZ):
    X, Y, argX, argY, X_shape, Y_shape = cache
    dtype, device = dZ.dtype, dZ.device
    reduce_last = _need_reduce_last_dim(X, Y)
    req_grad_X = True # X.requires_grad if X is not None else False
    req_grad_Y = True # Y.requires_grad if Y is not None else False
    if op != 'copy_rhs':
        g_rev = gidx.reverse()
        if reduce_op == 'sum':
            if op == 'mul':
                dX = gspmm(g_rev, 'mul', 'sum', dZ, Y)
            elif op == 'add':
                dX = gspmm(g_rev, 'copy_lhs', 'sum', dZ, Y)
            elif op == 'copy_lhs':
                dX = gspmm(g_rev, 'copy_lhs', 'sum', dZ, None)
        else:  # max/min
            dX = jnp.zeros((X_shape[0],) + dZ.shape[1:], dtype=dtype)
            if op == 'mul':
                # grad = _expand(Y, dZ.shape[1:]).gather(
                #     0, argY.long()) * dZ

                # grad = jnp.take(
                #     _expand(Y, dZ.shape[1:]),
                #     argY,
                #     axis=0
                # ) * dZ

                grad = _gather(
                    _expand(Y, dZ.shape[1:]),
                    argY,
                ) * dZ

                # dX.scatter_add_(0, argX.long(), grad)
                dX = scatter_add(dX, argX, grad)
            elif op in ['add', 'copy_lhs']:
                # dX.scatter_add_(0, argX.long(), dZ)
                dX = scatter_add(dX, argX, dZ)


        dX = _reduce_grad(dX, X_shape)
    else:  # X has not gradient
        dX = None
    if op != 'copy_lhs':
        if reduce_op == 'sum':
            if op == 'mul' and reduce_last:
                dY = gsddmm(gidx, 'dot', X, dZ)
            elif op == 'mul':
                dY = gsddmm(gidx, 'mul', X, dZ)
            elif op in ['add', 'copy_rhs']:
                dY = gsddmm(gidx, 'copy_rhs', X, dZ)
        else:  # max/min
            dY = jnp.zeros((Y_shape[0],) + dZ.shape[1:], dtype=dtype)
            if op == 'mul':

                grad = _gather(
                    _expand(X, dZ.shape[1:]),
                    argX,
                ) * dZ

                # dY.scatter_add_(0, argY.long(), grad)
                dY = scatter_add(dY, argY, grad)
            elif op in ['add', 'copy_rhs']:
                # dY.scatter_add_(0, argY.long(), dZ)
                dY = scatter_add(dY, argY, dZ)
        dY = _reduce_grad(dY, Y_shape)
    else:  # Y has no gradient
        dY = None
    return dX, dY

GSpMM.defvjp(GSpMM_fwd, GSpMM_bwd)

def sddmm_cache_X(op, req_grad_X, req_grad_Y):
    """Rules to identify whether to cache X in SDDMM forward stage."""
    if op in ['mul', 'dot'] and req_grad_Y:
        return True
    return False


def sddmm_cache_Y(op, req_grad_X, req_grad_Y):
    """Rules to identify whether to cache Y in SDDMM forward stage."""
    if op in ['mul', 'dot'] and req_grad_X:
        return True
    return False

@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 4, 5))
def GSDDMM(gidx, op, X, Y, lhs_target, rhs_target):
    out = _gsddmm(gidx, op, X, Y, lhs_target, rhs_target)
    return out

def GSDDMM_fwd(gidx, op, X, Y, lhs_target, rhs_target):
    out = _gsddmm(gidx, op, X, Y, lhs_target, rhs_target)
    X_shape = X.shape if X is not None else None
    Y_shape = Y.shape if Y is not None else None
    req_grad_X = True # X.requires_grad if X is not None else False
    req_grad_Y = True # Y.requires_grad if Y is not None else False
    if not sddmm_cache_X(op, req_grad_X, req_grad_Y):
        X = None
    if not sddmm_cache_Y(op, req_grad_X, req_grad_Y):
        Y = None
    cache = X, Y, X_shape, Y_shape
    return out, cache

def GSDDMM_bwd(gidx, op, lhs_target, rhs_target, cache, dZ):
    X, Y, X_shape, Y_shape = cache
    if op != 'copy_rhs':
        if lhs_target in ['u', 'v']:
            _gidx = gidx if lhs_target == 'v' else gidx.reverse()
            if op in ['add', 'copy_lhs']:
                dX = gspmm(_gidx, 'copy_rhs', 'sum', None, dZ)
            else:  # mul, dot
                if rhs_target == lhs_target:
                    dX = gspmm(_gidx, 'copy_rhs', 'sum', None, dZ) *  Y
                elif rhs_target == 'e':
                    dX = gspmm(_gidx, 'copy_rhs', 'sum', None, dZ * Y)
                else:  # rhs_target = !lhs_target
                    dX = gspmm(_gidx, 'mul', 'sum', Y, dZ)
        else:  # lhs_target == 'e'
            if op in ['add', 'copy_lhs']:
                dX = dZ
            else:  # mul, dot
                dX = gsddmm(gidx, 'mul', dZ, Y, 'e', rhs_target)
        dX = _reduce_grad(dX, X_shape)
    else:
        dX = None
    if op != 'copy_lhs':
        if rhs_target in ['u', 'v']:
            _gidx = gidx if rhs_target == 'v' else gidx.reverse()
            if op in ['add', 'copy_rhs']:
                dY = gspmm(_gidx, 'copy_rhs', 'sum', None, dZ)
            else:  # mul, dot
                if lhs_target == rhs_target:
                    dY = gspmm(_gidx, 'copy_rhs', 'sum', None, dZ) * X
                elif lhs_target == 'e':
                    dY = gspmm(_gidx, 'copy_rhs', 'sum', None, dZ * X)
                else:  # rhs_target = !lhs_target
                    dY = gspmm(_gidx, 'mul', 'sum', X, dZ)
        else:
            if op in ['add', 'copy_rhs']:
                dY = dZ
            else:  # mul, dot
                dY = gsddmm(gidx, 'mul', dZ, X, 'e', lhs_target)
        dY = _reduce_grad(dY, Y_shape)
    else:
        dY = None
    return dX, dY

GSDDMM.defvjp(GSDDMM_fwd, GSDDMM_bwd)

@partial(jax.custom_vjp, nondiff_argnums=(0, 2, 3))
def EdgeSoftmax(gidx, score, eids, norm_by):
    """Forward function.

    Pseudo-code:

    .. code:: python

        score = dgl.EData(g, score)
        score_max = score.dst_max()  # of type dgl.NData
        score = score - score_max  # edge_sub_dst, ret dgl.EData
        score_sum = score.dst_sum()  # of type dgl.NData
        out = score / score_sum    # edge_div_dst, ret dgl.EData
        return out.data
    """
    # remember to save the graph to backward cache before making it
    # a local variable
    if not is_all(eids):
        gidx = gidx.edge_subgraph([eids], True).graph
    if norm_by == 'src':
        gidx = gidx.reverse()
    #Note: Now _edge_softmax_forward op only supports CPU
    #TODO(Zhejiang): We will support GPU in the future
    if jax.devices()[0].platform == "gpu":
        score_max = _gspmm(gidx, 'copy_rhs', 'max', None, score)[0]
        score = jnp.exp(_gsddmm(gidx, 'sub', score, score_max, 'e', 'v'))
        score_sum = _gspmm(gidx, 'copy_rhs', 'sum', None, score)[0]
        out = _gsddmm(gidx, 'div', score, score_sum, 'e', 'v')
    else:
        out = _edge_softmax_forward(gidx, score, 'copy_rhs')

    return out

def EdgeSoftmax_fwd(gidx, score, eids, norm_by):
    """Forward function.

    Pseudo-code:

    .. code:: python

        score = dgl.EData(g, score)
        score_max = score.dst_max()  # of type dgl.NData
        score = score - score_max  # edge_sub_dst, ret dgl.EData
        score_sum = score.dst_sum()  # of type dgl.NData
        out = score / score_sum    # edge_div_dst, ret dgl.EData
        return out.data
    """
    # remember to save the graph to backward cache before making it
    # a local variable
    if not is_all(eids):
        gidx = gidx.edge_subgraph([eids], True).graph
    if norm_by == 'src':
        gidx = gidx.reverse()
    #Note: Now _edge_softmax_forward op only supports CPU
    #TODO(Zhejiang): We will support GPU in the future
    if jax.devices()[0].platform == "gpu":
        score_max = _gspmm(gidx, 'copy_rhs', 'max', None, score)[0]
        score = jnp.exp(_gsddmm(gidx, 'sub', score, score_max, 'e', 'v'))
        score_sum = _gspmm(gidx, 'copy_rhs', 'sum', None, score)[0]
        out = _gsddmm(gidx, 'div', score, score_sum, 'e', 'v')
    else:
        out = _edge_softmax_forward(gidx, score, 'copy_rhs')

    cache = (out,)
    return out, cache

def EdgeSoftmax_bwd(gidx, eids, norm_by, cache, grad_out):
    """Backward function.

    Pseudo-code:

    .. code:: python

        g, out = ctx.backward_cache
        grad_out = dgl.EData(g, grad_out)
        out = dgl.EData(g, out)
        sds = out * grad_out  # type dgl.EData
        sds_sum = sds.dst_sum()  # type dgl.NData
        grad_score = sds - out * sds_sum  # multiple expressions
        return grad_score.data
    """
    if norm_by == "src":
        gidx = gidx.reverse()
    out = cache[0]
    sds = out * grad_out
    # print(out)
    # print(sds)
    #Note: Now _edge_softmax_backward op only supports CPU
    #TODO(Zhejiang): We will support GPU in the future
    if jax.devices()[0].platform == "gpu":
        accum = gspmm(gidx, 'copy_rhs', 'sum', None, sds)
        grad_score = sds - gsddmm(gidx, 'mul', out, accum, 'e', 'v')
    else:
        grad_score = _edge_softmax_backward(gidx, out, sds)
    # print(grad_score)
    return (grad_score, )

EdgeSoftmax.defvjp(EdgeSoftmax_fwd, EdgeSoftmax_bwd)

@partial(jax.custom_vjp, nondiff_argnums=(0,))
def SegmentReduce(op, x, offsets):
    y, arg = _segment_reduce(op, x, offsets)
    return y

def SegmentReduce_fwd(op, x, offsets):
    y, arg = _segment_reduce(op, x, offsets)
    cache = (arg, )
    return y, cache

def SegmentReduce_bwd(cache, dy):
    arg = cache[0]
    m = offsets[-1].item()
    if op == 'sum':
        offsets = offsets[1:]
        # To address the issue of trailing zeros, related issue:
        # https://github.com/dmlc/dgl/pull/2610
        indices = jnp.zeros(
            (m + 1,), dtype=offsets.dtype)

        indices = scatter_add(
            indices,
            offsets,
            jnp.ones_like(offsets),
        )

        indices = jnp.cumsum(indices, -1)[:-1]
        dx = dy[indices]
    else:
        dx = _bwd_segment_cmp(dy, arg, m)
    return None, dx, None

SegmentReduce.defvjp(SegmentReduce_fwd, SegmentReduce_bwd)

@partial(jax.custom_vjp, nondiff_argnums=(0, 2, 4))
def CSRMM(gidxA, A_weights, gidxB, B_weights, num_vtypes):
    gidxC, C_weights = _csrmm(gidxA, A_weights, gidxB, B_weights, num_vtypes)
    nrows, ncols, C_indptr, C_indices, C_eids = gidxC.adjacency_matrix_tensors(0, False, 'csr')
    # Note: the returned C_indptr, C_indices and C_eids tensors MUST be the same
    # as the underlying tensors of the created graph gidxC.
    return jnp.array(nrows), jnp.array(ncols), C_indptr, C_indices, C_eids, C_weights

def CSRMM_fwd(gidxA, A_weights, gidxB, B_weights, num_vtypes):
    gidxC, C_weights = _csrmm(gidxA, A_weights, gidxB, B_weights, num_vtypes)
    nrows, ncols, C_indptr, C_indices, C_eids = gidxC.adjacency_matrix_tensors(0, False, 'csr')
    # Note: the returned C_indptr, C_indices and C_eids tensors MUST be the same
    # as the underlying tensors of the created graph gidxC.
    cache = A_weights, B_weights, gidxC
    return (jnp.array(nrows), jnp.array(ncols), C_indptr, C_indices, C_eids, C_weights), cache

def CSRMM_bwd(gidxA, gidxB, num_vtypes, cache, dZ):
    dnrows, dncols, dC_indptr, dC_indices, dC_eids, dC_weights = dZ
    A_weights, B_weights, gidxC = cache
    dgidxA, dA_weights = csrmm(
        gidxC, dC_weights, gidxB.reverse(), B_weights, gidxA.number_of_ntypes())
    dgidxB, dB_weights = csrmm(
        gidxA.reverse(), A_weights, gidxC, dC_weights, gidxB.number_of_ntypes())
    dA_weights = csrmask(dgidxA, dA_weights, gidxA)
    dB_weights = csrmask(dgidxB, dB_weights, gidxB)
    return dA_weights, dB_weights

CSRMM.defvjp(CSRMM_fwd, CSRMM_bwd)

@partial(jax.custom_vjp, nondiff_argnums=(0,))
def CSRSum(gidxs, *weights):
    gidxC, C_weights = _csrsum(gidxs, weights)
    nrows, ncols, C_indptr, C_indices, C_eids = gidxC.adjacency_matrix_tensors(
        0, False, 'csr')
    # Note: the returned C_indptr, C_indices and C_eids tensors MUST be the same
    # as the underlying tensors of the created graph gidxC.
    return jnp.array(nrows), jnp.array(ncols), C_indptr, C_indices, C_eids, C_weights

def CSRSum_fwd(gidxs, *weights):
    # PyTorch tensors must be explicit arguments of the forward function
    gidxC, C_weights = _csrsum(gidxs, weights)
    nrows, ncols, C_indptr, C_indices, C_eids = gidxC.adjacency_matrix_tensors(
        0, False, 'csr')
    # Note: the returned C_indptr, C_indices and C_eids tensors MUST be the same
    # as the underlying tensors of the created graph gidxC.
    cache = (gidxC, )
    return (jnp.tensor(nrows), jnp.tensor(ncols), C_indptr, C_indices, C_eids, C_weights), cache

def CSRSum_bwd(gidx, cache, dZ):
    dnrows, dncols, dC_indptr, dC_indices, dC_eids, dC_weights = dZ
    gidxC = cache[0]
    return tuple(csrmask(gidxC, dC_weights, gidx) for gidx in gidxs)

CSRSum.defvjp(CSRMM_fwd, CSRMM_bwd)

@partial(jax.custom_vjp, nondiff_argnums=(0, 2))
def CSRMask(gidxA, A_weights, gidxB):
    return _csrmask(gidxA, A_weights, gidxB)

def CSRMask_fwd(gidxA, A_weights, gidxB):
    cache = None
    return _csrmask(gidxA, A_weights, gidxB), cache

def CSRMask_bwd(gidxA, gidxB, cache, dB_weights):
    gidxA, gidxB = cache
    return csrmask(gidxB, dB_weights, gidxA)

CSRMask.defvjp(CSRMask_fwd, CSRMask_bwd)

def gspmm(gidx, op, reduce_op, lhs_data, rhs_data):
    if op == 'sub':
        op = 'add'
        rhs_data = -rhs_data
    if op == 'div':
        op = 'mul'
        rhs_data = 1. / rhs_data

    return GSpMM(gidx, op, reduce_op, lhs_data, rhs_data)

def gsddmm(gidx, op, lhs_data, rhs_data, lhs_target='u', rhs_target='v'):
    if op == 'sub':
        op = 'add'
        rhs_data = -rhs_data
    if op == 'div':
        op = 'mul'
        rhs_data = 1. / rhs_data
    return GSDDMM(gidx, op, lhs_data, rhs_data, lhs_target, rhs_target)

def edge_softmax(gidx, logits, eids=ALL, norm_by='dst'):
    return EdgeSoftmax(gidx, logits, eids, norm_by)

def segment_reduce(op, x, offsets):
    return SegmentReduce(op, x, offsets)

# def scatter_add(x, idx, m):
#     return ScatterAdd(x, idx, m)

def csrmm(gidxA, A_weights, gidxB, B_weights, num_vtypes):
    nrows, ncols, C_indptr, C_indices, C_eids, C_weights = \
        CSRMM(gidxA, A_weights, gidxB, B_weights, num_vtypes)
    gidxC = create_unitgraph_from_csr(
        num_vtypes, nrows.item(), ncols.item(), C_indptr, C_indices, C_eids,
        ["coo", "csr", "csc"])
    return gidxC, C_weights

def csrsum(gidxs, weights):
    nrows, ncols, C_indptr, C_indices, C_eids, C_weights = CSRSum.apply(gidxs, *weights)
    gidxC = create_unitgraph_from_csr(
        gidxs[0].number_of_ntypes(), nrows.item(), ncols.item(), C_indptr, C_indices, C_eids,
        ["coo", "csr", "csc"])
    return gidxC, C_weights

def csrmask(gidxA, A_weights, gidxB):
    return CSRMask(gidxA, A_weights, gidxB)
