import jax
from jax import numpy as jnp
from .tensor import SparseMatrix2D
from ...base import is_all, ALL
from ...sparse import _gspmm, _gsddmm, infer_broadcast_shape
from functools import partial

__all__ = ['gspmm', 'gsddmm', 'edge_softmax']


# =============================================================================
# REGISTER JAX PRIMITIVES
# =============================================================================
from ... import utils as dgl_utils
from ... import heterograph_index
from ...heterograph_index import HeteroGraphIndex
#
# def _flatten_HeteroGraphIndex(gidx):
#     adj, _ = gidx.adjacency_matrix(
#         etype=0,
#         transpose=False,
#         ctx=jax.devices('cpu')[0],
#     )
#
#     srctype, dsttype = gidx.metagraph.find_edge(0)
#     num_src = gidx.number_of_nodes(srctype)
#     num_dst = gidx.number_of_nodes(dsttype)
#
#     idx = adj.index
#
#     u = idx[0, :]
#     v = idx[1, :]
#     return ((u, v, num_src, num_dst), None)
#
# def _unflatten_HeteroGraphIndex(_, children):
#     u, v, num_src, num_dst = children
#     if u is None or v is None:
#         return None
#
#     return heterograph_index.create_unitgraph_from_coo(
#         1, num_src, num_dst, u, v, ["coo"],
#     )
#
# jax.tree_util.register_pytree_node(
#     HeteroGraphIndex,
#     _flatten_HeteroGraphIndex,
#     _unflatten_HeteroGraphIndex,
# )

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================

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
    reduce_idx = jnp.nonzero(jnp.tensor(grad_shape) - jnp.tensor(in_shape))
    reduce_idx += 1  # skip batch dim
    if len(reduce_idx) > 0:
        grad = grad.sum(dim=tuple(reduce_idx), keepdim=True)
    return grad.view(-1, *shape[1:])


def _need_reduce_last_dim(ufeat, efeat):
    """Indicates whether to reduce the last dimension on edges
    in the backward pass of spmm,
    if so, use dot instead of mul."""
    ushp = ufeat.shape
    eshp = efeat.shape
    return ushp[1:-1] == eshp[1:-1] and eshp[-1] == 1 and ushp[-1] > 1


def _muldiv(op, x):
    return 1. / x if op == 'div' else x


def _addsub(op, x):
    return -x if op == 'sub' else x


def _expand(x, shape):
    return x.expand(-1, *shape)

OPS = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
    "div": lambda x, y: x / y,
    "copy_lhs": lambda x, y: x,
    "copy_rhs": lambda x, y: y,
    "dot": lambda x, y: jnp.sum(x * y, -1, keepdims=True)
}


# OP_STRINGS = list(OPS.keys()) + [
#         "min", "max", "mean", "sum"
#     ] + [
#         "u", "e", "v", "src", "dst", "edge"
#     ]
#
# CHAR_2_IDX = dict(zip(OP_STRINGS, range(len(OP_STRINGS))))
# IDX_2_CHAR = dict(zip(range(len(OP_STRINGS)), OP_STRINGS))
#
# def _flatten_op_string(op_string):
#     children = [CHAR_2_IDX[op_string]]
#     return children, None
#
# def _unflatten_op_string(_, children):
#     try:
#         return IDX_2_CHAR[children[0]]
#     except:
#         return None
#
# jax.tree_util.register_pytree_node(
#     str,
#     _flatten_op_string,
#     _unflatten_op_string,
# )

# @partial(jax.jit, static_argnums=(5, 6))
# def _jax_gspmm_u_and_e(X, Y, Z, dst_idxs, src_idxs, _op, _reduce_op):
#     def body_fun(edge_idx, Z):
#         dst_idx = dst_idxs[edge_idx]
#         src_idx = src_idxs[edge_idx]
#         _X = X[src_idx]
#         _Y = Y[edge_idx]
#         _Z = _op(_X, _Y)
#         Z = _reduce_op(Z, dst_idx, _Z)
#         return Z
#
#     Z = jax.lax.fori_loop(
#         lower=0,
#         upper=dst_idxs.shape[0],
#         body_fun=body_fun,
#         init_val=Z,
#     )
#
#     return Z

# @partial(jax.jit, static_argnums=(4, 5))
# def _jax_gspmm_only_u(X, Z, dst_idxs, src_idxs, _op, _reduce_op):
#     def body_fun(edge_idx, Z):
#         dst_idx = dst_idxs[edge_idx]
#         src_idx = src_idxs[edge_idx]
#         _X = X[src_idx]
#         _Z = _op(_X, None)
#         Z = _reduce_op(Z, dst_idx, _Z)
#         return Z
#
#     Z = jax.lax.fori_loop(
#         lower=0,
#         upper=dst_idxs.shape[0],
#         body_fun=body_fun,
#         init_val=Z,
#     )
#
#     return Z
#
# @partial(jax.jit, static_argnums=(4, 5))
# def _jax_gspmm_only_e(Y, Z, dst_idxs, src_idxs, _op, _reduce_op):
#     def body_fun(edge_idx, Z):
#         dst_idx = dst_idxs[edge_idx]
#         _Y = Y[edge_idx]
#         _Z = _op(None, _Y)
#         Z = _reduce_op(Z, dst_idx, _Z)
#         return Z
#
#     Z = jax.lax.fori_loop(
#         lower=0,
#         upper=dst_idxs.shape[0],
#         body_fun=body_fun,
#         init_val=Z,
#     )
#
#     return Z

# @partial(jax.jit, static_argnums=(5, 6))
def _jax_gspmm_u_and_e(X, Y, Z, dst_idxs, src_idxs, _op, _reduce_op):
    edge_idxs = jnp.arange(dst_idxs.shape[0])
    _X = X[src_idxs]
    _Y = Y[edge_idxs]
    _Z = _op(_X, _Y)
    Z = _reduce_op(Z, dst_idxs, _Z)

    return Z

# @partial(jax.jit, static_argnums=(4, 5))
def _jax_gspmm_only_u(X, Z, dst_idxs, src_idxs, _op, _reduce_op):
    _X = X[src_idxs]
    _Z = _op(_X, None)
    Z = _reduce_op(Z, dst_idxs, _Z)
    return Z

# @partial(jax.jit, static_argnums=(4, 5))
def _jax_gspmm_only_e(Y, Z, dst_idxs, src_idxs, _op, _reduce_op):
    edge_idxs = jnp.arange(dst_idxs.shape[0])
    _Y = Y[edge_idxs]
    _Z = _op(None, _Y)
    Z = _reduce_op(Z, dst_idxs, _Z)
    return Z

@partial(jax.jit, static_argnums=(0, 1, 2))
def _jax_gspmm(gidx, op, reduce_op, X, Y):
    # out, (argX, argY) = _gspmm(gidx, op, reduce_op, X, Y)
    # return out

    if gidx.number_of_etypes() != 1:
        from .base import DGLError
        raise DGLError("We only support gspmm on graph with one edge type")

    use_u = op != 'copy_rhs'
    use_e = op != 'copy_lhs'

    if use_u:
        if X.ndim == 1:
            jnp.expand_dims(X, -1)
    if use_e:
        if Y.ndim == 1:
            jnp.expand_dims(Y, -1)

    u_shp = X.shape if use_u else (0,)
    e_shp = Y.shape if use_e else (0,)
    _, dsttype = gidx.metagraph.find_edge(0)
    v_shp = (gidx.number_of_nodes(dsttype), ) +\
        infer_broadcast_shape(op, u_shp[1:], e_shp[1:])
    dtype = X.dtype if use_u else Y.dtype

    a, _ = gidx.adjacency_matrix(0, False, jax.devices('cpu')[0])
    dst_idxs, src_idxs = a.index

    _reduce_op = "add" if reduce_op == "mean" or reduce_op == "sum" else reduce_op
    _reduce_op = getattr(jax.ops, "index_%s" % _reduce_op)

    if reduce_op == "mean" or reduce_op == "sum":
        Z = jnp.zeros(v_shp, dtype)

    elif reduce_op == "min":
        Z = jnp.ones(v_shp, dtype) * jnp.inf

    elif reduce_op == "max":
        Z = jnp.ones(v_shp, dtype) * (-jnp.inf)

    _op = OPS[op]

    if not use_u:
        return _jax_gspmm_only_e(Y, Z, dst_idxs, src_idxs, _op, _reduce_op)

    if not use_e:
        return _jax_gspmm_only_u(X, Z, dst_idxs, src_idxs, _op, _reduce_op)

    return _jax_gspmm_u_and_e(X, Y, Z, dst_idxs, src_idxs, _op, _reduce_op)

@partial(jax.jit, static_argnums=(0, 1, 4, 5))
def _jax_gsddmm(gidx, op, X, Y, lhs_target, rhs_target):
    # out = _gsddmm(gidx, op, X, Y, lhs_target, rhs_target)
    # return out
    if gidx.number_of_etypes() != 1:
        from .base import DGLError
        raise DGLError("We only support gsddmm on graph with one edge type")
    use_lhs = op != 'copy_rhs'
    use_rhs = op != 'copy_lhs'
    expand_lhs, expand_rhs = False, False
    # deal with scalar features.
    if use_lhs:
        if X.ndim == 1:
            X = jnp.expand_dims(X, -1)
            expand_lhs = True
    if use_rhs:
        if Y.ndim == 1:
            Y = jnp.expand_dims(Y, -1)
            expand_rhs = True

    a, _ = gidx.adjacency_matrix(0, False, jax.devices('cpu')[0])
    dst_idxs, src_idxs = a.index
    edge_idxs = jnp.arange(dst_idxs.shape[0])
    idxs_mapping = {
        'u': src_idxs,
        'e': edge_idxs,
        'v': dst_idxs,
        'src': src_idxs,
        'edge': edge_idxs,
        'dst': dst_idxs,
    }

    if X is not None:
        _X = jnp.take(X, idxs_mapping[lhs_target], axis=0)
    else:
        _X = None

    if Y is not None:
        _Y = jnp.take(Y, idxs_mapping[rhs_target], axis=0)
    else:
        _Y = None

    Z = OPS[op](_X, _Y)

    if (expand_lhs or not use_lhs) and (expand_rhs or not use_rhs):
        Z = jnp.expand_dims(Z, -1)

    return Z

# ==============================
# IMPLMENTATION USING CUSTOM VJP
# ==============================
# @partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
# def _dgl_gspmm(gidx, op, reduce_op, X, Y):
#     out, (argX, argY) = _gspmm(gidx, op, reduce_op, X, Y)
#     return out
#
# # def _dgl_gspmm_vjp_forward(gidx, op, reduce_op, X, Y):
# #     out, (argX, argY) = _gspmm(gidx, op, reduce_op, X, Y)
# #     res = gidx, op, reduce_op, X, Y, argX, argY
# #     return out, res
#
# def _dgl_gspmm_vjp_backward(res, dZ):
#     gidx, op, reduce_op, X, Y, argX, argY = res
#
#     if op != 'copy_rhs':
#         g_rev = gidx.reverse()
#         if reduce_op == 'sum':
#             if op in ['mul', 'div']:
#                 dX = gspmm(g_rev, 'mul', 'sum', dZ, _muldiv(op, Y))
#             elif op in ['add', 'sub']:
#                 dX = gspmm(g_rev, 'copy_lhs', 'sum', dZ, Y)
#             elif op == 'copy_lhs':
#                 dX = gspmm(g_rev, 'copy_lhs', 'sum', dZ, None)
#         else:  # max/min
#             dX = jnp.zeros((X.shape[0],) + dZ.shape[1:],
#                           dtype=X.dtype, device=X.device)
#             if op in ['mul', 'div']:
#                 grad = _muldiv(op, _expand(Y, dZ.shape[1:]).gather(
#                     0, argY.long())) * dZ
#                 dX.scatter_add_(0, argX.long(), grad)
#             elif op in ['add', 'sub', 'copy_lhs']:
#                 dX.scatter_add_(0, argX.long(), dZ)
#         dX = _reduce_grad(dX, X.shape)
#     else:  # X has not gradient
#         dX = None
#     if op != 'copy_lhs':
#         if reduce_op == 'sum':
#             if op == 'mul' and _need_reduce_last_dim(X, Y):
#                 dY = gsddmm(gidx, 'dot', X, dZ)
#             elif op in ['mul', 'div']:
#                 dY = gsddmm(gidx, 'mul', X, dZ)
#                 if op == 'div':
#                     dY = -dY / (Y ** 2)
#             elif op in ['add', 'sub', 'copy_rhs']:
#                 dY = gsddmm(gidx, 'copy_rhs', X, _addsub(op, dZ))
#         else:  # max/min
#             dY = jnp.zeros((Y.shape[0],) + dZ.shape[1:],
#                           dtype=Y.dtype, device=Y.device)
#             if op in ['mul',  'div']:
#                 grad = _expand(X, dZ.shape[1:]).gather(
#                     0, argX.long()) * dZ
#                 dY.scatter_add_(0, argY.long(), grad)
#                 if op == 'div':
#                     dY = -dY / (Y ** 2)
#             elif op in ['add', 'sub', 'copy_rhs']:
#                 dY.scatter_add_(0, argY.long(), _addsub(op, dZ))
#         dY = _reduce_grad(dY, Y.shape)
#     else:  # Y has no gradient
#         dY = None
#     return dX, dY
#
#
# def _dgl_gspmm_vjp_foward_and_backward(gidx, op, reduce_op, X, Y):
#     out, (argX, argY) = _gspmm(gidx, op, reduce_op, X, Y)
#
#     def grad(dZ):
#         if op != 'copy_rhs':
#             g_rev = gidx.reverse()
#             if reduce_op == 'sum':
#                 if op in ['mul', 'div']:
#                     dX = gspmm(g_rev, 'mul', 'sum', dZ, _muldiv(op, Y))
#                 elif op in ['add', 'sub']:
#                     dX = gspmm(g_rev, 'copy_lhs', 'sum', dZ, Y)
#                 elif op == 'copy_lhs':
#                     dX = gspmm(g_rev, 'copy_lhs', 'sum', dZ, None)
#             else:  # max/min
#                 dX = jnp.zeros((X.shape[0],) + dZ.shape[1:],
#                               dtype=X.dtype, device=X.device)
#                 if op in ['mul', 'div']:
#                     grad = _muldiv(op, _expand(Y, dZ.shape[1:]).gather(
#                         0, argY.long())) * dZ
#                     dX.scatter_add_(0, argX.long(), grad)
#                 elif op in ['add', 'sub', 'copy_lhs']:
#                     dX.scatter_add_(0, argX.long(), dZ)
#             dX = _reduce_grad(dX, X.shape)
#         else:  # X has not gradient
#             dX = None
#         if op != 'copy_lhs':
#             if reduce_op == 'sum':
#                 if op == 'mul' and _need_reduce_last_dim(X, Y):
#                     dY = gsddmm(gidx, 'dot', X, dZ)
#                 elif op in ['mul', 'div']:
#                     dY = gsddmm(gidx, 'mul', X, dZ)
#                     if op == 'div':
#                         dY = -dY / (Y ** 2)
#                 elif op in ['add', 'sub', 'copy_rhs']:
#                     dY = gsddmm(gidx, 'copy_rhs', X, _addsub(op, dZ))
#             else:  # max/min
#                 dY = jnp.zeros((Y.shape[0],) + dZ.shape[1:],
#                               dtype=Y.dtype, device=Y.device)
#                 if op in ['mul',  'div']:
#                     grad = _expand(X, dZ.shape[1:]).gather(
#                         0, argX.long()) * dZ
#                     dY.scatter_add_(0, argY.long(), grad)
#                     if op == 'div':
#                         dY = -dY / (Y ** 2)
#                 elif op in ['add', 'sub', 'copy_rhs']:
#                     dY.scatter_add_(0, argY.long(), _addsub(op, dZ))
#             dY = _reduce_grad(dY, Y.shape)
#         else:  # Y has no gradient
#             dY = None
#         return dX, dY
#
#     return out, grad
#
# # @partial(jax.custom_vjp, nondiff_argnums=(0, 1, 4, 5))
# def _dgl_gsddmm(gidx, op, X, Y, lhs_target, rhs_target):
#     out, _ = _gsddmm(gidx, op, X, Y, lhs_target, rhs_target)
#     return out
#
# # @partial(jax.jit, static_argnums=(0, 1, 4, 5))
# # def _dgl_gsddmm_vjp_forward(gidx, op, X, Y, lhs_target, rhs_target):
# #     out = _gsddmm(gidx, op, X, Y, lhs_target, rhs_target)
# #     res = gidx, op, X, Y, lhs_target, rhs_target
# #     return out, res
#
# # @partial(jax.jit, static_argnums=(0,))
# def _dgl_gsddmm_vjp_backward(res, dZ):
#     gidx, op, X, Y, lhs_target, rhs_target = res
#     if op != 'copy_rhs':
#         if lhs_target in ['u', 'v']:
#             _gidx = gidx if lhs_target == 'v' else gidx.reverse()
#             if op in ['add', 'sub', 'copy_lhs']:
#                 dX = _gspmm_primitive(_gidx, 'copy_rhs', 'sum', None, dZ)
#             else:  # mul, div, dot
#                 if rhs_target == lhs_target:
#                     dX = _gspmm_primitive(_gidx, 'copy_rhs', 'sum', None, dZ) * _muldiv(op, Y)
#                 elif rhs_target == 'e':
#                     dX = _gspmm_primitive(_gidx, 'copy_rhs', 'sum', None, dZ * _muldiv(op, Y))
#                 else:  # rhs_target = !lhs_target
#                     dX = _gspmm_primitive(_gidx, 'mul', 'sum', _muldiv(op, Y), dZ)
#         else:  # lhs_target == 'e'
#             if op in ['add', 'sub', 'copy_lhs']:
#                 dX = dZ
#             else:  # mul, div, dot
#                 dX = gsddmm(gidx, 'mul', dZ, _muldiv(op, Y), 'e', rhs_target)
#         dX = _reduce_grad(dX, X.shape)
#     else:
#         dX = None
#     if op != 'copy_lhs':
#         if rhs_target in ['u', 'v']:
#             _gidx = gidx if rhs_target == 'v' else gidx.reverse()
#             if op in ['add', 'sub', 'copy_rhs']:
#                 dY = _gspmm_primitive(_gidx, 'copy_rhs', 'sum', None, _addsub(op, dZ))
#             else:  # mul, div, dot
#                 if lhs_target == rhs_target:
#                     dY = _gspmm_primitive(_gidx, 'copy_rhs', 'sum', None, dZ) * X
#                 elif lhs_target == 'e':
#                     dY = _gspmm_primitive(_gidx, 'copy_rhs', 'sum', None, dZ * X)
#                 else:  # rhs_target = !lhs_target
#                     dY = _gspmm_primitive(_gidx, 'mul', 'sum', X, dZ)
#                 if op == 'div':
#                     dY = -dY / (Y ** 2)
#         else:
#             if op in ['add', 'sub', 'copy_rhs']:
#                 dY = _addsub(op, dZ)
#             else:  # mul, div, dot
#                 dY = gsddmm(gidx, 'mul', dZ, X, 'e', lhs_target)
#                 if op == 'div':
#                     dY = -dY / (Y ** 2)
#         dY = _reduce_grad(dY, Y.shape)
#     else:
#         dY = None
#     return dX, dY
#
# # ==============================
# # IMPLMENTATION USING PRIMITIVES
# # ==============================
# def _gspmm_abstract_eval(gidx, op, reduce_op, X, Y):
#     use_u = op != 'copy_rhs'
#     use_e = op != 'copy_lhs'
#
#     u_shp = X.shape if use_u else (0,)
#     e_shp = Y.shape if use_e else (0,)
#
#     if X.ndim == 1 and use_u == True:
#         u_shp.append(1)
#
#     if Y.ndim == 1 and use_e == True:
#         e_shp.append(1)
#
#     _, dsttype = gidx.metagraph.find_edge(0)
#     v_shp = (gidx.number_of_nodes(dsttype), ) +\
#         infer_broadcast_shape(op, u_shp[1:], e_shp[1:])
#     dtype = X.dtype if use_u else Y.dtype
#     return jax.abstract_arrays.ShapedArray(v_shp, dtype)
#
# def _gsddmm_abstract_eval(gidx, op, X, Y, lhs_target, rhs_target):
#     use_lhs = op != 'copy_rhs'
#     use_rhs = op != 'copy_lhs'
#
#     expand_lhs, expand_rhs = False, False
#     # deal with scalar features.
#     if use_lhs:
#         if X.ndim == 1:
#             expand_lhs = True
#         ctx = X.device_buffer.device()
#         dtype = X.dtype
#
#     if use_rhs:
#         if Y.ndim == 1:
#             expand_rhs = True
#         ctx = Y.device_buffer.device()
#         dtype = Y.shape
#
#     a, _ = gidx.adjacency_matrix(0, False, ctx)
#     n_nodes = a.shape[0]
#     n_edges = a.nnz
#
#     idxs_mapping = {
#         'u': n_nodes,
#         'e': n_edges,
#         'v': n_nodes,
#         'src': n_nodes,
#         'edge': n_edges,
#         'dst': n_edges,
#     }
#
#     v_shp = [idxs_mapping[lhs_target], idxs_mapping[rhs_target]]
#
#     if (expand_lhs or not use_lhs) and (expand_rhs or not use_rhs):
#         v_shp.append(1)
#
#     return jax.abstract_arrays.ShapedArray(v_shp, dtype)
#

# from jax.interpreters import ad
# _gspmm_primitive = jax.core.Primitive("gspmm")
# _gspmm_primitive_call = lambda *args, **kwargs: _gspmm_primitive.bind(
#     *args, **kwargs,
# )
# _gspmm_primitive.def_impl(_dgl_gspmm)
# _gspmm_primitive.def_abstract_eval(_gspmm_abstract_eval)
# ad.defvjp_all(_gspmm_primitive, _dgl_gspmm_vjp_foward_and_backward)
# gspmm = _gspmm_primitive_call

def gspmm(*args, **kwargs):
    return _jax_gspmm(*args, **kwargs)
    # return _dgl_gspmm(*args, **kwargs)

def gsddmm(*args, **kwargs):
    return _jax_gsddmm(*args, **kwargs)
    # return _dgl_gsddmm(*args, **kwargs)

def edge_softmax(gidx, logits, eids=ALL, norm_by='dst'):
    if not is_all(eids):
        gidx = gidx.edge_subgraph([eids], True).graph
    if norm_by == 'src':
        gidx = gidx.reverse()
    score_max = gspmm(gidx, 'copy_rhs', 'max', None, logits)
    score = jnp.exp(gsddmm(gidx, 'sub', logits, score_max, 'e', 'v'))
    score_sum = gspmm(gidx, 'copy_rhs', 'sum', None, score)
    out = gsddmm(gidx, 'div', score, score_sum, 'e', 'v')
    return out
