import jax
from jax import numpy as jnp
from .tensor import SparseMatrix2D
from ...base import is_all, ALL
from ...sparse import _gspmm, _gsddmm, infer_broadcast_shape
from functools import partial

__all__ = ['gspmm', 'gsddmm', 'edge_softmax']


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


@partial(jax.jit, static_argnums=(5, 6))
def _jax_gspmm(X, Y, Z, dst_idxs, src_idxs, _op, _reduce_op):
    def body_fun(edge_idx, Z):
        dst_idx = dst_idxs[edge_idx]
        src_idx = src_idxs[edge_idx]
        _X = X[src_idx]
        _Y = Y[edge_idx]
        _Z = _op(_X, _Y)
        Z = _reduce_op(Z, dst_idx, _Z)
        return Z

    Z = jax.lax.fori_loop(
        lower=0,
        upper=dst_idxs.shape[0],
        body_fun=body_fun,
        init_val=Z,
    )

    return Z

@partial(jax.jit, static_argnums=(4, 5))
def _jax_gspmm_only_u(X, Z, dst_idxs, src_idxs, _op, _reduce_op):
    def body_fun(edge_idx, Z):
        dst_idx = dst_idxs[edge_idx]
        src_idx = src_idxs[edge_idx]
        _X = X[src_idx]
        _Z = _op(_X, None)
        Z = _reduce_op(Z, dst_idx, _Z)
        return Z

    Z = jax.lax.fori_loop(
        lower=0,
        upper=dst_idxs.shape[0],
        body_fun=body_fun,
        init_val=Z,
    )

    return Z

@partial(jax.jit, static_argnums=(4, 5))
def _jax_gspmm_only_e(Y, Z, dst_idxs, src_idxs, _op, _reduce_op):
    def body_fun(edge_idx, Z):
        dst_idx = dst_idxs[edge_idx]
        _Y = Y[edge_idx]
        _Z = _op(None, _Y)
        Z = _reduce_op(Z, dst_idx, _Z)
        return Z

    Z = jax.lax.fori_loop(
        lower=0,
        upper=dst_idxs.shape[0],
        body_fun=body_fun,
        init_val=Z,
    )

    return Z

def gspmm(gidx, op, reduce_op, X, Y):
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

    try:
        if use_u:
            ctx = X.device_buffer.device()
        elif use_e:
            ctx = Y.device_buffer.device()
    except:
        ctx = jax.devices('cpu')[0]

    a, _ = gidx.adjacency_matrix(0, False, ctx)
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

    return _jax_gspmm(X, Y, Z, dst_idxs, src_idxs, _op, _reduce_op)

def gsddmm(gidx, op, X, Y, lhs_target, rhs_target):
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

    if use_lhs:
        ctx = X.device_buffer.device()
    elif use_rhs:
        ctx = Y.device_buffer.device()

    a, _ = gidx.adjacency_matrix(0, False, ctx)
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
