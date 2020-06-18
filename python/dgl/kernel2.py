"""Module for sparse matrix operators."""
# pylint: disable = invalid name
import dgl.ndarray as nd
from ._ffi.function import _init_api
from .base import DGLError
from .utils import to_dgl_context
from . import backend as F

def infer_broadcast_shape(op, shp1, shp2):
    """
    Parameters
    ----------
    op : str
    shp1 : tuple[int]
    shp2 : tuple[int]
    Returns
    -------
    shape after broadcasting
    """
    pad_shp1, pad_shp2 = shp1, shp2
    if op == "copy_u":
        return shp1
    if op == "copy_e":
        return shp2
    if len(shp1) > len(shp2):
        pad_shp2 = (1,) * (len(shp1) - len(shp2)) + shp2
    elif len(shp1) < len(shp2):
        pad_shp1 = (1,) * (len(shp2) - len(shp1)) + shp1
    for d1, d2 in zip(pad_shp1, pad_shp2):
        if d1 != d2 and d1 != 1 and d2 != 1:
            raise DGLError("Feature shapes {} and {} are not valid for broadcasting."
                           .format(shp1, shp2))
    rst = tuple(max(d1, d2) for d1, d2 in zip(pad_shp1, pad_shp2))
    return rst[:-1] + (1,) if op == "dot" else rst

def to_dgl_nd(x):
    """Convert tensor/None to ndarray."""
    return nd.NULL['int64'] if x is None else F.zerocopy_to_dgl_ndarray(x)

op_mapping = {
    '+': 'add',
    '-': 'sub',
    '*': 'mul',
    '/': 'div',
    '.': 'dot',
    'add': 'add',
    'sub': 'sub',
    'mul': 'mul',
    'div': 'div',
    'dot': 'dot',
    'copy_u': 'copy_u',
    'copy_e': 'copy_e'
}

def gspmm(g, op, reduce_op, u, e):
    """ Generalized Sparse Matrix Multiplication interface.

    Parameters
    ----------
    g : DGLHeteroGraph
        The input graph.
    op : str
        Binary operator, could be ``add``, ``sub``, ``mul``, ``div``, ``dot``, ``copy_u``,
        ``copy_e``, or their alias ``+``, ``-``, ``*``, ``/``, ``.``.
    reduce_op : str
        Reduce operator, could be ``sum``, ``max``, ``min``.
    u : tensor or None
        The feature on source nodes, could be None if op is ``copy_e``.
    e : tensor or None
        The feature on edges, could be None if op is ``copy_u``.

    Returns
    -------
    tensor
        The result tensor.
    """
    op = op_mapping[op]
    ctx = F.context(u) if u is not None else F.context(e)
    gidx = g._graph.get_unitgraph(0, to_dgl_context(ctx))
    dtype = F.dtype(u) if u is not None else F.dtype(e)
    use_u = (op != 'copy_e')
    use_e = (op != 'copy_u')
    u_shp = F.shape(u) if use_u else (0,)
    e_shp = F.shape(e) if use_e else (0,)
    v_shp = (g.number_of_dst_nodes(), ) +\
        infer_broadcast_shape(op, u_shp[1:], e_shp[1:])
    v = F.zeros(v_shp, dtype, ctx)
    use_cmp = reduce_op in ['max', 'min']
    arg_u = F.zeros(v_shp, g.idtype, ctx) if use_cmp and use_u else None
    arg_e = F.zeros(v_shp, g.idtype, ctx) if use_cmp and use_e else None
    _CAPI_DGLKernelSpMM(gidx, op, reduce_op,
                        to_dgl_nd(u), to_dgl_nd(e), to_dgl_nd(v),
                        to_dgl_nd(arg_u), to_dgl_nd(arg_e))
    return v, (arg_u, arg_e)

def gsddmm(g, op, u, v):
    """ Generalized Sampled-Dense-Dense Matrix Multiplication interface.

    Parameters
    ----------
    g : DGLHeteroGraph
        The input graph.
    op : str
        Binary operator, could be ``add``, ``sub``, ``mul``, ``div``, ``dot``, ``copy_u``,
        or their alias ``+``, ``-``, ``*``, ``/``, ``.``.
    u : tensor or None
        The feature on source nodes.
    v : tensor or None
        The feature on destination, could be None if op is ``copy_u``.

    Returns
    -------
    tensor
        The result tensor.
    """
    op = op_mapping[op]
    gidx = g._graph
    ctx = F.context(u)
    gidx = g._graph.get_unitgraph(0, to_dgl_context(ctx))
    dtype = F.dtype(u)
    u_shp = F.shape(u)
    v_shp = F.shape(v) if v is not None else (0,)
    e_shp = (g.number_of_edges(), ) +\
        infer_broadcast_shape(op, u_shp[1:], v_shp[1:])
    e = F.zeros(e_shp, dtype, ctx)
    _CAPI_DGLKernelSDDMM(gidx, op, to_dgl_nd(u), to_dgl_nd(v), to_dgl_nd(e))
    return e

_init_api("dgl.kernel2")
