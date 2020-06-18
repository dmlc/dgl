from ._ffi.function import _init_api
from .base import DGLError
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

to_dgl_nd = F.zerocopy_to_dgl_ndarray

def gspmm(g, op, reduce_op, u, e):
    gidx = g._graph
    ctx = F.context(u)
    use_u = (op != 'copy_e')
    use_e = (op != 'copy_u')
    u_shp = F.shape(u) if use_u else (0,)
    e_shp = F.shape(e) if use_e else (0,)
    v_shp = (g.number_of_dst_nodes(), ) +\
        infer_broadcast_shape(op, u_shp[1:], e_shp[1:])
    v = F.zeros(v_shp, F.dtype(u), ctx)
    use_cmp = (reduce_op == 'max' or reduce_op == 'min')
    arg_u = F.zeros(v_shp, g.idtype, ctx) if use_cmp and use_u else F.zeros((0,), g.idtype, ctx)
    arg_e = F.zeros(v_shp, g.idtype, ctx) if use_cmp and use_e else F.zeros((0,), g.idtype, ctx)
    _CAPI_DGLKernelSpMM(gidx, op, reduce_op,
            to_dgl_nd(u), to_dgl_nd(e), to_dgl_nd(v),
            to_dgl_nd(arg_u), to_dgl_nd(arg_e))
    return v, (arg_u, arg_e)

def gsddmm(g, op, u, v):
    gidx = g._graph
    ctx = F.context(u)
    u_shp = F.shape(u)
    v_shp = F.shape(v)
    e_shp = (g.number_of_edges(), ) +\
        infer_broadcast_shape(op, u_shp[1:], v_shp[1:])
    e = F.zeros(e_shp, F.dtype(u), ctx)
    _CAPI_DGLKernelSDDMM(gidx, op, to_dgl_nd(u), to_dgl_nd(v), to_dgl_nd(e))
    return e

_init_api("dgl.kernel2")
