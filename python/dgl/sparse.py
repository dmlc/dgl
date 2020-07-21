"""Module for sparse matrix operators."""
# pylint: disable= invalid-name
from __future__ import absolute_import

import dgl.ndarray as nd
from ._ffi.function import _init_api
from .base import DGLError
from . import backend as F
import tvm
from featgraph.module import gsddmm, gspmm

def infer_broadcast_shape(op, shp1, shp2):
    r"""Check the shape validity, and infer the output shape given input shape and operator.
    Note the both :attr:`shp1`, :attr:`shp2` and the returned shape are feature
    shapes (i.e. we remove the first dimension, which correspond to graph statistics
    such as number of nodes, number of edges, etc.).

    We allow applying op on operands with different shapes, according to the
    broadcasting semantics of Numpy/Scipy:
    https://numpy.org/doc/stable/user/basics.broadcasting.html

    Parameters
    ----------
    op : str
        The binary op's name, could be `add`, `sub`, `mul`, `div`, `dot`, `copy_lhs`, `copy_rhs`.
    shp1 : tuple[int]
        The shape of lhs operand.
    shp2 : tuple[int]
        The shape of rhs operand.

    Returns
    -------
    tuple[int]
        shape after broadcasting
    """
    pad_shp1, pad_shp2 = shp1, shp2
    if op == "dot":
        if shp1[-1] != shp2[-1]:
            raise DGLError("Dot operator is only available for arrays with the "
                           "same size on last dimension, but got {} and {}."
                           .format(shp1, shp2))
    if op == "copy_lhs":
        return shp1
    if op == "copy_rhs":
        return shp2
    # operands are padded to have the same dimensionality with leading 1's.
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

def use_bcast(op, lhs_shape, rhs_shape):
    if op in ['copy_lhs', 'copy_rhs']:
        return False
    if len(lhs_shape) != len(rhs_shape):
        return True
    for i in range(len(lhs_shape)):
        if lhs_shape[i] != rhs_shape[i]:
            return True
    return False

def calc_bcast(op, lhs_shape, rhs_shape):
    # remove first dimension to only consider feature
    lhs_len, rhs_len = 1, 1
    for v in lhs_shape[1:]:
        lhs_len *= v
    for v in rhs_shape[1:]:
        rhs_len *= v
    lhs_ndim, rhs_ndim = len(lhs_shape), len(rhs_shape)
    if_bcast = use_bcast(op, lhs_shape[1:], rhs_shape[1:])
    reduce_size = 1
    lhs_off, rhs_off = [], []
    rst_out_len = 1
    if if_bcast:
        max_ndim = max(lhs_ndim, rhs_ndim)-1
        out_len = 1
        j = 0
        if op == 'dot':
            j += 1
            reduce_size = lhs_shape[lhs_ndim - 1]
        stride_l, stride_r = 1, 1
        lhs_off.append(0)
        rhs_off.append(0)
        while j < max_ndim:
            dl = 1 if lhs_ndim - 1 - j < 1 else lhs_shape[lhs_ndim-1-j]
            dr = 1 if rhs_ndim - 1 - j < 1 else rhs_shape[rhs_ndim-1-j]
            for i in range(1, max(dl, dr)):
                for k in range(out_len):
                    lhs_off.append(lhs_off[k] + i * (i < dl) * stride_l)
                    rhs_off.append(rhs_off[k] + i * (i < dr) * stride_r)
            out_len *= max(dl, dr)
            stride_l *= dl
            stride_r *= dr
            j += 1
        rst_out_len = out_len
    else:
        rst_out_len = rhs_len if op == 'copy_rhs' else lhs_len
        if op == 'dot':
            reduce_size = lhs_shape[lhs_ndim-1]
            rst_out_len //= reduce_size
    return if_bcast, lhs_len, rhs_len, rst_out_len, reduce_size, lhs_off, rhs_off

def to_dgl_nd(x):
    """Convert framework-specific tensor/None to dgl ndarray."""
    return nd.NULL['int64'] if x is None else F.zerocopy_to_dgl_ndarray(x)

def to_dgl_nd_for_write(x):
    """Convert framework-specific tensor/None to dgl ndarray for write."""
    return nd.NULL['int64'] if x is None else F.zerocopy_to_dgl_ndarray_for_write(x)

target_mapping = {
    'u': 0,
    'e': 1,
    'v': 2,
    'src': 0,
    'edge': 1,
    'dst': 2
}

compiled_gspmm_kernels = {}
compiled_gsddmm_kernels = {}

def _gspmm(gidx, op, reduce_op, u, e):
    r""" Generalized Sparse Matrix Multiplication interface. It takes the result of
    :attr:`op` on source node feature and edge feature, leads to a message on edge.
    Then aggregates the message by :attr:`reduce_op` on destination nodes.

    .. math::
        x_v = \psi_{(u, v, e)\in \mathcal{G}}(\rho(x_u, x_e))

    where :math:`x_v` is the returned feature on destination nodes, and :math`x_u`,
    :math:`x_e` refers to :attr:`u`, :attr:`e` respectively. :math:`\rho` means binary
    operator :attr:`op` and :math:`\psi` means reduce operator :attr:`reduce_op`,
    :math:`\mathcal{G}` is the graph we apply gspmm on: :attr:`g`.

    Note that this function does not handle gradients.

    Parameters
    ----------
    gidx : HeteroGraphIndex
        The input graph index.
    op : str
        The binary op's name, could be ``add``, ``sub``, ``mul``, ``div``, ``copy_lhs``,
        ``copy_rhs``.
    reduce_op : str
        Reduce operator, could be ``sum``, ``max``, ``min``.
    u : tensor or None
        The feature on source nodes, could be None if op is ``copy_rhs``.
    e : tensor or None
        The feature on edges, could be None if op is ``copy_lhs``.

    Returns
    -------
    tuple
        The returned tuple is composed of two elements:
        - The first element refers to the result tensor.
        - The second element refers to a tuple composed of arg_u and arg_e
          (which is useful when reducer is `min`/`max`).

    Notes
    -----
    This function does not handle gradients, and for scalar input features,
    we expand its dimension with an additional dimension of length one. (e.g.
    (90,) to (90, 1) for a graph with 90 nodes/edges).
    """
    if gidx.number_of_etypes() != 1:
        raise DGLError("We only support gsddmm on graph with one edge type")
    nnz = gidx.number_of_edges(0)
    if nnz <= 0:
        return None
    indptr, indices, data = map(lambda x: tvm.nd.from_dlpack(x.to_dlpack()), gidx.get_csc_dlpack(0))
    edge_mapping = F.zerocopy_from_dgl_ndarray(data).long()
    f_input = [indptr, indices]
    use_u = op != 'copy_rhs'
    use_e = op != 'copy_lhs'
    if use_u and F.ndim(u) == 1:
        u = F.unsqueeze(u, -1)
    if use_e:
        e = e[edge_mapping]
        if F.ndim(e) == 1:
            e = F.unsqueeze(e, -1)
    ctx = F.context(u) if use_u else F.context(e)
    feat_type = F.dtype(u) if use_u else F.dtype(e)
    u_shp = F.shape(u) if use_u else (0,)
    e_shp = F.shape(e) if use_e else (0,)
    indice_type = gidx.dtype
    key = (id(gidx), op, reduce_op, u_shp, e_shp, indice_type, feat_type)
    if key not in compiled_gspmm_kernels:
        srctype, dsttype = gidx.metagraph.find_edge(0)
        num_rows = gidx.number_of_nodes(dsttype)
        num_cols = gidx.number_of_nodes(srctype)
        target = F.device_type(ctx)
        if target == 'cuda':
            tvm_ctx = tvm.gpu(0)
        elif target == 'cpu':
            target = 'llvm'
            tvm_ctx = tvm.cpu(0)
        v_shp = (num_rows, ) +\
            infer_broadcast_shape(op, u_shp[1:], e_shp[1:])
        if_bcast, lhs_len, rhs_len, out_len, _, lhs_off, rhs_off = \
            calc_bcast(op, u_shp, e_shp)
        mod = gspmm.spmm(
            op, reduce_op, nnz, num_rows, num_cols,
            lhs_len, rhs_len, out_len, indice_type, str(feat_type),
            use_bcast=if_bcast, target=target
        )
        compiled = (mod, v_shp, lhs_len, rhs_len, out_len)
        if if_bcast:
            lhs_offset = tvm.nd.empty((out_len,), dtype=indice_type, ctx=tvm_ctx)
            rhs_offset = tvm.nd.empty((out_len,), dtype=indice_type, ctx=tvm_ctx)
            lhs_offset.copyfrom(lhs_off)
            rhs_offset.copyfrom(rhs_off)
            compiled += (lhs_offset, rhs_offset)
        compiled_gsddmm_kernels[key] = compiled
    else:
        compiled = compiled_gspmm_kernels[key]
    mod, v_shp, lhs_len, rhs_len, out_len = compiled[:5]
    if use_u:
        u = u.view((u_shp[0], lhs_len))
        f_input.append(tvm.nd.from_dlpack(to_dgl_nd(u).to_dlpack()))
    if use_e:
        e = e.view((e_shp[0], rhs_len))
        f_input.append(tvm.nd.from_dlpack(to_dgl_nd(e).to_dlpack()))
    if len(compiled) > 5:
        lhs_offset, rhs_offset = compiled[5], compiled[6]
        f_input += [lhs_offset, rhs_offset]
    idtype = getattr(F, gidx.dtype)
    arg_u, arg_e = None, None
    use_cmp = reduce_op != 'sum'
    if use_cmp:
        if use_u:
            arg_u = F.zeros(v_shp, idtype, ctx)
            arg_u = arg_u.view((v_shp[0], out_len))
            f_input.append(tvm.nd.from_dlpack(to_dgl_nd_for_write(arg_u).to_dlpack()))
        if use_e:
            arg_e = F.zeros(v_shp, idtype, ctx)
            arg_e = arg_e.view((v_shp[0], out_len))
            f_input.append(tvm.nd.from_dlpack(to_dgl_nd_for_write(arg_e).to_dlpack()))
    v = F.zeros(v_shp, feat_type, ctx)
    v = v.view((v_shp[0], out_len))
    f_input.append(tvm.nd.from_dlpack(to_dgl_nd_for_write(v).to_dlpack()))
    mod(*f_input)
    if use_u:
        u = u.view(u_shp)
        if use_cmp:
            arg_u = arg_u.view(v_shp)
    if use_e:
        e = e.view(e_shp)
        if use_cmp:
            arg_e = arg_e.view(v_shp)
            arg_e = edge_mapping[arg_e.long()]
    v = v.view(v_shp)
    return v, (arg_u, arg_e)

def _gsddmm(gidx, op, lhs, rhs, lhs_target='u', rhs_target='v'):
    r""" Generalized Sampled-Dense-Dense Matrix Multiplication interface. It
    takes the result of :attr:`op` on source node feature and destination node
    feature, leads to a feature on edge.

    .. math::
        x_{e} = \phi(x_u, x_e, x_v), \forall (u,e,v)\in \mathcal{G}

    where :math:`x_{e}` is the returned feature on edges and :math:`x_u`,
    :math:`x_v` refers to :attr:`u`, :attr:`v` respectively. :math:`\phi`
    is the binary operator :attr:`op`, and :math:`\mathcal{G}` is the graph
    we apply gsddmm on: :attr:`g`.

    Parameters
    ----------
    gidx : HeteroGraphIndex
        The input graph index.
    op : str
        Binary operator, could be ``add``, ``sub``, ``mul``, ``div``, ``dot``,
        ``copy_lhs``, ``copy_rhs``.
    lhs : tensor or None
        Left hand operand.
    rhs : tensor or None
        Right hand operand.
    lhs_target : str
        The target of left hand operand, could be ``src``, ``edge``, ``dst``
        or their alias ``u``, ``e``, ``v``.
    rhs_target : str
        The target of right hand operand, could be ``src``, ``edge``, ``dst``
        or their alias ``u``, ``e``, ``v``.

    Returns
    -------
    tensor
        The result tensor.

    Notes
    -----
    This function does not handle gradients, and for scalar input features,
    we expand its dimension with an additional dimension of length one. (e.g.
    (90,) to (90, 1) for a graph with 90 nodes/edges).
    """
    if gidx.number_of_etypes() != 1:
        raise DGLError("We only support gsddmm on graph with one edge type")
    nnz = gidx.number_of_edges(0)
    if nnz <= 0:
        return None
    lhs_target = target_mapping[lhs_target]
    rhs_target = target_mapping[rhs_target]
    row, col, _ = map(lambda x: tvm.nd.from_dlpack(x.to_dlpack()), gidx.get_coo_dlpack(0))
    f_input = []
    if lhs_target == 0 or rhs_target == 0:
        f_input.append(row)
    if lhs_target == 2 or rhs_target == 2:
        f_input.append(col)
    use_lhs = op != 'copy_rhs'
    use_rhs = op != 'copy_lhs'
    feat_type = F.dtype(lhs) if use_lhs else F.dtype(rhs)
    ctx = F.context(lhs) if use_lhs else F.context(rhs)
    if use_lhs and F.ndim(lhs) == 1:
        lhs = F.unsqueeze(lhs, -1)
        # f_input.append(tvm.nd.from_dlpack(to_dgl_nd(lhs).to_dlpack()))
    if use_rhs and F.ndim(rhs) == 1:
        rhs = F.unsqueeze(rhs, -1)
        # f_input.append(tvm.nd.from_dlpack(to_dgl_nd(rhs).to_dlpack()))
    lhs_shp = F.shape(lhs) if use_lhs else (0,)
    rhs_shp = F.shape(rhs) if use_rhs else (0,)
    indice_type = gidx.dtype
    # key = (nnz, op, lhs_target, lhs_shp, rhs_target, rhs_shp, indice_type, feat_type)
    key = (id(gidx), op, lhs_shp, rhs_shp, indice_type, feat_type)
    if key not in compiled_gsddmm_kernels:
        srctype, dsttype = gidx.metagraph.find_edge(0)
        num_cols = gidx.number_of_nodes(dsttype)
        num_rows = gidx.number_of_nodes(srctype)
        # print(num_rows, num_cols)
        target = F.device_type(ctx)
        if target == 'cuda':
            tvm_ctx = tvm.gpu(0)
        elif target == 'cpu':
            tvm_ctx = tvm.cpu(0)
            target = 'llvm'
        else:
            raise DGLError("We only support graph on cpu or gpu")
        out_shp = (gidx.number_of_edges(0), ) +\
            infer_broadcast_shape(op, lhs_shp[1:], rhs_shp[1:])
        if_bcast, lhs_len, rhs_len, out_len, reduce_size, lhs_off, rhs_off = \
            calc_bcast(op, lhs_shp, rhs_shp)
        mod = gsddmm.sddmm(
            op, nnz, num_rows, num_cols,
            lhs_len, rhs_len, out_len, str(indice_type), str(feat_type),
            reduce_size=reduce_size, lhs_target=lhs_target, 
            rhs_target=rhs_target, use_bcast=if_bcast, target=target
        )
        compiled = (mod, out_shp, lhs_len, rhs_len, out_len)
        if if_bcast:
            lhs_offset = tvm.nd.empty((out_len,), dtype=indice_type, ctx=tvm_ctx)
            rhs_offset = tvm.nd.empty((out_len,), dtype=indice_type, ctx=tvm_ctx)
            lhs_offset.copyfrom(lhs_off)
            rhs_offset.copyfrom(rhs_off)
            compiled += (lhs_offset, rhs_offset)
        compiled_gsddmm_kernels[key] = compiled
    else:
        compiled = compiled_gsddmm_kernels[key]
    mod, out_shp, lhs_len, rhs_len, out_len = compiled[:5]
    if use_lhs:
        lhs = lhs.view((lhs_shp[0], lhs_len))
        f_input.append(tvm.nd.from_dlpack(to_dgl_nd(lhs).to_dlpack()))
    if use_rhs:
        rhs = rhs.view((rhs_shp[0], rhs_len))
        f_input.append(tvm.nd.from_dlpack(to_dgl_nd(rhs).to_dlpack()))
    if len(compiled) > 5:
        lhs_offset, rhs_offset = compiled[5], compiled[6]
        f_input += [lhs_offset, rhs_offset]
    out = F.zeros(out_shp, feat_type, ctx)
    out = out.view((out_shp[0], out_len))
    f_input.append(tvm.nd.from_dlpack(to_dgl_nd_for_write(out).to_dlpack()))
    mod(*f_input)
    if use_lhs:
        lhs = lhs.view(lhs_shp)
    if use_rhs:
        rhs = rhs.view(rhs_shp)
    out = out.view(out_shp)
    return out

_init_api("dgl.sparse")