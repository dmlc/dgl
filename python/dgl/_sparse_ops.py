"""Module for sparse matrix operators."""
# pylint: disable= invalid-name
from __future__ import absolute_import

from . import backend as F, ndarray as nd
from ._ffi.function import _init_api
from .base import DGLError


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
            raise DGLError(
                "Dot operator is only available for arrays with the "
                "same size on last dimension, but got {} and {}.".format(
                    shp1, shp2
                )
            )
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
            raise DGLError(
                "Feature shapes {} and {} are not valid for broadcasting.".format(
                    shp1, shp2
                )
            )
    rst = tuple(max(d1, d2) for d1, d2 in zip(pad_shp1, pad_shp2))
    return rst[:-1] + (1,) if op == "dot" else rst


def to_dgl_nd(x):
    """Convert framework-specific tensor/None to dgl ndarray."""
    return nd.NULL["int64"] if x is None else F.zerocopy_to_dgl_ndarray(x)


def to_dgl_nd_for_write(x):
    """Convert framework-specific tensor/None to dgl ndarray for write."""
    return (
        nd.NULL["int64"]
        if x is None
        else F.zerocopy_to_dgl_ndarray_for_write(x)
    )


def get_typeid_by_target(gidx, etid, target):
    """Find the src/dst/etype id based on the target 'u', 'v' or 'e'."""
    src_id, dst_id = gidx.metagraph.find_edge(etid)
    if target in [0, "u"]:
        return src_id
    if target in [2, "v"]:
        return dst_id
    return etid


target_mapping = {"u": 0, "e": 1, "v": 2, "src": 0, "edge": 1, "dst": 2}


def _edge_softmax_backward(gidx, out, sds):
    r"""Edge_softmax backward interface.

    Parameters
    ----------
    gidx : HeteroGraphIndex
        The input graph index.
    out : tensor
        The result of Edge_softmax during forward.
    sds : tensor
        The result of out * gradient.

    Returns
    -------
    The result of Edge_softmax during backward

    Notes
    -----
    This function does not support gpu op.
    """
    op = "copy_rhs"
    back_out = F.zeros_like(out)
    _CAPI_DGLKernelEdge_softmax_backward(
        gidx,
        op,
        to_dgl_nd(out),
        to_dgl_nd(sds),
        to_dgl_nd_for_write(back_out),
        to_dgl_nd(None),
    )
    return back_out


def _edge_softmax_forward(gidx, e, op):
    r"""Edge_softmax forward interface.

    Parameters
    ----------
    gidx : HeteroGraphIndex
        The input graph index.
    op : str
        The binary op's name, default as ``copy_rhs``.
    e : tensor or None
        The feature on edges.

    Returns
    -------
    The result of Edge_softmax during forward

    Notes
    -----
    This function does not support gpu op.
    """
    if F.ndim(e) == 1:
        e = F.unsqueeze(e, -1)
        expand = True
    else:
        expand = False
    myout = F.zeros_like(e)
    _CAPI_DGLKernelEdge_softmax_forward(
        gidx, op, to_dgl_nd(None), to_dgl_nd(e), to_dgl_nd_for_write(myout)
    )
    myout = F.squeeze(myout, -1) if expand else myout
    return myout


def _gspmm(gidx, op, reduce_op, u, e):
    r"""Generalized Sparse Matrix Multiplication interface. It takes the result of
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
    This function does not handle gradients.
    """
    if gidx.number_of_etypes() != 1:
        raise DGLError("We only support gspmm on graph with one edge type")
    use_u = op != "copy_rhs"
    use_e = op != "copy_lhs"
    if use_u and use_e:
        if F.dtype(u) != F.dtype(e):
            raise DGLError(
                "The node features' data type {} doesn't match edge"
                " features' data type {}, please convert them to the"
                " same type.".format(F.dtype(u), F.dtype(e))
            )
    # deal with scalar features.
    expand_u, expand_e = False, False
    if use_u:
        if F.ndim(u) == 1:
            u = F.unsqueeze(u, -1)
            expand_u = True
    if use_e:
        if F.ndim(e) == 1:
            e = F.unsqueeze(e, -1)
            expand_e = True

    ctx = F.context(u) if use_u else F.context(e)
    dtype = F.dtype(u) if use_u else F.dtype(e)
    u_shp = F.shape(u) if use_u else (0,)
    e_shp = F.shape(e) if use_e else (0,)
    _, dsttype = gidx.metagraph.find_edge(0)
    v_shp = (gidx.num_nodes(dsttype),) + infer_broadcast_shape(
        op, u_shp[1:], e_shp[1:]
    )
    v = F.zeros(v_shp, dtype, ctx)
    use_cmp = reduce_op in ["max", "min"]
    arg_u, arg_e = None, None
    idtype = getattr(F, gidx.dtype)
    if use_cmp:
        if use_u:
            arg_u = F.zeros(v_shp, idtype, ctx)
        if use_e:
            arg_e = F.zeros(v_shp, idtype, ctx)
    arg_u_nd = to_dgl_nd_for_write(arg_u)
    arg_e_nd = to_dgl_nd_for_write(arg_e)
    if gidx.num_edges(0) > 0:
        _CAPI_DGLKernelSpMM(
            gidx,
            op,
            reduce_op,
            to_dgl_nd(u if use_u else None),
            to_dgl_nd(e if use_e else None),
            to_dgl_nd_for_write(v),
            arg_u_nd,
            arg_e_nd,
        )
    # NOTE(zihao): actually we can avoid the following step, because arg_*_nd
    # refers to the data that stores arg_*. After we call _CAPI_DGLKernelSpMM,
    # arg_* should have already been changed. But we found this doesn't work
    # under Tensorflow when index type is int32. (arg_u and arg_e would be
    # all zero).
    # The workaround is proposed by Jinjing, and we still need to investigate
    # where the problem is.
    arg_u = None if arg_u is None else F.zerocopy_from_dgl_ndarray(arg_u_nd)
    arg_e = None if arg_e is None else F.zerocopy_from_dgl_ndarray(arg_e_nd)
    # To deal with scalar node/edge features.
    if (expand_u or not use_u) and (expand_e or not use_e):
        v = F.squeeze(v, -1)
    if expand_u and use_cmp:
        arg_u = F.squeeze(arg_u, -1)
    if expand_e and use_cmp:
        arg_e = F.squeeze(arg_e, -1)
    return v, (arg_u, arg_e)


def _gspmm_hetero(gidx, op, reduce_op, u_len, u_and_e_tuple):
    r"""Generalized Sparse Matrix Multiplication interface on heterogeneous graphs.
    It handles multiple node and edge types of the graph. For each edge type, it takes
    the result of :attr:`op` on source node feature and edge feature, and leads to a
    message on edge. Then it aggregates the message by :attr:`reduce_op` on the destination
    nodes of the etype.

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
    u_len : int
        The number of tensors in ``u`` (source node features)
    u_and_e_tuple : Tuple of tensors
        Tuple of source nodes' features and edges' features. ``u_and_e_tuple[:u_len]``
        stores the source nodes's features of all source node types. ``u_and_e_tuple[u_len:]``
        stores the edges's features of all the edge types.
        The source nodes' features of the soruce node types could be None if op is ``copy_rhs``.
        The edges' features of the edge types could be None if op is ``copy_lhs``.

    Returns
    -------
    tuple
        The returned tuple is composed of two elements:
        - The first element refers to the tuple of result tensors.
        - The second element refers to a tuple composed of arg_u and arg_e
          (which is useful when reducer is `min`/`max`).

    Notes
    -----
    This function does not handle gradients.
    """
    u_tuple, e_tuple = u_and_e_tuple[:u_len], u_and_e_tuple[u_len:]
    use_u = op != "copy_rhs"
    use_e = op != "copy_lhs"
    # TODO (Israt): Add check - F.dtype(u) != F.dtype(e):

    # deal with scalar features.
    expand_u, expand_e = False, False
    num_ntypes = gidx.number_of_ntypes()
    num_etypes = gidx.number_of_etypes()
    list_u = [None] * num_ntypes
    list_v = [None] * num_ntypes
    list_e = [None] * num_etypes
    list_arg_u_nd = [None] * num_ntypes
    list_arg_u = [None] * num_ntypes
    list_arg_u_ntype_nd = [None] * num_ntypes
    list_arg_u_ntype = [None] * num_ntypes
    # TODO(Israt): double check ntype or etype
    list_arg_e_nd = [None] * num_ntypes
    list_arg_e = [None] * num_ntypes
    list_arg_e_etype_nd = [None] * num_ntypes
    list_arg_e_etype = [None] * num_ntypes

    use_cmp = reduce_op in ["max", "min"]
    idtype = getattr(F, gidx.dtype)

    for etid in range(num_etypes):
        src_id, dst_id = gidx.metagraph.find_edge(etid)
        u = u_tuple[src_id] if use_u else None
        e = e_tuple[etid] if use_e else None
        if use_u:
            if u is not None and F.ndim(u) == 1:
                u = F.unsqueeze(u, -1)
                expand_u = True
            list_u[src_id] = u if use_u else None
        if use_e:
            if e is not None and F.ndim(e) == 1:
                e = F.unsqueeze(e, -1)
                expand_e = True
            list_e[etid] = e if use_e else None
        ctx = (
            F.context(u) if use_u else F.context(e)
        )  # TODO(Israt): Put outside of loop
        dtype = (
            F.dtype(u) if use_u else F.dtype(e)
        )  # TODO(Israt): Put outside of loop
        u_shp = F.shape(u) if use_u else (0,)
        e_shp = F.shape(e) if use_e else (0,)
        v_shp = (gidx.num_nodes(dst_id),) + infer_broadcast_shape(
            op, u_shp[1:], e_shp[1:]
        )
        list_v[dst_id] = F.zeros(v_shp, dtype, ctx)
        if use_cmp:
            if use_u:
                list_arg_u[dst_id] = F.zeros(v_shp, idtype, ctx)
                list_arg_u_ntype[dst_id] = F.zeros(v_shp, idtype, ctx)
            if use_e:
                list_arg_e[dst_id] = F.zeros(v_shp, idtype, ctx)
                list_arg_e_etype[dst_id] = F.zeros(v_shp, idtype, ctx)
        list_arg_u_nd[dst_id] = to_dgl_nd_for_write(list_arg_u[dst_id])
        list_arg_u_ntype_nd[dst_id] = to_dgl_nd_for_write(
            list_arg_u_ntype[dst_id]
        )
        list_arg_e_nd[dst_id] = to_dgl_nd_for_write(list_arg_e[dst_id])
        list_arg_e_etype_nd[dst_id] = to_dgl_nd_for_write(
            list_arg_e_etype[dst_id]
        )

    if gidx.num_edges(0) > 0:
        _CAPI_DGLKernelSpMMHetero(
            gidx,
            op,
            reduce_op,
            [to_dgl_nd(u_i) for u_i in list_u],
            [to_dgl_nd(e_i) for e_i in list_e],
            [to_dgl_nd_for_write(v_i) for v_i in list_v],
            list_arg_u_nd,
            list_arg_e_nd,
            list_arg_u_ntype_nd,
            list_arg_e_etype_nd,
        )
    for l, arg_u_nd in enumerate(list_arg_u_nd):
        # TODO(Israt): l or src_id as index of lhs
        list_arg_u[l] = (
            None
            if list_arg_u[l] is None
            else F.zerocopy_from_dgl_ndarray(arg_u_nd)
        )
        if list_arg_u[l] is not None and expand_u and use_cmp:
            list_arg_u[l] = F.squeeze(list_arg_u[l], -1)
    for l, arg_e_nd in enumerate(list_arg_e_nd):
        list_arg_e[l] = (
            None
            if list_arg_e[l] is None
            else F.zerocopy_from_dgl_ndarray(arg_e_nd)
        )
        if list_arg_e[l] is not None and expand_e and use_cmp:
            list_arg_e[l] = F.squeeze(list_arg_e[l], -1)
    for l, arg_u_ntype_nd in enumerate(list_arg_u_ntype_nd):
        list_arg_u_ntype[l] = (
            None
            if arg_u_ntype_nd is None
            else F.zerocopy_from_dgl_ndarray(arg_u_ntype_nd)
        )
    for l, arg_e_etype_nd in enumerate(list_arg_e_etype_nd):
        list_arg_e_etype[l] = (
            None
            if arg_e_etype_nd is None
            else F.zerocopy_from_dgl_ndarray(arg_e_etype_nd)
        )
    # To deal with scalar node/edge features.
    for l in range(num_ntypes):
        # replace None by empty tensor. Forward func doesn't accept None in tuple.
        v = list_v[l]
        v = F.tensor([]) if v is None else v
        if (expand_u or not use_u) and (expand_e or not use_e):
            v = F.squeeze(v, -1)  # To deal with scalar node/edge features.
        list_v[l] = v
    out = tuple(list_v)
    return out, (list_arg_u, list_arg_e, list_arg_u_ntype, list_arg_e_etype)


def _segment_mm(A, B, out, seglen_A, b_trans=False):
    """Invoke the C API of segment_mm."""
    _CAPI_DGLKernelSEGMENTMM(
        to_dgl_nd(A),
        to_dgl_nd(B),
        to_dgl_nd_for_write(out),
        to_dgl_nd(seglen_A),
        False,
        b_trans,
    )
    return out


def _segment_mm_backward_B(A, dC, dB, seglen):
    """Invoke the C API of the backward of segment_mm on B."""
    _CAPI_DGLKernelSEGMENTMMBackwardB(
        to_dgl_nd(A), to_dgl_nd(dC), to_dgl_nd_for_write(dB), to_dgl_nd(seglen)
    )
    return dB


def _gather_mm(A, B, out, idx_a=None, idx_b=None):
    r"""Invoke the C API of the gather_mm operator."""
    _CAPI_DGLKernelGATHERMM(
        to_dgl_nd(A),
        to_dgl_nd(B),
        to_dgl_nd_for_write(out),
        to_dgl_nd(idx_a),
        to_dgl_nd(idx_b),
    )
    return out


def _gather_mm_scatter(A, B, out, idx_a=None, idx_b=None, idx_c=None):
    r"""Invoke the C API of the gather_mm_scatter operator."""
    _CAPI_DGLKernelGATHERMMSCATTER(
        to_dgl_nd(A),
        to_dgl_nd(B),
        to_dgl_nd_for_write(out),
        to_dgl_nd(idx_a),
        to_dgl_nd(idx_b),
        to_dgl_nd(idx_c),
    )
    return out


def _gsddmm(gidx, op, lhs, rhs, lhs_target="u", rhs_target="v"):
    r"""Generalized Sampled-Dense-Dense Matrix Multiplication interface. It
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
    This function does not handle gradients.
    """
    if gidx.number_of_etypes() != 1:
        raise DGLError("We only support gsddmm on graph with one edge type")
    use_lhs = op != "copy_rhs"
    use_rhs = op != "copy_lhs"
    if use_lhs and use_rhs:
        if F.dtype(lhs) != F.dtype(rhs):
            raise DGLError(
                "The operands data type don't match: {} and {}, please convert them"
                " to the same type.".format(F.dtype(lhs), F.dtype(rhs))
            )
    # deal with scalar features.
    expand_lhs, expand_rhs = False, False
    if use_lhs:
        if F.ndim(lhs) == 1:
            lhs = F.unsqueeze(lhs, -1)
            expand_lhs = True
    if use_rhs:
        if F.ndim(rhs) == 1:
            rhs = F.unsqueeze(rhs, -1)
            expand_rhs = True
    lhs_target = target_mapping[lhs_target]
    rhs_target = target_mapping[rhs_target]

    ctx = F.context(lhs) if use_lhs else F.context(rhs)
    dtype = F.dtype(lhs) if use_lhs else F.dtype(rhs)
    lhs_shp = F.shape(lhs) if use_lhs else (0,)
    rhs_shp = F.shape(rhs) if use_rhs else (0,)
    out_shp = (gidx.num_edges(0),) + infer_broadcast_shape(
        op, lhs_shp[1:], rhs_shp[1:]
    )
    out = F.empty(out_shp, dtype, ctx)
    if gidx.num_edges(0) > 0:
        _CAPI_DGLKernelSDDMM(
            gidx,
            op,
            to_dgl_nd(lhs if use_lhs else None),
            to_dgl_nd(rhs if use_rhs else None),
            to_dgl_nd_for_write(out),
            lhs_target,
            rhs_target,
        )
    if (expand_lhs or not use_lhs) and (expand_rhs or not use_rhs):
        out = F.squeeze(out, -1)
    return out


def _gsddmm_hetero(
    gidx, op, lhs_len, lhs_target="u", rhs_target="v", lhs_and_rhs_tuple=None
):
    r"""Generalized Sampled-Dense-Dense Matrix Multiplication interface."""
    lhs_tuple, rhs_tuple = (
        lhs_and_rhs_tuple[:lhs_len],
        lhs_and_rhs_tuple[lhs_len:],
    )

    use_lhs = op != "copy_rhs"
    use_rhs = op != "copy_lhs"

    # TODO (Israt): Add check - F.dtype(u) != F.dtype(e):
    # deal with scalar features.
    expand_lhs, expand_rhs = False, False
    num_ntype = gidx.number_of_ntypes()
    num_etype = gidx.number_of_etypes()
    lhs_list = (
        [None] * num_ntype if lhs_target in ["u", "v"] else [None] * num_etype
    )
    rhs_list = (
        [None] * num_ntype if rhs_target in ["u", "v"] else [None] * num_etype
    )
    out_list = [None] * gidx.number_of_etypes()

    lhs_target = target_mapping[lhs_target]
    rhs_target = target_mapping[rhs_target]

    for etid in range(gidx.number_of_etypes()):
        lhs_id = get_typeid_by_target(gidx, etid, lhs_target)
        rhs_id = get_typeid_by_target(gidx, etid, rhs_target)
        lhs = lhs_tuple[lhs_id]
        rhs = rhs_tuple[rhs_id]
        if use_lhs:
            if lhs is not None and F.ndim(lhs) == 1:
                lhs = F.unsqueeze(lhs, -1)
                expand_lhs = True
        if use_rhs:
            if rhs is not None and F.ndim(rhs) == 1:
                rhs = F.unsqueeze(rhs, -1)
                expand_rhs = True
        ctx = F.context(lhs) if use_lhs else F.context(rhs)
        dtype = F.dtype(lhs) if use_lhs else F.dtype(rhs)
        lhs_shp = F.shape(lhs) if use_lhs else (0,)
        rhs_shp = F.shape(rhs) if use_rhs else (0,)
        lhs_list[lhs_id] = lhs if use_lhs else None
        rhs_list[rhs_id] = rhs if use_rhs else None
        out_shp = (gidx.num_edges(etid),) + infer_broadcast_shape(
            op, lhs_shp[1:], rhs_shp[1:]
        )
        out_list[etid] = F.empty(out_shp, dtype, ctx)
    if gidx.num_edges(0) > 0:
        _CAPI_DGLKernelSDDMMHetero(
            gidx,
            op,
            [to_dgl_nd(lhs) for lhs in lhs_list],
            [to_dgl_nd(rhs) for rhs in rhs_list],
            [to_dgl_nd_for_write(out) for out in out_list],
            lhs_target,
            rhs_target,
        )

    for l in range(gidx.number_of_etypes()):
        # Replace None by empty tensor. Forward func doesn't accept None in tuple.
        e = out_list[l]
        e = F.tensor([]) if e is None else e
        if (expand_lhs or not use_lhs) and (expand_rhs or not use_rhs):
            e = F.squeeze(e, -1)
        out_list[l] = e
    out = tuple(out_list)
    return out


def _segment_reduce(op, feat, offsets):
    r"""Segment reduction operator.

    It aggregates the value tensor along the first dimension by segments.
    The argument ``offsets`` specifies the start offset of each segment (and
    the upper bound of the last segment). Zero-length segments are allowed.

    .. math::
      y_i = \Phi_{j=\mathrm{offsets}_i}^{\mathrm{offsets}_{i+1}-1} x_j

    where :math:`\Phi` is the reduce operator.

    Parameters
    ----------
    op : str
        Aggregation method. Can be ``sum``, ``max``, ``min``.
    x : Tensor
        Value to aggregate.
    offsets : Tensor
        The start offsets of segments.

    Returns
    -------
    tuple(Tensor)
        The first tensor correspond to aggregated tensor of shape
        ``(len(seglen), value.shape[1:])``, and the second tensor records
        the argmin/max at each position for computing gradients.

    Notes
    -----
    This function does not handle gradients.
    """
    n = F.shape(offsets)[0] - 1
    out_shp = (n,) + F.shape(feat)[1:]
    ctx = F.context(feat)
    dtype = F.dtype(feat)
    idtype = F.dtype(offsets)
    out = F.zeros(out_shp, dtype, ctx)
    arg = None
    if op in ["min", "max"]:
        arg = F.zeros(out_shp, idtype, ctx)
    arg_nd = to_dgl_nd_for_write(arg)
    _CAPI_DGLKernelSegmentReduce(
        op,
        to_dgl_nd(feat),
        to_dgl_nd(offsets),
        to_dgl_nd_for_write(out),
        arg_nd,
    )
    arg = None if arg is None else F.zerocopy_from_dgl_ndarray(arg_nd)
    return out, arg


def _scatter_add(x, idx, m):
    r"""Scatter add operator (on first dimension) implementation.

    Math: y[idx[i], *] += x[i, *]

    Parameters
    ----------
    x : Tensor
        The input feature.
    idx : Tensor
        The indices array.
    m : int
        The length of output.

    Returns
    -------
    Tensor
        The output tensor.
    """
    out_shp = (m,) + F.shape(x)[1:]
    ctx = F.context(x)
    dtype = F.dtype(x)
    out = F.zeros(out_shp, dtype, ctx)
    _CAPI_DGLKernelScatterAdd(
        to_dgl_nd(x), to_dgl_nd(idx), to_dgl_nd_for_write(out)
    )
    return out


def _update_grad_minmax_hetero(
    gidx, op, list_x, list_idx, list_idx_etype, list_dX
):
    r"""Update gradients for reduce operator max and min (on first dimension) implementation.

    Parameters
    ----------
    gidx : HeteroGraphIndex
        The input graph index.
    list_x : List of tensors
        List of the input features.
    list_idx : List of tensors
        List of the indices array.
    list_idx_etype : List of tensors
        List of the node- or edge-type array.
    list_dX : List of tensors
        List of gradients.

    Returns
    -------
    Tensor
        The output tensor.
    """
    use_u = op != "copy_rhs"
    use_e = op != "copy_lhs"
    list_out = [None] * len(list_dX)
    for etid in range(gidx.number_of_etypes()):
        src_id, dst_id = gidx.metagraph.find_edge(etid)  # gidx is reveresed
        x = list_x[src_id]
        ctx = F.context(x)
        dtype = F.dtype(x)
        if use_u:
            out_shp = (len(list_dX[dst_id]),) + F.shape(x)[1:]
            list_out[dst_id] = F.zeros(out_shp, dtype, ctx)
        if use_e:
            out_shp = (len(list_dX[etid]),) + F.shape(x)[1:]
            list_out[etid] = F.zeros(out_shp, dtype, ctx)

    _CAPI_DGLKernelUpdateGradMinMaxHetero(
        gidx,
        op,
        [to_dgl_nd(x) for x in list_x],
        [to_dgl_nd(idx) for idx in list_idx],
        [to_dgl_nd(idx_etype) for idx_etype in list_idx_etype],
        [to_dgl_nd_for_write(out) for out in list_out],
    )
    return tuple(list_out)


def _bwd_segment_cmp(feat, arg, m):
    r"""Backward phase of segment reduction (for 'min'/'max' reduction).

    It computes the gradient of input feature given output gradient of
    the segment reduction result.

    Parameters
    ----------
    feat : Tensor
        The output gradient
    arg : Tensor
        The ArgMin/Max tensor produced by segment_reduce op.
    m : int
        The length of input gradients' first dimension.

    Returns
    -------
    Tensor
        The input gradient.
    """
    out_shp = (m,) + F.shape(feat)[1:]
    ctx = F.context(feat)
    dtype = F.dtype(feat)
    out = F.zeros(out_shp, dtype, ctx)
    _CAPI_DGLKernelBwdSegmentCmp(
        to_dgl_nd(feat), to_dgl_nd(arg), to_dgl_nd_for_write(out)
    )
    return out


def _csrmm(A, A_weights, B, B_weights, num_vtypes):
    """Return a graph whose adjacency matrix is the sparse matrix multiplication
    of those of two given graphs.

    Note that the edge weights of both graphs must be scalar, i.e. :attr:`A_weights`
    and :attr:`B_weights` must be 1D vectors.

    Parameters
    ----------
    A : HeteroGraphIndex
        The input graph index as left operand.
    A_weights : Tensor
        The edge weights of graph A as 1D tensor.
    B : HeteroGraphIndex
        The input graph index as right operand.
    B_weights : Tensor
        The edge weights of graph B as 1D tensor.
    num_vtypes : int
        The number of node types for the returned graph (must be either 1 or 2).

    Returns
    -------
    C : HeteroGraphIndex
        The output graph index.
    C_weights : Tensor
        The edge weights of the output graph.
    """
    C, C_weights = _CAPI_DGLCSRMM(
        A, F.to_dgl_nd(A_weights), B, F.to_dgl_nd(B_weights), num_vtypes
    )
    return C, F.from_dgl_nd(C_weights)


def _csrsum(As, A_weights):
    """Return a graph whose adjacency matrix is the sparse matrix summation
    of the given list of graphs.

    Note that the edge weights of all graphs must be scalar, i.e. the arrays in
    :attr:`A_weights` must be 1D vectors.

    Parameters
    ----------
    As : list[HeteroGraphIndex]
        The input graph indices.
    A_weights : list[Tensor]
        The edge weights of graph A as 1D tensor.

    Returns
    -------
    C : HeteroGraphIndex
        The output graph index.
    C_weights : Tensor
        The edge weights of the output graph.
    """
    C, C_weights = _CAPI_DGLCSRSum(As, [F.to_dgl_nd(w) for w in A_weights])
    return C, F.from_dgl_nd(C_weights)


def _csrmask(A, A_weights, B):
    """Return the weights of A at the locations identical to the sparsity pattern
    of B.

    If a non-zero entry in B does not exist in A, DGL returns 0 for that location
    instead.

    Note that the edge weights of the graph must be scalar, i.e. :attr:`A_weights`
    must be a 1D vector.

    In scipy notation this is identical to ``A[B != 0]``.

    Parameters
    ----------
    A : HeteroGraphIndex
        The input graph index as left operand.
    A_weights : Tensor
        The edge weights of graph A as 1D tensor.
    B : HeteroGraphIndex
        The input graph index as right operand.

    Returns
    -------
    B_weights : Tensor
        The output weights.
    """
    return F.from_dgl_nd(_CAPI_DGLCSRMask(A, F.to_dgl_nd(A_weights), B))


###################################################################################################
## Libra Graph Partition
def libra_vertex_cut(
    nc,
    node_degree,
    edgenum_unassigned,
    community_weights,
    u,
    v,
    w,
    out,
    N,
    N_e,
    dataset,
):
    """
    This function invokes C/C++ code for Libra based graph partitioning.
    Parameter details are present in dgl/src/array/libra_partition.cc
    """
    _CAPI_DGLLibraVertexCut(
        nc,
        to_dgl_nd_for_write(node_degree),
        to_dgl_nd_for_write(edgenum_unassigned),
        to_dgl_nd_for_write(community_weights),
        to_dgl_nd(u),
        to_dgl_nd(v),
        to_dgl_nd(w),
        to_dgl_nd_for_write(out),
        N,
        N_e,
        dataset,
    )


def libra2dgl_build_dict(
    a,
    b,
    indices,
    ldt_key,
    gdt_key,
    gdt_value,
    node_map,
    offset,
    nc,
    c,
    fsize,
    dataset,
):
    """
    This function invokes C/C++ code for pre-processing Libra output.
    After graph partitioning using Libra, during conversion from Libra output to DGL/DistGNN input,
    this function creates dictionaries to assign local node ids to the partitioned nodes
    and also to create a database of the split nodes.
    Parameter details are present in dgl/src/array/libra_partition.cc
    """
    ret = _CAPI_DGLLibra2dglBuildDict(
        to_dgl_nd_for_write(a),
        to_dgl_nd_for_write(b),
        to_dgl_nd_for_write(indices),
        to_dgl_nd_for_write(ldt_key),
        to_dgl_nd_for_write(gdt_key),
        to_dgl_nd_for_write(gdt_value),
        to_dgl_nd_for_write(node_map),
        to_dgl_nd_for_write(offset),
        nc,
        c,
        fsize,
        dataset,
    )
    return ret


def libra2dgl_build_adjlist(
    feat,
    gfeat,
    adj,
    inner_node,
    ldt,
    gdt_key,
    gdt_value,
    node_map,
    lr,
    lrtensor,
    num_nodes,
    nc,
    c,
    feat_size,
    labels,
    trainm,
    testm,
    valm,
    glabels,
    gtrainm,
    gtestm,
    gvalm,
    feat_shape,
):
    """
    This function invokes C/C++ code for pre-processing Libra output.
    After graph partitioning using Libra, once the local and global dictionaries are built,
    for each node in each partition, this function copies the split node details from the
    global dictionary. It also copies features, label, train, test, and validation information
    for each node from the input graph to the corresponding partitions.
    Parameter details are present in dgl/src/array/libra_partition.cc
    """
    _CAPI_DGLLibra2dglBuildAdjlist(
        to_dgl_nd(feat),
        to_dgl_nd_for_write(gfeat),
        to_dgl_nd_for_write(adj),
        to_dgl_nd_for_write(inner_node),
        to_dgl_nd(ldt),
        to_dgl_nd(gdt_key),
        to_dgl_nd(gdt_value),
        to_dgl_nd(node_map),
        to_dgl_nd_for_write(lr),
        to_dgl_nd(lrtensor),
        num_nodes,
        nc,
        c,
        feat_size,
        to_dgl_nd(labels),
        to_dgl_nd(trainm),
        to_dgl_nd(testm),
        to_dgl_nd(valm),
        to_dgl_nd_for_write(glabels),
        to_dgl_nd_for_write(gtrainm),
        to_dgl_nd_for_write(gtestm),
        to_dgl_nd_for_write(gvalm),
        feat_shape,
    )


def libra2dgl_set_lr(gdt_key, gdt_value, lrtensor, nc, Nn):
    """
    This function invokes C/C++ code for pre-processing Libra output.
    To prepare the graph partitions for DistGNN input, this function sets the leaf
    and root (1-level tree) among the split copies (across different partitions)
    of a node from input graph.
    Parameter details are present in dgl/src/array/libra_partition.cc
    """
    _CAPI_DGLLibra2dglSetLR(
        to_dgl_nd(gdt_key),
        to_dgl_nd(gdt_value),
        to_dgl_nd_for_write(lrtensor),
        nc,
        Nn,
    )


_init_api("dgl.sparse", __name__)
