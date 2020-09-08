"""Module for sparse matrix operators."""
# pylint: disable= invalid-name
from __future__ import absolute_import

import dgl.ndarray as nd
from ._ffi.function import _init_api
from .base import DGLError
from . import backend as F
import tvm
from .tvm import gsddmm, gspmm
from .function import TargetCode

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

def to_dgl_nd(x):
    """Convert framework-specific tensor/None to dgl ndarray."""
    return nd.NULL['int64'] if x is None else F.zerocopy_to_dgl_ndarray(x)


def to_dgl_nd_for_write(x):
    """Convert framework-specific tensor/None to dgl ndarray for write."""
    return nd.NULL['int64'] if x is None else F.zerocopy_to_dgl_ndarray_for_write(x)

target_mapping = {
    'u': TargetCode.SRC,
    'e': TargetCode.EDGE,
    'v': TargetCode.DST,
    'src': TargetCode.SRC,
    'edge': TargetCode.EDGE,
    'dst': TargetCode.DST
}

use_tvm = True

def _gspmm(gidx, op, reduce_op, u, e):
    return _gspmm_tvm(gidx, op, reduce_op, u, e) if use_tvm else _gspmm_native(gidx, op, reduce_op, u, e)

def _gsddmm(gidx, op, lhs, rhs, lhs_target='u', rhs_target='v'):
    return _gsddmm_tvm(gidx, op, lhs, rhs, lhs_target, rhs_target) if use_tvm else _gsddmm_native(gidx, op, lhs, rhs, lhs_target, rhs_target)

def _gspmm_native(gidx, op, reduce_op, u, e):
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
    This function does not handle gradients.
    """
    if gidx.number_of_etypes() != 1:
        raise DGLError("We only support gspmm on graph with one edge type")
    use_u = op != 'copy_rhs'
    use_e = op != 'copy_lhs'
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
    v_shp = (gidx.number_of_nodes(dsttype), ) +\
        infer_broadcast_shape(op, u_shp[1:], e_shp[1:])
    v = F.zeros(v_shp, dtype, ctx)
    use_cmp = reduce_op in ['max', 'min']
    arg_u, arg_e = None, None
    idtype = getattr(F, gidx.dtype)
    if use_cmp:
        if use_u:
            arg_u = F.zeros(v_shp, idtype, ctx)
        if use_e:
            arg_e = F.zeros(v_shp, idtype, ctx)
    arg_u_nd = to_dgl_nd_for_write(arg_u)
    arg_e_nd = to_dgl_nd_for_write(arg_e)
    if gidx.number_of_edges(0) > 0:
        _CAPI_DGLKernelSpMM(gidx, op, reduce_op,
                            to_dgl_nd(u if use_u else None),
                            to_dgl_nd(e if use_e else None),
                            to_dgl_nd_for_write(v),
                            arg_u_nd,
                            arg_e_nd)
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
    return v, (arg_u, arg_e)


def _gsddmm_native(gidx, op, lhs, rhs, lhs_target='u', rhs_target='v'):
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
    This function does not handle gradients.
    """
    if gidx.number_of_etypes() != 1:
        raise DGLError("We only support gsddmm on graph with one edge type")
    use_lhs = op != 'copy_rhs'
    use_rhs = op != 'copy_lhs'
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
    out_shp = (gidx.number_of_edges(0), ) +\
        infer_broadcast_shape(op, lhs_shp[1:], rhs_shp[1:])
    out = F.zeros(out_shp, dtype, ctx)
    if gidx.number_of_edges(0) > 0:
        _CAPI_DGLKernelSDDMM(gidx, op,
                             to_dgl_nd(lhs if use_lhs else None),
                             to_dgl_nd(rhs if use_rhs else None),
                             to_dgl_nd_for_write(out),
                             lhs_target, rhs_target)
    if (expand_lhs or not use_lhs) and (expand_rhs or not use_rhs):
        out = F.squeeze(out, -1)
    return out

compiled_gspmm_kernels = {}
compiled_gsddmm_kernels = {}

partitioned_1d_graphs = {}
partitioned_2d_graphs = {}

def _gspmm_tvm(gidx, op, reduce_op, u, e, advise=True,
           num_feat_partitions=1, num_col_partitions=1):
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
    advise: Boolean
        If True, dgl will search for optimal parameters for partitions to speedup.
        default is True
        It will be overrided if any number of partitions(below) is set to larger than 1.
    num_feat_partitions: int
        Number of partitions on feature dimension. 
        It must be positive, and is smaller than feature length of result.
        If feature has multiple dimensions, this parameter is applied
        on flatten features of result.
    num_col_partitions: int
        Number of partitions on the columns of the sparse matrix. 
        It must be positive, and is smaller than number of columns.
        If this value is larger than 1, a partitioning algorithm is run on the graph. 
        The partitioned results is then cached for following runs. 
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
        raise DGLError("We only support gsddmm on graph with one edge type")
    nnz = gidx.number_of_edges(0)
    if nnz <= 0:
        return None
    use_u = op != 'copy_rhs'
    use_e = op != 'copy_lhs'
    if use_u and F.ndim(u) == 1:
        u = F.unsqueeze(u, -1)
    if use_e and F.ndim(e) == 1:
        e = F.unsqueeze(e, -1)
    ctx = F.context(u) if use_u else F.context(e)
    feat_type = F.dtype(u) if use_u else F.dtype(e)
    u_shp = F.shape(u) if use_u else (0,)
    e_shp = F.shape(e) if use_e else (0,)
    indice_type = gidx.dtype
    srctype, dsttype = gidx.metagraph.find_edge(0)
    num_rows = gidx.number_of_nodes(dsttype)
    num_cols = gidx.number_of_nodes(srctype)
    target = F.device_type(ctx)
    if target == 'cpu' and num_feat_partitions == 1 and num_col_partitions == 1 and advise:
        # search for parameter
        pass
        # num_col_partitions = 8
        # num_feat_partitions = 2
    if target == 'cuda' and (num_col_partitions > 1 or num_feat_partitions > 1):
        print('Partitioning not supported on GPU')
        num_col_partitions, num_feat_partitions = 1, 1
    if num_col_partitions == 1:
        indptr, indices, edge_mapping = map(lambda x: tvm.nd.from_dlpack(x.to_dlpack()), gidx.get_csc_dlpack(0))
    else:
        graph_key = (id(gidx), num_col_partitions, num_rows, num_cols, nnz)
        if graph_key in partitioned_1d_graphs:
            dds = partitioned_1d_graphs[graph_key]
        else:
            # perform partition and cache the result
            tmp = _CAPI_DGLPartition1D(gidx, 0, num_col_partitions)
            dds = [tvm.nd.from_dlpack(tmp(x).to_dlpack()) for x in range(3)]
            partitioned_1d_graphs[graph_key] = dds
        indptr, indices, edge_mapping = dds
    edge_shuffled = edge_mapping.shape != (0,)
    use_idx = edge_shuffled and use_e
    f_input = [indptr, indices]
    key = (num_rows, num_cols, nnz, op, reduce_op, u_shp, e_shp, use_idx, \
           num_feat_partitions, num_col_partitions, indice_type, feat_type, target)
    if key not in compiled_gspmm_kernels:
        if target == 'cpu':
            target = 'llvm'
        v_shp = (num_rows, ) +\
            infer_broadcast_shape(op, u_shp[1:], e_shp[1:])
        mod = gspmm.spmm(
            op, reduce_op, nnz, num_rows, num_cols,
            u_shp[1:], e_shp[1:], v_shp[1:], indice_type, str(feat_type),
            use_idx=use_idx, target=target,
            num_feat_partitions=num_feat_partitions,
            num_col_partitions=num_col_partitions
        )
        compiled = (mod, v_shp)
        compiled_gspmm_kernels[key] = compiled
    else:
        compiled = compiled_gspmm_kernels[key]
    mod, v_shp = compiled
    if use_idx:
        f_input.append(tvm.nd.from_dlpack(edge_mapping.to_dlpack()))
    if use_u:
        f_input.append(tvm.nd.from_dlpack(to_dgl_nd(u).to_dlpack()))
    if use_e:
        f_input.append(tvm.nd.from_dlpack(to_dgl_nd(e).to_dlpack()))
    idtype = getattr(F, gidx.dtype)
    arg_u, arg_e = None, None
    use_cmp = reduce_op != 'sum'
    if use_cmp:
        if use_e:
            arg_e = F.zeros(v_shp, idtype, ctx)
            f_input.append(tvm.nd.from_dlpack(to_dgl_nd_for_write(arg_e).to_dlpack()))
        if use_u:
            arg_u = F.zeros(v_shp, idtype, ctx)
            f_input.append(tvm.nd.from_dlpack(to_dgl_nd_for_write(arg_u).to_dlpack()))
    v = F.zeros(v_shp, feat_type, ctx)
    f_input.append(tvm.nd.from_dlpack(to_dgl_nd_for_write(v).to_dlpack()))
    mod(*f_input)
    return v, (arg_u, arg_e)


def _gsddmm_tvm(gidx, op, lhs, rhs, lhs_target='u', rhs_target='v', advise=True,
            num_feat_partitions=1, num_row_partitions=1, num_col_partitions=1):
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
    advise: Boolean
        If True, dgl will search for optimal partition parameters.
        default is True
        It will be overrided if any number of partitions(below) is set to larger than 1.
    num_feat_partitions: int
        Number of partitions on feature dimension. 
        It must be positive, and is smaller than feature length of result.
        If feature has multiple dimensions, this parameter is applied
        on flatten features of result.
    num_row_partitions: int
        Number of partitions on the rows of the sparse matrix.
        It must be positive, and is smaller than number of rows.
        If this value and/or the next parameter is larger than 1, 
        a partitioning algorithm is run on the graph. 
        The partitioned results is then cached for following runs. 
    num_col_partitions: int
        Number of partitions on the columns of the sparse matrix.
        It must be positive, and is smaller than number of columns.

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
    nnz = gidx.number_of_edges(0)
    if nnz <= 0:
        return None
    lhs_target = target_mapping[lhs_target]
    rhs_target = target_mapping[rhs_target]
    if op in ['copy_lhs', 'copy_rhs']:
        row, col, edge_mapping = gidx.get_coo_dlpack(0)
        t = lhs_target if op == 'copy_lhs' else rhs_target
        if t == 0:
            ind = F.zerocopy_from_dgl_ndarray(row).long()
        elif t == 1:
            return lhs if op == 'copy_lhs' else rhs
        else:
            ind = F.zerocopy_from_dgl_ndarray(col).long()
        return lhs[ind] if op == 'copy_lhs' else rhs[ind]
    ctx = F.context(lhs)
    target = F.device_type(ctx)
    feat_type = F.dtype(lhs)
    if F.ndim(lhs) == 1:
        lhs = F.unsqueeze(lhs, -1)
    if F.ndim(rhs) == 1:
        rhs = F.unsqueeze(rhs, -1)
    lhs_shp = F.shape(lhs)
    rhs_shp = F.shape(rhs)
    indice_type = gidx.dtype
    srctype, dsttype = gidx.metagraph.find_edge(0)
    num_cols = gidx.number_of_nodes(dsttype)
    num_rows = gidx.number_of_nodes(srctype)
    if target == 'cpu' and num_row_partitions == 1 and num_col_partitions == 1 and num_feat_partitions == 1 and advise:
        # search for parameters
        pass
        # num_row_partitions, num_col_partitions = 2, 2
        # num_feat_partitions = 2
    if target == 'cuda' and (num_row_partitions > 1 or num_col_partitions > 1 or num_feat_partitions > 1):
        print('Partitioning not supported on GPU')
        num_row_partitions, num_col_partitions, num_feat_partitions = 1, 1, 1
    graph_key = (id(gidx), num_row_partitions, num_col_partitions, num_rows, num_cols, nnz)
    if graph_key in partitioned_2d_graphs:
        row, col, edge_mapping, reverse_mapping = partitioned_2d_graphs[graph_key]
    else:
        if num_row_partitions == 1 and num_col_partitions == 1:
            row, col, edge_mapping = map(lambda x: tvm.nd.from_dlpack(x.to_dlpack()), gidx.get_coo_dlpack(0))
        else:
            tmp = _CAPI_DGLPartition2D(gidx, 0, num_row_partitions, num_col_partitions)
            row, col, edge_mapping = [tvm.nd.from_dlpack(tmp(x).to_dlpack()) for x in range(3)]
        reverse_mapping = F.zerocopy_to_dgl_ndarray(F.argsort(F.from_dgl_nd(edge_mapping), 0, False))
        partitioned_2d_graphs[graph_key] = (row, col, edge_mapping, reverse_mapping)
    edge_shuffled = edge_mapping.shape != (0,)
    use_idx = (lhs_target == TargetCode.EDGE or rhs_target == TargetCode.EDGE) \
              and edge_shuffled
    f_input = []
    if lhs_target == TargetCode.SRC or rhs_target == TargetCode.SRC:
        f_input.append(row)
    if lhs_target == TargetCode.DST or rhs_target == TargetCode.DST:
        f_input.append(col)
    if use_idx:
        f_input.append(edge_mapping)
    key = (num_rows, num_cols, nnz, op, lhs_target, rhs_target, num_feat_partitions,\
           use_idx, lhs_shp, rhs_shp, indice_type, feat_type, target)
    if key not in compiled_gsddmm_kernels:
        if target == 'cpu':
            target = 'llvm'
        out_shp = (gidx.number_of_edges(0), ) +\
            infer_broadcast_shape(op, lhs_shp[1:], rhs_shp[1:])
        mod = gsddmm.sddmm(
            op, nnz, num_rows, num_cols,
            lhs_shp[1:], rhs_shp[1:], out_shp[1:], str(indice_type), str(feat_type),
            lhs_target=lhs_target, rhs_target=rhs_target, target=target,
            num_feat_partitions=num_feat_partitions, is_sorted=2, use_idx=use_idx
        )
        compiled = (mod, out_shp)
        compiled_gsddmm_kernels[key] = compiled
    else:
        compiled = compiled_gsddmm_kernels[key]
    mod, out_shp = compiled
    f_input.append(tvm.nd.from_dlpack(to_dgl_nd(lhs).to_dlpack()))
    f_input.append(tvm.nd.from_dlpack(to_dgl_nd(rhs).to_dlpack()))
    out = F.zeros(out_shp, feat_type, ctx)
    f_input.append(tvm.nd.from_dlpack(to_dgl_nd_for_write(out).to_dlpack()))
    mod(*f_input)
    if edge_shuffled:
        out = F.gather_row(out, F.from_dgl_nd(reverse_mapping))
    return out

_init_api("dgl.sparse")
