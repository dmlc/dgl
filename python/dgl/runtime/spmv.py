"""Module for SPMV rules."""
from __future__ import absolute_import

from ..base import DGLError
from .. import backend as F
from .. import utils
from .. import ndarray

from . import ir
from .ir import var

import scipy.sparse as sp
from functools import partial

def analyze_v2v_spmv(graph, mfunc, rfunc):
    """Analyze if SPMV from node space to node space can be applied.

    Parameters
    ----------
    graph: DGLGraph
        DGLGraph to use
    mfunc : list of dgl.function.BuiltinFunction
        The message function list.
    rfunc : list of dgl.function.BuiltinFunction
        The reduce function list.

    Returns
    -------
    spmv_pairs : list of pair of builtin functions
        The pair of spvm applicable message/reduce functions.
    """
    spmv_pairs = []
    fld2mfunc = {fn.out_field: fn for fn in mfunc}

    for rfn in rfunc:
        mfld = rfn.msg_field
        if mfld not in fld2mfunc:
            raise DGLError('Reduce function requires message field "%s",'
                           ' but no message function generates it.' % mfld)
        mfn = fld2mfunc[mfld]
        spmv_pairs.append((mfn, rfn))

    return spmv_pairs

def gen_v2v_spmv_schedule(adj, spmv_pairs, src_frame, dst_frame, edge_frame,
                          out, out_size, edge_mapping, out_mapping):
    """Generate v2v spmv schedule.

    Parameters
    ----------
    adj : tuple (sparse matrix, utils.Index)
    spmv_pairs : list of pair
    src_frame : var.Var
        input source node features
    dst_frame : var.Var
        input destination node features
    edge_frame : var.Var
        input edge features
    out : var.Var
        output node features
    out_mapping : var.Var
        mapping from recv nodes to relabeled consecutive node ids
    """
    spmat = var.SPMAT(adj)
    for mfn, rfn in spmv_pairs:
        ftdst = mfn(spmat, src_frame, dst_frame, edge_frame, reducer=rfn.name,
                    edge_mapping=edge_mapping, out_mapping=out_mapping)
        # save for merge
        ir.WRITE_COL_(out, var.STR(rfn.out_field), ftdst)

def gen_v2e_spmv_schedule(adj, mfunc, src_frame, dst_frame, edge_frame,
                          out_frame, out_size, edge_mapping, eid=None):
    """Generate v2e SPMV schedule"""
    spmat = var.SPMAT(adj)
    if eid is not None:
        write_back = partial(ir.WRITE_, row=eid)
    else:
        write_back = ir.WRITE_COL_
    for mfn in mfunc:
        fmsg = mfn(spmat, src_frame, dst_frame, edge_frame, out_size,
                   edge_mapping=edge_mapping)
        write_back(out_frame, var.STR(mfn.out_field), fmsg)

def gen_e2v_spmv_schedule(adj, rfunc, mfr, edge_mapping, out, out_size,
                          out_mapping):
    """Generate e2v SPMV schedule.

    Parameters
    ----------
    inc : tuple (sparse matrix, utils.Index)
    rfunc : list of builtin reducers
    mf : var.Var
        Variable for message frame.
    out : var.Var
        Variable for output reduced features.
    """
    spmat = var.SPMAT(adj)
    for rfn in rfunc:
        ftdst = rfn(spmat, mfr, edge_mapping, out_mapping, out_size)
        ir.WRITE_COL_(out, var.STR(rfn.out_field), ftdst)

def build_block_adj_matrix_graph(graph, block_id):
    """Build adjacency matrix of the whole graph.

    Parameters
    ----------
    graph : NodeFlow
        The NodeFlow

    block_id : int
        the block Id

    Returns
    -------
    utils.CtxCachedObject
        Get be used to get adjacency matrix on the provided ctx.
    utils.Index
        A index for data shuffling due to sparse format change. Return None
        if shuffle is not required.
    """
    # TODO why is this constructed twice?
    _, shuffle_idx = graph.block_adjacency_matrix(block_id, F.cpu())
    shuffle_idx = utils.toindex(shuffle_idx) if shuffle_idx is not None else None
    return lambda ctx: graph.block_adjacency_matrix(block_id, ctx)[0], shuffle_idx

def build_adj_matrix_graph(graph):
    """Build adjacency matrix of the whole graph.

    Parameters
    ----------
    graph : DGLGraph
        The graph

    Returns
    -------
    utils.CtxCachedObject
        Get be used to get adjacency matrix on the provided ctx.
    """
    gidx = graph._graph
    eid = gidx.adjacency_matrix(True)[2]
    inv_eid = gidx.adjacency_matrix(False)[2]
    return lambda ctx: (gidx.adjacency_matrix(True, ctx)[:2] +
                        gidx.adjacency_matrix(False, ctx)[:2]), eid, inv_eid

def _build_adj_matrix_index_uv(edge_tuple, num_nodes):
    """Build adj matrix index and shape using the given (u, v) edges.

    The matrix is of shape (len(reduce_nodes), n), where n is the number of nodes
    in the graph. Therefore, when doing SPMV, the src node data
    should be all the node features.

    The dst nodes will be sorted in the *unique-ascending* order of
    their ids. This is compatible with other reduce scheduler such as
    degree-bucketing scheduler.

    Paramters
    ---------
    edges : tuple of utils.Index
        (u, v)
    num_sources : int
        The number of source nodes.

    Returns
    -------
    sparse index
        The sparse index.
    tuple of int
        The dense shape.
    """
    u, v, eid = edge_tuple
    u = u.tousertensor()
    v = v.tousertensor()
    eid = eid.tousertensor()
    u = F.zerocopy_to_numpy(u)
    v = F.zerocopy_to_numpy(v)
    eid = F.zerocopy_to_numpy(eid)
    csr = sp.csr_matrix((eid, (u, v)), shape=(num_nodes, num_nodes))
    inv_csr = sp.csr_matrix((eid, (v, u)), shape=(num_nodes, num_nodes))
    spmat = (csr.indptr, csr.indices, csr.data)
    inv_spmat = (inv_csr.indptr, inv_csr.indices, inv_csr.data)
    return spmat, inv_spmat


def build_adj_matrix_uv(edge_tuple, num_nodes):
    """Build adj matrix using the given (u, v, eid) edges and target nodes.

    The matrix is of shape (len(reduce_nodes), n), where n is the number of
    nodes in the graph. Therefore, when doing SPMV, the src node data should be
    all the node features.

    Parameters
    ---------
    edge_tuple : tuple of utils.Index
        (u, v, eid)
    reduce_nodes : utils.Index
        The nodes to reduce messages, which will be target dimension
        of the adjmat. The nodes include unique(v) and zero-degree-nodes.
    num_sources : int
        The number of source nodes.

    Returns
    -------
    utils.CtxCachedObject
        Get be used to get adjacency matrix and on the provided ctx.
    utils.Index
        A index for data shuffling due to sparse format change. Return None
        if shuffle is not required.
    """
    u, v, eid = edge_tuple
    spmat, inv_spmat = _build_adj_matrix_index_uv(edge_tuple, num_nodes)

    def copy_to(ctx):
        indptr = ndarray.array(spmat[0], ctx=ctx)
        indices = ndarray.array(spmat[1], ctx=ctx)
        inv_indptr = ndarray.array(inv_spmat[0], ctx=ctx)
        inv_indices = ndarray.array(inv_spmat[1], ctx=ctx)
        return indptr, indices, inv_indptr, inv_indices

    return utils.CtxCachedObject(copy_to), spmat[2], inv_spmat[2]

def build_block_inc_matrix_graph(graph, block_id):
    """Build incidence matrix.

    Parameters
    ----------
    graph : NodeFlow
        The NodeFlow.

    block_id : int
        The block Id

    Returns
    -------
    utils.CtxCachedObject
        Get be used to get incidence matrix on the provided ctx.
    utils.Index
        A index for data shuffling due to sparse format change. Return None
        if shuffle is not required.
    """
    # inc mat will not use data tensor so conversion index is not needed
    return lambda ctx: graph.block_incidence_matrix(block_id, 'in', ctx)[0], None


'''
def build_inc_matrix_graph(graph):
    """Build incidence matrix.

    Parameters
    ----------
    graph : DGLGraph
        The graph.

    Returns
    -------
    utils.CtxCachedObject
        Get be used to get incidence matrix on the provided ctx.
    utils.Index
        A index for data shuffling due to sparse format change. Return None
        if shuffle is not required.
    """
    gidx = graph._graph
    # inc mat will not use data tensor so conversion index is not needed
    return lambda ctx: gidx.incidence_matrix('in', ctx)[0], None

def build_inc_matrix_eid(m, eid, dst, reduce_nodes):
    """Build incidence matrix using edge id and edge dst nodes.

    The incidence matrix is of shape (n, m), where n=len(reduce_nodes).
    The nnz is equal to len(eid).

    Invariant: len(eid) == len(dst)

    The dst nodes will be sorted in the *unique-ascending* order of
    their ids. This is compatible with other reduce scheduler such as
    degree-bucketing scheduler.

    Examples
    --------
    Total of seven edges. Three edges point to node 1 (eid=0,1,2);
    two point to node 3 (eid=3,4); two point to node 4 (eid=5,6).
    Only five edges should be included in the result incmat (eid=1,2,3,5,6).
    There are five nodes in the final target dimension (0~4),
    where node 0 and 2 are two 0-deg nodes.
    >>> m = 7
    >>> eid = [1, 2, 3, 5, 6]
    >>> dst = [1, 1, 3, 4, 4]
    >>> reduce_nodes = [0, 1, 2, 3, 4]
    >>> build_inc_matrix_eid(m, eid, dst, reduce_nodes)
    tensor([[0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1]], shape=(5, 7))

    Parameters
    ---------
    m : int
        The source dimension size of the incidence matrix.
    eid : utils.Index
        The edge ids. All ids must be in range [0, m).
    dst : utils.Index
        The edge destination nodes. len(eid) == len(dst).
    reduce_nodes : utils.Index
        The nodes to reduce messages, which will be target dimension
        of the incmat. The nodes include unique(dst) and zero-degree-nodes.

    Returns
    -------
    utils.CtxCachedObject
        Get be used to get incidence matrix on the provided ctx.
    utils.Index
        A index for data shuffling due to sparse format change. Return None
        if shuffle is not required.
    """
    _, old2new = utils.build_relabel_map(reduce_nodes, is_sorted=True)
    dst = dst.tousertensor()
    eid = eid.tousertensor()
    # relabel edges dsts
    new_v = old2new[dst]  # FIXME(minjie): no use []
    # create sparse index tensor
    n = len(reduce_nodes)
    row = F.unsqueeze(new_v, 0)
    col = F.unsqueeze(eid, 0)
    idx = F.cat([row, col], dim=0)
    # create dat tensor
    nnz = len(eid)
    dat = F.ones((nnz,), dtype=F.float32, ctx=F.cpu())
    mat, _ = F.sparse_matrix(dat, ('coo', idx), (n, m))
    # inc mat will not use data tensor so conversion index is not needed
    return utils.CtxCachedObject(lambda ctx: F.copy_to(mat, ctx)), None

def build_inc_matrix_dst(dst, reduce_nodes):
    """Build incidence matrix using only edge destinations.

    The incidence matrix is of shape (n, m), where n=len(reduce_nodes), m=len(dst).
    The nnz is equal to len(dst).

    Examples
    --------
    Five edges. Two edges point to node 1; one points to node 3;
    two point to node 4. There are five nodes in the final
    target dimension (0~4), where node 0 and 2 are two 0-deg nodes.
    >>> dst = [1, 1, 3, 4, 4]
    >>> reduce_nodes = [0, 1, 2, 3, 4]
    >>> build_inc_matrix_dst(dst, reduced_nodes)
    tensor([[0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1]], shape=(5, 5))

    Parameters
    ----------
    dst : utils.Index
        The edge destinations.
    reduce_nodes : utils.Index
        The nodes to reduce messages, which will be target dimension
        of the incmat. The nodes include unique(dst) and zero-degree-nodes.

    Returns
    -------
    utils.CtxCachedObject
        Get be used to get incidence matrix on the provided ctx.
    utils.Index
        A index for data shuffling due to sparse format change. Return None
        if shuffle is not required.
    """
    eid = utils.toindex(F.arange(0, len(dst)))
    return build_inc_matrix_eid(len(eid), eid, dst, reduce_nodes)

'''
