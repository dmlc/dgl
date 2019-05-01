"""Module for SPMV rules."""
from __future__ import absolute_import

from ..base import DGLError
from .. import backend as F
from .. import utils
from .. import ndarray as nd

from . import ir
from .ir import var

import scipy.sparse as sp
import numpy as np


def gen_v2v_spmv_schedule(adj, mfunc, rfunc, src_frame, dst_frame, edge_frame,
                          out, out_size, edge_map, out_map):
    """Generate v2v spmv schedule.

    Parameters
    ----------
    adj : utils.CtxCachedObject
        function that generates four dgl.ndarray (indptr, indices, inv_indptr,
        inv_indices) representing CSR and transposed-CSR matrix, copies to and
        caches on to given context
    mfunc : list of builtin message func
    rfunc : list of builtin reduce func
    src_frame : var.Var
        input source node features
    dst_frame : var.Var
        input destination node features
    edge_frame : var.Var
        input edge features
    out : var.Var
        output node features
    out_size : int
        number of output nodes
    out_map : utils.CtxCachedObject
        function that generates a mapping from destination nodes to consecutive
        ids and caches on given context
    """
    spmat = var.SPMAT(adj)
    fld2mfunc = {fn.out_field: fn for fn in mfunc}
    for rfn in rfunc:
        mfld = rfn.msg_field
        if mfld not in fld2mfunc:
            raise DGLError('Reduce function requires message field "%s",'
                           ' but no message function generates it.' % mfld)
        mfn = fld2mfunc[mfld]
        ftdst = mfn(spmat, src_frame, dst_frame, edge_frame, out_size,
                    reducer=rfn.name, edge_map=edge_map, out_map=out_map)
        ir.WRITE_COL_(out, var.STR(rfn.out_field), ftdst)


def gen_v2e_spmv_schedule(adj, mfunc, src_frame, dst_frame, edge_frame,
                          out, out_size, edge_map, out_map):
    """Generate v2e SPMV schedule

    Parameters
    ----------
    adj : utils.CtxCachedObject
        function that generates four dgl.ndarray (indptr, indices, inv_indptr,
        inv_indices) representing CSR and transposed-CSR matrix, copies to and
        caches on to given context
    mfunc : list of builtin message func
    src_frame : var.Var
        input source node features
    dst_frame : var.Var
        input destination node features
    edge_frame : var.Var
        input edge features
    out : var.Var
        output message features
    out_size : int
        number of output messages
    out_map : utils.CtxCachedObject
        function that generates a mapping from message ids to edge ids and
        caches on given context
    """
    spmat = var.SPMAT(adj)
    for mfn in mfunc:
        fmsg = mfn(spmat, src_frame, dst_frame, edge_frame, out_size,
                   edge_map=edge_map, out_map=out_map)
        ir.WRITE_COL_(out, var.STR(mfn.out_field), fmsg)


def gen_e2v_spmv_schedule(adj, rfunc, message_frame, edge_map, out, out_size,
                          out_map):
    """Generate e2v SPMV schedule.

    Parameters
    ----------
    adj : utils.CtxCachedObject
        function that generates four dgl.ndarray (indptr, indices, inv_indptr,
        inv_indices) representing CSR and transposed-CSR matrix, copies to and
        caches on to given context
    rfunc : list of builtin reduce func
    message_frame : var.Var
        input message features
    edge_map : CtxCachedObject
        Function that copies and caches mapping from message id to edge id for
        CSR and transposed CSR matrix on given context
    out : var.Var
        output node features
    out_size : int
        number of output nodes
    out_map : utils.CtxCachedObject
        function that generates a mapping from destination nodes to consecutive
        ids and caches on given context
    """
    spmat = var.SPMAT(adj)
    for rfn in rfunc:
        ftdst = rfn(spmat, message_frame, out_size, edge_map=edge_map,
                    out_map=out_map)
        ir.WRITE_COL_(out, var.STR(rfn.out_field), ftdst)


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

    def edge_map(ctx):
        return gidx.csr_adjacency_matrix(True, ctx)[2]

    def inv_edge_map(ctx):
        return gidx.csr_adjacency_matrix(False, ctx)[2]

    msg_map = edge_map
    inv_msg_map = inv_edge_map

    return lambda ctx: (gidx.csr_adjacency_matrix(True, ctx)[:2] +
                        gidx.csr_adjacency_matrix(False, ctx)[:2]), \
        edge_map, inv_edge_map, msg_map, inv_msg_map


def _build_adj_matrix_index_uv(u, v, num_src, num_dst):
    """Build adj matrix index and shape using the given (u, v) edges.

    The matrix is of shape (len(reduce_nodes), n), where n is the number of
    nodes in the graph. Therefore, when doing SPMV, the src node data should be
    all the node features.

    The dst nodes will be sorted in the *unique-ascending* order of
    their ids. This is compatible with other reduce scheduler such as
    degree-bucketing scheduler.

    Paramters
    ---------
    u : utils.index
        Source nodes
    v : utils.index
        Destination nodes
    num_src : int
        Number of source nodes.
    num_dst : int
        Number of destination nodes

    Returns
    -------
    sparse index
        The sparse index.
    tuple of int
        The dense shape.
    """
    u = u.tousertensor()
    v = v.tousertensor()
    u = F.zerocopy_to_numpy(u)
    v = F.zerocopy_to_numpy(v)
    dat = np.arange(len(v), dtype=np.int64)
    csr = sp.csr_matrix((dat, (u, v)), shape=(num_src, num_dst))
    inv_csr = sp.csr_matrix((dat, (v, u)), shape=(num_dst, num_src))
    spmat = (csr.indptr.astype(np.int32),
             csr.indices.astype(np.int32),
             csr.data,
             inv_csr.indptr.astype(np.int32),
             inv_csr.indices.astype(np.int32),
             inv_csr.data)
    spmat = list(map(F.zerocopy_from_numpy, spmat))
    return spmat


def build_adj_matrix_uv(edge_tuples, num_src, num_dst):
    """Build adj matrix using the given (u, v) edges and target nodes.

    The matrix is of shape (len(reduce_nodes), n), where n is the number of
    nodes in the graph. Therefore, when doing SPMV, the src node data should be
    all the node features.

    Parameters
    ---------
    u : utils.Index
        Source nodes
    v : utils.Index
        Destination nodes
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
    u, v, eid = edge_tuples
    eid = eid.tousertensor()
    res = _build_adj_matrix_index_uv(u, v, num_src, num_dst)
    edge_map = F.zerocopy_to_dgl_ndarray(eid[res[2]])
    inv_edge_map = F.zerocopy_to_dgl_ndarray(eid[res[5]])
    res = list(map(F.zerocopy_to_dgl_ndarray, res))
    indptr, indices, msg_map, inv_indptr, inv_indices, inv_msg_map= res

    def spmat(ctx):
        return list(map(lambda x: nd.array(x, ctx=ctx),
                        (indptr, indices, inv_indptr, inv_indices)))

    return utils.CtxCachedObject(spmat), \
        utils.CtxCachedObject(lambda ctx: nd.array(edge_map, ctx=ctx)), \
        utils.CtxCachedObject(lambda ctx: nd.array(inv_edge_map, ctx=ctx)), \
        utils.CtxCachedObject(lambda ctx: nd.array(msg_map, ctx=ctx)), \
        utils.CtxCachedObject(lambda ctx: nd.array(inv_msg_map, ctx=ctx)) \


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
    # FIXME (lingfan): nodeflow does not support get both csr and transposed
    # csr, for now use scipy to implement
    u, v, _ = graph.block_edges(block_id)
    u = utils.Index(u)
    v = utils.Index(v)
    num_src = graph.layer_size(block_id)
    num_dst = graph.layer_size(block_id + 1)
    return build_adj_matrix_uv(u, v, num_src, num_dst)
