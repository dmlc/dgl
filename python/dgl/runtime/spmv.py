"""Module for SPMV rules."""
from __future__ import absolute_import

import scipy.sparse as sp
import numpy as np

from ..base import DGLError
from .. import backend as F
from .. import utils
from .. import ndarray as nd
from ..graph_index import create_graph_index

from . import ir
from .ir import var


def gen_v2v_spmv_schedule(graph, mfunc, rfunc, src_frame, dst_frame, edge_frame,
                          out, out_size, src_map=None, dst_map=None,
                          edge_map=None, out_map=None):
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
    fld2mfunc = {fn.out_field: fn for fn in mfunc}
    for rfn in rfunc:
        mfld = rfn.msg_field
        if mfld not in fld2mfunc:
            raise DGLError('Reduce function requires message field "%s",'
                           ' but no message function generates it.' % mfld)
        mfn = fld2mfunc[mfld]
        ftdst = mfn(graph, src_frame, dst_frame, edge_frame, out_size, src_map,
                    dst_map, edge_map, out_map, reducer=rfn.name)
        ir.WRITE_COL_(out, var.STR(rfn.out_field), ftdst)


def gen_v2e_spmv_schedule(graph, mfunc, src_frame, dst_frame, edge_frame,
                          out, out_size, edge_map, out_map=None):
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
    for mfn in mfunc:
        fmsg = mfn(graph, src_frame, dst_frame, edge_frame, out_size,
                   edge_map=edge_map, out_map=out_map)
        ir.WRITE_COL_(out, var.STR(mfn.out_field), fmsg)


def gen_e2v_spmv_schedule(graph, rfunc, message_frame, edge_map, out, out_size,
                          out_map=None):
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
    for rfn in rfunc:
        ftdst = rfn(graph, message_frame, out_size, edge_map=edge_map,
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
    return lambda ctx: gidx.get_immutable_gidx(ctx), None, gidx.bits_needed()


def build_adj_matrix_uv(edge_tuples, num_nodes):
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
    gidx = create_graph_index()
    gidx.add_nodes(num_nodes)
    gidx.add_edges(u, v)
    forward, backward = gidx.get_csr_shuffle_order()
    eid = eid.tousertensor()
    nbits = gidx.bits_needed()
    forward_map = utils.to_nbits_int(eid[forward.tousertensor()], nbits)
    backward_map = utils.to_nbits_int(eid[backward.tousertensor()], nbits)
    forward_map = F.zerocopy_to_dgl_ndarray(forward_map)
    backward_map = F.zerocopy_to_dgl_ndarray(backward_map)
    return utils.CtxCachedObject(lambda ctx: gidx.get_immutable_gidx(ctx)), \
        utils.CtxCachedObject(lambda ctx: (nd.array(forward_map, ctx=ctx),
                                           nd.array(backward_map, ctx=ctx))), \
        nbits


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
    u, v, eid = graph.block_edges(block_id)
    u = utils.Index(u)
    v = utils.Index(v)
    eid = utils.Index(eid)
    num_src = graph.layer_size(block_id)
    num_dst = graph.layer_size(block_id + 1)
    return build_adj_matrix_uv((u, v, eid), num_src, num_dst)
