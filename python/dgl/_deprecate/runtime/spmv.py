"""Module for SPMV rules."""
from __future__ import absolute_import
from functools import partial

from ...base import DGLError
from ... import backend as F
from ... import utils
from ... import ndarray as nd
from ...heterograph_index import create_unitgraph_from_coo

from . import ir
from .ir import var


def gen_v2v_spmv_schedule(graph, mfunc, rfunc, src_frame, dst_frame,
                          edge_frame, out, out_size, src_map=None,
                          dst_map=None, edge_map=None, out_map=None):
    """Generate v2v spmv schedule.

    Parameters
    ----------
    graph : utils.CtxCachedObject
        Function that generates immutable graph index on given context
    mfunc : list of builtin message func
        Builtin message function list
    rfunc : list of builtin reduce func
        Builtin reduce function list
    src_frame : var.Var
        Input source node features
    dst_frame : var.Var
        Input destination node features
    edge_frame : var.Var
        Input edge features
    out : var.Var
        Output node features
    out_size : int
        Number of output nodes
    src_map : utils.CtxCachedObject
        Function that generates source node id mapping array on given context
    dst_map : utils.CtxCachedObject
        Function that generates destination node id mapping array on given
        context
    edge_map : utils.CtxCachedObject
        Function that generates edge id mapping array on given context
    out_map : utils.CtxCachedObject
        Function that generates output id mapping array on given context
    """
    fld2mfunc = {fn.out_field: fn for fn in mfunc}
    for rfn in rfunc:
        mfld = rfn.msg_field
        if mfld not in fld2mfunc:
            raise DGLError('Reduce function requires message field "%s",'
                           ' but no message function generates it.' % mfld)
        mfn = fld2mfunc[mfld]
        ftdst = mfn._invoke(graph, src_frame, dst_frame, edge_frame, out_size,
                            src_map, dst_map, edge_map, out_map,
                            reducer=rfn.name)
        ir.WRITE_COL_(out, var.STR(rfn.out_field), ftdst)


def gen_v2e_spmv_schedule(graph, mfunc, src_frame, dst_frame, edge_frame, out,
                          out_size, src_map=None, dst_map=None, edge_map=None,
                          out_map=None):
    """Generate v2e SPMV schedule

    Parameters
    ----------
    graph : utils.CtxCachedObject
        Function that generates immutable graph index on given context
    mfunc : list of builtin message func
        Builtin message function list
    src_frame : var.Var
        Input source node features
    dst_frame : var.Var
        Input destination node features
    edge_frame : var.Var
        Input edge features
    out : var.Var
        Output node features
    out_size : int
        Number of output nodes
    src_map : utils.CtxCachedObject
        Function that generates source node id mapping array on given context
    dst_map : utils.CtxCachedObject
        Function that generates destination node id mapping array on given
        context
    edge_map : utils.CtxCachedObject
        Function that generates edge id mapping array on given context
    out_map : utils.CtxCachedObject
        Function that generates output id mapping array on given context
    """
    for mfn in mfunc:
        fmsg = mfn._invoke(graph, src_frame, dst_frame, edge_frame, out_size,
                           src_map, dst_map, edge_map, out_map=out_map,
                           reducer="none")
        ir.WRITE_COL_(out, var.STR(mfn.out_field), fmsg)


def gen_e2v_spmv_schedule(graph, rfunc, message_frame, out, out_size,
                          edge_map=None, out_map=None):
    """Generate e2v SPMV schedule.

    Parameters
    ----------
    graph : utils.CtxCachedObject
        Function that generates immutable graph index on given context
    rfunc : list of builtin reduce func
        Builtin reduce function list
    message_frame : var.Var
        Message features
    out : var.Var
        Output node features
    out_size : int
        Number of output nodes
    edge_map : utils.CtxCachedObject
        Function that generates edge id mapping array on given context
    out_map : utils.CtxCachedObject
        Function that generates output id mapping array on given context
    """
    for rfn in rfunc:
        ftdst = rfn._invoke(graph, message_frame, out_size, edge_map=edge_map,
                            out_map=out_map)
        ir.WRITE_COL_(out, var.STR(rfn.out_field), ftdst)


def build_gidx_and_mapping_graph(graph):
    """Build immutable graph index of the whole graph.

    Parameters
    ----------
    graph : GraphAdapter
        Graph

    Returns
    -------
    graph : utils.CtxCachedObject
        Function that generates a immutable graph index on given context
    edge_map : utils.CtxCachedObject
        Function that generates forward and backward edge mapping on given
        context
    nbits : int
        Number of ints needed to represent the graph
    """
    return graph.get_immutable_gidx, None, graph.bits_needed()

def build_gidx_and_mapping_uv(edge_tuples, num_src, num_dst):
    """Build immutable graph index and mapping using the given (u, v) edges

    The matrix is of shape (num_src, num_dst).

    Parameters
    ---------
    edge_tuples : tuple of three utils.Index
        A tuple of (u, v, eid)
    num_src : int
        Number of source nodes.
    num_dst : int
        Number of destination nodes.

    Returns
    -------
    graph : utils.CtxCachedObject
        Function that generates a immutable graph index on given context
    edge_map : utils.CtxCachedObject
        Function that generates forward and backward edge mapping on given
        context
    nbits : int
        Number of ints needed to represent the graph
    """
    u, v, eid = edge_tuples
    gidx = create_unitgraph_from_coo(2, num_src, num_dst,
                                     u.tousertensor(), v.tousertensor(), ['coo', 'csr', 'csc'])
    forward, backward = gidx.get_csr_shuffle_order(0)
    eid = eid.tousertensor()
    nbits = gidx.bits_needed(0)
    forward_map = utils.to_nbits_int(F.gather_row(eid, forward.tousertensor()), nbits)
    backward_map = utils.to_nbits_int(F.gather_row(eid, backward.tousertensor()), nbits)
    forward_map = F.zerocopy_to_dgl_ndarray(forward_map)
    backward_map = F.zerocopy_to_dgl_ndarray(backward_map)
    edge_map = utils.CtxCachedObject(
        lambda ctx: (nd.array(forward_map, ctx=ctx),
                     nd.array(backward_map, ctx=ctx)))
    return partial(gidx.get_unitgraph, 0), edge_map, nbits

def build_gidx_and_mapping_block(graph, block_id, edge_tuples=None):
    """Build immutable graph index and mapping for node flow

    Parameters
    ----------
    graph : NodeFlow
        The NodeFlow
    block_id : int
        the block Id
    edge_tuple :  tuple of three utils.Index
        A tuple of (u, v, eid)

    Returns
    -------
    graph : utils.CtxCachedObject
        Function that generates a immutable graph index on given context
    edge_map : utils.CtxCachedObject
        Function that generates forward and backward edge mapping on given
        context
    nbits : int
        Number of ints needed to represent the graph
    """
    if edge_tuples is None:
        u, v, eid = graph.block_edges(block_id, remap_local=True)
        u = utils.toindex(u)
        v = utils.toindex(v)
        eid = utils.toindex(eid)
    else:
        u, v, eid = edge_tuples
    num_src, num_dst = graph.layer_size(block_id), graph.layer_size(block_id + 1)
    gidx, edge_map, nbits = build_gidx_and_mapping_uv((u, v, eid), num_src, num_dst)
    return gidx, edge_map, nbits
