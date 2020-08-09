"""For different schedulers"""
from __future__ import absolute_import

from ... import utils
from ..._ffi.function import _init_api
from ...base import DGLError
from ... import backend as F
from ..frame import frame_like, FrameRef
from ...function.base import BuiltinFunction
from ..udf import EdgeBatch, NodeBatch
from ... import ndarray as nd

from . import ir
from .ir import var
from . import degree_bucketing as db
from . import spmv

__all__ = [
    "schedule_send",
    "schedule_recv",
    "schedule_update_all",
    "schedule_snr",
    "schedule_apply_nodes",
    "schedule_apply_edges",
    "schedule_group_apply_edge",
    "schedule_push",
    "schedule_pull"
]

def schedule_send(graph,
                  u, v, eid,
                  message_func,
                  msgframe=None):
    """Schedule send

    Parameters
    ----------
    graph: GraphAdaptor
        Graph
    u : utils.Index
        Source nodes
    v : utils.Index
        Destination nodes
    eid : utils.Index
        Ids of sending edges
    message_func: callable or list of callable
        The message function
    msgframe : FrameRef, optional
        The storage to write messages to. If None, use graph.msgframe.
    """
    var_mf = var.FEAT_DICT(msgframe if msgframe is not None else graph.msgframe)
    var_src_nf = var.FEAT_DICT(graph.srcframe)
    var_dst_nf = var.FEAT_DICT(graph.dstframe)
    var_ef = var.FEAT_DICT(graph.edgeframe)
    var_eid = var.IDX(eid)

    var_msg = _gen_send(graph=graph,
                        u=u,
                        v=v,
                        eid=eid,
                        mfunc=message_func,
                        var_src_nf=var_src_nf,
                        var_dst_nf=var_dst_nf,
                        var_ef=var_ef)

    # write tmp msg back
    ir.WRITE_ROW_(var_mf, var_eid, var_msg)
    # set message indicator to 1
    graph.msgindicator = graph.msgindicator.set_items(eid, 1)

def schedule_recv(graph,
                  recv_nodes,
                  reduce_func,
                  apply_func,
                  inplace,
                  outframe=None):
    """Schedule recv.

    Parameters
    ----------
    graph: GraphAdaptor
        Graph
    recv_nodes: utils.Index
        Nodes to recv.
    reduce_func: callable or list of callable
        The reduce function
    apply_func: callable
        The apply node function
    inplace: bool
        If True, the update will be done in place
    outframe : FrameRef, optional
        The storage to write output data. If None, use graph.dstframe.
    """
    src, dst, eid = graph.in_edges(recv_nodes)
    if len(eid) > 0:
        nonzero_idx = graph.msgindicator.get_items(eid).nonzero()
        eid = eid.get_items(nonzero_idx)
        src = src.get_items(nonzero_idx)
        dst = dst.get_items(nonzero_idx)
    if len(eid) == 0:
        # Downgrade to apply nodes if
        #   1) all recv nodes are 0-degree nodes
        #   2) no send has been called
        if apply_func is not None:
            schedule_apply_nodes(recv_nodes, apply_func, graph.dstframe,
                                 inplace, outframe, ntype=graph.canonical_etype[-1])
    else:
        var_dst_nf = var.FEAT_DICT(graph.dstframe, 'dst_nf')
        var_out_nf = var_dst_nf if outframe is None else var.FEAT_DICT(outframe, name='out_nf')
        # sort and unique the argument
        recv_nodes, _ = F.sort_1d(F.unique(recv_nodes.tousertensor()))
        recv_nodes = utils.toindex(recv_nodes, graph.gidx.dtype)
        var_recv_nodes = var.IDX(recv_nodes, name='recv_nodes')
        # reduce
        reduced_feat = _gen_reduce(graph, reduce_func, (src, dst, eid),
                                   recv_nodes)
        # apply
        final_feat = _apply_with_accum(var_recv_nodes, var_dst_nf,
                                       reduced_feat, apply_func,
                                       ntype=graph.canonical_etype[-1])
        if inplace:
            ir.WRITE_ROW_INPLACE_(var_out_nf, var_recv_nodes, final_feat)
        else:
            ir.WRITE_ROW_(var_out_nf, var_recv_nodes, final_feat)
        # set message indicator to 0
        graph.msgindicator = graph.msgindicator.set_items(eid, 0)
        if not graph.msgindicator.has_nonzero():
            ir.CLEAR_FRAME_(var.FEAT_DICT(graph.msgframe, name='mf'))

def schedule_snr(graph,
                 edge_tuples,
                 message_func,
                 reduce_func,
                 apply_func,
                 inplace,
                 outframe=None):
    """Schedule send_and_recv.

    Currently it builds a subgraph from edge_tuples with the same number of
    nodes as the original graph, so that routines for whole-graph updates
    (e.g. fused kernels) could be reused.

    Parameters
    ----------
    graph: GraphAdaptor
        Graph
    edge_tuples: tuple
        A tuple of (src ids, dst ids, edge ids) representing edges to perform
        send_and_recv
    message_func: callable or list of callable
        The message function
    reduce_func: callable or list of callable
        The reduce function
    apply_func: callable
        The apply node function
    inplace: bool
        If True, the update will be done in place
    outframe : FrameRef, optional
        The storage to write output data. If None, use graph.dstframe.
    """
    u, v, eid = edge_tuples
    recv_nodes, _ = F.sort_1d(F.unique(v.tousertensor()))
    recv_nodes = utils.toindex(recv_nodes, graph.gidx.dtype)
    # create vars
    var_dst_nf = var.FEAT_DICT(graph.dstframe, 'dst_nf')
    var_out_nf = var_dst_nf if outframe is None else var.FEAT_DICT(outframe, name='out_nf')
    var_u = var.IDX(u)
    var_v = var.IDX(v)
    var_eid = var.IDX(eid)
    var_recv_nodes = var.IDX(recv_nodes, name='recv_nodes')
    # generate send and reduce schedule
    uv_getter = lambda: (var_u, var_v)
    adj_creator = lambda: spmv.build_gidx_and_mapping_uv(
        edge_tuples, graph.num_src(), graph.num_dst())
    out_map_creator = lambda nbits: _build_idx_map(recv_nodes, nbits)
    reduced_feat = _gen_send_reduce(src_node_frame=graph.srcframe,
                                    dst_node_frame=graph.dstframe,
                                    edge_frame=graph.edgeframe,
                                    message_func=message_func,
                                    reduce_func=reduce_func,
                                    var_send_edges=var_eid,
                                    var_reduce_nodes=var_recv_nodes,
                                    uv_getter=uv_getter,
                                    adj_creator=adj_creator,
                                    out_map_creator=out_map_creator,
                                    canonical_etype=graph.canonical_etype)
    # generate apply schedule
    final_feat = _apply_with_accum(var_recv_nodes, var_dst_nf, reduced_feat,
                                   apply_func, ntype=graph.canonical_etype[-1])
    if inplace:
        ir.WRITE_ROW_INPLACE_(var_out_nf, var_recv_nodes, final_feat)
    else:
        ir.WRITE_ROW_(var_out_nf, var_recv_nodes, final_feat)

def schedule_update_all(graph,
                        message_func,
                        reduce_func,
                        apply_func,
                        outframe=None):
    """Get send and recv schedule

    Parameters
    ----------
    graph: GraphAdaptor
        Graph
    message_func: callable or list of callable
        The message function
    reduce_func: callable or list of callable
        The reduce function
    apply_func: callable
        The apply node function
    outframe : FrameRef, optional
        The storage to write output data. If None, use graph.dstframe.
    """
    if graph.num_edges() == 0:
        # All the nodes are zero degree; downgrade to apply nodes
        if apply_func is not None:
            nodes = utils.toindex(slice(0, graph.num_dst()), graph.gidx.dtype)
            schedule_apply_nodes(nodes, apply_func, graph.dstframe,
                                 inplace=False, outframe=outframe,
                                 ntype=graph.canonical_etype[-1])
    else:
        eid = utils.toindex(slice(0, graph.num_edges()), graph.gidx.dtype) # ALL
        recv_nodes = utils.toindex(slice(0, graph.num_dst()), graph.gidx.dtype) # ALL
        # create vars
        var_dst_nf = var.FEAT_DICT(graph.dstframe, name='dst_nf')
        var_out_nf = var_dst_nf if outframe is None else var.FEAT_DICT(outframe, name='out_nf')
        var_recv_nodes = var.IDX(recv_nodes, name='recv_nodes')
        var_eid = var.IDX(eid)
        # generate send + reduce
        def uv_getter():
            src, dst, _ = graph.edges('eid')
            return var.IDX(src), var.IDX(dst)
        adj_creator = lambda: spmv.build_gidx_and_mapping_graph(graph)
        out_map_creator = lambda nbits: None
        reduced_feat = _gen_send_reduce(src_node_frame=graph.srcframe,
                                        dst_node_frame=graph.dstframe,
                                        edge_frame=graph.edgeframe,
                                        message_func=message_func,
                                        reduce_func=reduce_func,
                                        var_send_edges=var_eid,
                                        var_reduce_nodes=var_recv_nodes,
                                        uv_getter=uv_getter,
                                        adj_creator=adj_creator,
                                        out_map_creator=out_map_creator,
                                        canonical_etype=graph.canonical_etype)
        # generate optional apply
        final_feat = _apply_with_accum(var_recv_nodes, var_dst_nf,
                                       reduced_feat, apply_func,
                                       ntype=graph.canonical_etype[-1])
        ir.WRITE_DICT_(var_out_nf, final_feat)

def schedule_apply_nodes(v,
                         apply_func,
                         node_frame,
                         inplace,
                         outframe=None,
                         ntype=None):
    """Get apply nodes schedule

    Parameters
    ----------
    v : utils.Index
        Nodes to apply
    apply_func : callable
        The apply node function
    node_frame : FrameRef
        Node feature frame.
    inplace: bool
        If True, the update will be done in place
    outframe : FrameRef, optional
        The storage to write output data. If None, use the given node_frame.
    ntype : str, optional
        The node type, if running on a heterograph.
        If None, assuming it's running on a homogeneous graph.

    Returns
    -------
    A list of executors for DGL Runtime
    """
    var_v = var.IDX(v)
    var_nf = var.FEAT_DICT(node_frame, name='nf')
    var_out_nf = var_nf if outframe is None else var.FEAT_DICT(outframe, name='out_nf')
    v_nf = ir.READ_ROW(var_nf, var_v)
    def _afunc_wrapper(node_data):
        nbatch = NodeBatch(v, node_data, ntype=ntype)
        return apply_func(nbatch)
    afunc = var.FUNC(_afunc_wrapper)
    applied_feat = ir.NODE_UDF(afunc, v_nf)
    if inplace:
        ir.WRITE_ROW_INPLACE_(var_out_nf, var_v, applied_feat)
    else:
        ir.WRITE_ROW_(var_out_nf, var_v, applied_feat)

def schedule_nodeflow_apply_nodes(graph,
                                  layer_id,
                                  v,
                                  apply_func,
                                  inplace):
    """Get apply nodes schedule in NodeFlow.

    Parameters
    ----------
    graph: NodeFlow
        The NodeFlow to use
    layer_id : int
        The layer where we apply node update function.
    v : utils.Index
        Nodes to apply
    apply_func: callable
        The apply node function
    inplace: bool
        If True, the update will be done in place

    Returns
    -------
    A list of executors for DGL Runtime
    """
    var_nf = var.FEAT_DICT(graph._get_node_frame(layer_id), name='nf')
    var_v = var.IDX(v)
    v_nf = ir.READ_ROW(var_nf, var_v)
    def _afunc_wrapper(node_data):
        nbatch = NodeBatch(v, node_data)
        return apply_func(nbatch)
    afunc = var.FUNC(_afunc_wrapper)
    applied_feat = ir.NODE_UDF(afunc, v_nf)
    # TODO we need to avoid index_copy here.
    if inplace:
        ir.WRITE_ROW_INPLACE_(var_nf, var_v, applied_feat)
    else:
        ir.WRITE_ROW_(var_nf, var_v, applied_feat)

def schedule_apply_edges(graph,
                         u, v, eid,
                         apply_func,
                         inplace,
                         outframe=None):
    """Get apply edges schedule

    Parameters
    ----------
    graph: GraphAdaptor
        Graph
    u : utils.Index
        Source nodes of edges to apply
    v : utils.Index
        Destination nodes of edges to apply
    eid : utils.Index
        Ids of sending edges
    apply_func: callable
        The apply edge function
    inplace: bool
        If True, the update will be done in place
    outframe : FrameRef, optional
        The storage to write output data. If None, use graph.edge_frame.

    Returns
    -------
    A list of executors for DGL Runtime
    """
    # vars
    var_src_nf = var.FEAT_DICT(graph.srcframe, 'uframe')
    var_dst_nf = var.FEAT_DICT(graph.dstframe, 'vframe')
    var_ef = var.FEAT_DICT(graph.edgeframe, 'eframe')
    var_out_ef = var_ef if outframe is None else var.FEAT_DICT(outframe, 'out_ef')
    var_out = _gen_send(graph=graph, u=u, v=v, eid=eid, mfunc=apply_func,
                        var_src_nf=var_src_nf, var_dst_nf=var_dst_nf,
                        var_ef=var_ef)
    var_eid = var.IDX(eid)
    # schedule apply edges
    if inplace:
        ir.WRITE_ROW_INPLACE_(var_out_ef, var_eid, var_out)
    else:
        ir.WRITE_ROW_(var_ef, var_eid, var_out)

def schedule_nodeflow_apply_edges(graph, block_id,
                                  u, v, eid,
                                  apply_func,
                                  inplace):
    """Get apply edges schedule in NodeFlow.

    Parameters
    ----------
    graph: NodeFlow
        The NodeFlow to use
    block_id : int
        The block whose edges we apply edge update function.
    u : utils.Index
        Source nodes of edges to apply
    v : utils.Index
        Destination nodes of edges to apply
    eid : utils.Index
        Ids of sending edges
    apply_func: callable
        The apply edge function
    inplace: bool
        If True, the update will be done in place

    Returns
    -------
    A list of executors for DGL Runtime
    """
    # vars
    in_var_nf = var.FEAT_DICT(graph._get_node_frame(block_id), name='in_nf')
    out_var_nf = var.FEAT_DICT(graph._get_node_frame(block_id + 1),
                               name='out_nf')
    var_ef = var.FEAT_DICT(graph._get_edge_frame(block_id), name='ef')
    var_out = _gen_send(graph, u, v, eid, apply_func, in_var_nf, out_var_nf,
                        var_ef, block_id=block_id)
    var_eid = var.IDX(eid)
    if inplace:
        ir.WRITE_ROW_INPLACE_(var_ef, var_eid, var_out)
    else:
        ir.WRITE_ROW_(var_ef, var_eid, var_out)

def schedule_push(graph,
                  u,
                  message_func,
                  reduce_func,
                  apply_func,
                  inplace,
                  outframe=None):
    """Get push schedule

    Parameters
    ----------
    graph: GraphAdaptor
        Graph
    u : utils.Index
        Source nodes for push
    message_func: callable or list of callable
        The message function
    reduce_func: callable or list of callable
        The reduce function
    apply_func: callable
        The apply node function
    inplace: bool
        If True, the update will be done in place
    outframe : FrameRef, optional
        The storage to write output data. If None, use graph.dstframe.
    """
    u, v, eid = graph.out_edges(u)
    if len(eid) == 0:
        # All the pushing nodes have no out edges. No computation is scheduled.
        return
    schedule_snr(graph, (u, v, eid),
                 message_func, reduce_func, apply_func,
                 inplace, outframe)

def schedule_pull(graph,
                  pull_nodes,
                  message_func,
                  reduce_func,
                  apply_func,
                  inplace,
                  outframe=None):
    """Get pull schedule

    Parameters
    ----------
    graph: GraphAdaptor
        Graph
    pull_nodes : utils.Index
        Destination nodes for pull
    message_func: callable or list of callable
        The message function
    reduce_func: callable or list of callable
        The reduce function
    apply_func: callable
        The apply node function
    inplace: bool
        If True, the update will be done in place
    outframe : FrameRef, optional
        The storage to write output data. If None, use graph.dstframe.
    """
    # TODO(minjie): `in_edges` can be omitted if message and reduce func pairs
    #   can be specialized to SPMV. This needs support for creating adjmat
    #   directly from pull node frontier.
    u, v, eid = graph.in_edges(pull_nodes)
    if len(eid) == 0:
        # All the nodes are 0deg; downgrades to apply.
        if apply_func is not None:
            schedule_apply_nodes(pull_nodes, apply_func, graph.dstframe, inplace,
                                 outframe, ntype=graph.canonical_etype[-1])
    else:
        # TODO(Allen): Change operation to dgl operation
        pull_nodes, _ = F.sort_1d(F.unique(pull_nodes.tousertensor()))
        pull_nodes = utils.toindex(pull_nodes, graph.gidx.dtype)
        # create vars
        var_dst_nf = var.FEAT_DICT(graph.dstframe, name='dst_nf')
        var_out_nf = var_dst_nf if outframe is None else var.FEAT_DICT(outframe, name='out_nf')
        var_pull_nodes = var.IDX(pull_nodes, name='pull_nodes')
        var_u = var.IDX(u)
        var_v = var.IDX(v)
        var_eid = var.IDX(eid)
        # generate send and reduce schedule
        uv_getter = lambda: (var_u, var_v)
        adj_creator = lambda: spmv.build_gidx_and_mapping_uv(
            (u, v, eid), graph.num_src(), graph.num_dst())
        out_map_creator = lambda nbits: _build_idx_map(pull_nodes, nbits)
        reduced_feat = _gen_send_reduce(graph.srcframe,
                                        graph.dstframe, graph.edgeframe,
                                        message_func, reduce_func, var_eid,
                                        var_pull_nodes, uv_getter, adj_creator,
                                        out_map_creator,
                                        canonical_etype=graph.canonical_etype)
        # generate optional apply
        final_feat = _apply_with_accum(var_pull_nodes, var_dst_nf,
                                       reduced_feat, apply_func,
                                       ntype=graph.canonical_etype[-1])
        if inplace:
            ir.WRITE_ROW_INPLACE_(var_out_nf, var_pull_nodes, final_feat)
        else:
            ir.WRITE_ROW_(var_out_nf, var_pull_nodes, final_feat)

def schedule_group_apply_edge(graph,
                              u, v, eid,
                              apply_func,
                              group_by,
                              inplace,
                              outframe=None):
    """Group apply edges schedule

    Parameters
    ----------
    graph: GraphAdaptor
        Graph
    u : utils.Index
        Source nodes of edges to apply
    v : utils.Index
        Destination nodes of edges to apply
    eid : utils.Index
        Ids of sending edges
    apply_func: callable
        The apply edge function
    group_by : str
        Specify how to group edges. Expected to be either 'src' or 'dst'
    inplace: bool
        If True, the update will be done in place
    outframe : FrameRef, optional
        The storage to write output data. If None, use graph.edgeframe.
    """
    # vars
    var_src_nf = var.FEAT_DICT(graph.srcframe, name='src_nf')
    var_dst_nf = var.FEAT_DICT(graph.dstframe, name='dst_nf')
    var_ef = var.FEAT_DICT(graph.edgeframe, name='ef')
    var_out_ef = var_ef if outframe is None else var.FEAT_DICT(outframe, name='out_ef')
    var_out = var.FEAT_DICT(name='new_ef')
    db.gen_group_apply_edge_schedule(apply_func, u, v, eid, group_by,
                                     var_src_nf, var_dst_nf, var_ef, var_out,
                                     canonical_etype=graph.canonical_etype)
    var_eid = var.IDX(eid)
    if inplace:
        ir.WRITE_ROW_INPLACE_(var_out_ef, var_eid, var_out)
    else:
        ir.WRITE_ROW_(var_out_ef, var_eid, var_out)


def schedule_nodeflow_update_all(graph,
                                 block_id,
                                 message_func,
                                 reduce_func,
                                 apply_func):
    """Get update_all schedule in a block.

    Parameters
    ----------
    graph: NodeFlow
        The NodeFlow to use
    block_id : int
        The block where we perform computation.
    message_func: callable or list of callable
        The message function
    reduce_func: callable or list of callable
        The reduce function
    apply_func: callable
        The apply node function
    """
    # A NodeFlow shouldn't have 0 edges.
    assert graph.block_size(block_id) > 0
    eid = utils.toindex(slice(0, graph.block_size(block_id)))  # ALL
    dest_nodes = utils.toindex(slice(0, graph.layer_size(block_id + 1)))  # ALL
    # create vars
    var_nf = var.FEAT_DICT(graph._get_node_frame(block_id + 1), name='out_nf')
    var_dest_nodes = var.IDX(dest_nodes, name='dest_nodes')
    var_eid = var.IDX(eid)
    # generate send + reduce
    def uv_getter():
        src, dst, _ = graph.block_edges(block_id, remap_local=True)
        return var.IDX(utils.toindex(src)), var.IDX(utils.toindex(dst))
    adj_creator = lambda: spmv.build_gidx_and_mapping_block(graph, block_id)
    out_map_creator = lambda nbits: None
    reduced_feat = _gen_send_reduce(src_node_frame=graph._get_node_frame(block_id),
                                    dst_node_frame=graph._get_node_frame(block_id + 1),
                                    edge_frame=graph._get_edge_frame(block_id),
                                    message_func=message_func,
                                    reduce_func=reduce_func,
                                    var_send_edges=var_eid,
                                    var_reduce_nodes=var_dest_nodes,
                                    uv_getter=uv_getter,
                                    adj_creator=adj_creator,
                                    out_map_creator=out_map_creator)
    # generate optional apply
    final_feat = _apply_with_accum(var_dest_nodes, var_nf, reduced_feat, apply_func)
    ir.WRITE_DICT_(var_nf, final_feat)


def schedule_nodeflow_compute(graph,
                              block_id,
                              u, v, eid,
                              dest_nodes,
                              message_func,
                              reduce_func,
                              apply_func,
                              inplace):
    """Get flow compute schedule in NodeFlow

    Parameters
    ----------
    graph: NodeFlow
        The NodeFlow to use
    block_id : int
        The block where we perform computation.
    u : utils.Index
        Source nodes of edges to apply
    v : utils.Index
        Destination nodes of edges to apply
    eid : utils.Index
        Ids of sending edges
    dest_nodes : utils.Index
        Destination nodes ids
    message_func: callable or list of callable
        The message function
    reduce_func: callable or list of callable
        The reduce function
    apply_func: callable
        The apply node function
    inplace: bool
        If True, the update will be done in place
    """
    # TODO(minjie): `in_edges` can be omitted if message and reduce func pairs
    #   can be specialized to SPMV. This needs support for creating adjmat
    #   directly from pull node frontier.
    if len(eid) == 0:
        # All the nodes are 0deg; downgrades to apply.
        if apply_func is not None:
            schedule_nodeflow_apply_nodes(graph, block_id + 1, dest_nodes,
                                          apply_func, inplace)
    else:
        # create vars
        var_nf = var.FEAT_DICT(graph._get_node_frame(block_id + 1),
                               name='out_nf')
        var_u = var.IDX(u)
        var_v = var.IDX(v)
        var_eid = var.IDX(eid)
        var_dest_nodes = var.IDX(dest_nodes, name='dest_nodes')
        # generate send and reduce schedule
        uv_getter = lambda: (var_u, var_v)
        adj_creator = lambda: spmv.build_gidx_and_mapping_block(
            graph, block_id, (u, v, eid))
        out_map_creator = lambda nbits: _build_idx_map(utils.toindex(dest_nodes), nbits)

        reduced_feat = _gen_send_reduce(src_node_frame=graph._get_node_frame(block_id),
                                        dst_node_frame=graph._get_node_frame(block_id + 1),
                                        edge_frame=graph._get_edge_frame(block_id),
                                        message_func=message_func,
                                        reduce_func=reduce_func,
                                        var_send_edges=var_eid,
                                        var_reduce_nodes=var_dest_nodes,
                                        uv_getter=uv_getter,
                                        adj_creator=adj_creator,
                                        out_map_creator=out_map_creator)
        # generate optional apply
        final_feat = _apply_with_accum(var_dest_nodes, var_nf,
                                       reduced_feat, apply_func)
        if inplace:
            ir.WRITE_ROW_INPLACE_(var_nf, var_dest_nodes, final_feat)
        else:
            ir.WRITE_ROW_(var_nf, var_dest_nodes, final_feat)

def _check_builtin_func_list(func_list):
    """Check whether func_list only contains builtin functions."""
    for fn in func_list:
        if not isinstance(fn, BuiltinFunction):
            raise DGLError("If specify multiple message/reduce functions, \
                           all of them must be builtin")

def _standardize_func_usage(func, func_name):
    """Standardize usages of message and reduce functions
    Message or reduce funtion can be:
        1. a UDF
        2. a dgl builtin function
        3. a list of dgl builtin function

    This function checks if func meets the requirement, and merges last two
    cases by putting builtin function in case 2 into a list

    Returns:
    One single UDF function or a list of builtin function
    """

    if utils.is_iterable(func):
        # func is a list of builtin
        _check_builtin_func_list(func)
        return func
    elif isinstance(func, BuiltinFunction):
        # func is one builtin-in
        return [func]
    else:
        # func is one UDF
        if not callable(func):
            raise DGLError('User-defined %s function must be callable.'
                           ' Got: %s' % (func_name, str(func)))
        return func

def _apply_with_accum(var_nodes, var_nf, var_accum, apply_func, ntype=None):
    """Apply with accumulated features.

    Paramters
    ---------
    var_nodes : var.IDX
        The nodes.
    var_nf : var.FEAT_DICT
        The node features.
    var_accum : var.FEAT_DICT
        The accumulated features.
    apply_func : callable, None
        The apply function.
    ntype : str, optional
        The node type, if running on a heterograph.
        If None, assuming it's running on a homogeneous graph.
    """
    if apply_func:
        # To avoid writing reduced features back to node frame and reading
        # it again for apply phase. Instead, we first read the the node
        # features and "merge" it with the reduced features.
        v_nf = ir.READ_ROW(var_nf, var_nodes)
        v_nf = ir.UPDATE_DICT(v_nf, var_accum)

        def _afunc_wrapper(node_data):
            nbatch = NodeBatch(var_nodes.data, node_data, ntype=ntype)
            return apply_func(nbatch)
        afunc = var.FUNC(_afunc_wrapper)
        applied_feat = ir.NODE_UDF(afunc, v_nf)
        final_feat = ir.UPDATE_DICT(var_accum, applied_feat)
    else:
        final_feat = var_accum
    return final_feat

def _gen_reduce(graph, reduce_func, edge_tuples, recv_nodes):
    """Generate reduce schedule

    Parameters
    ----------
    graph : GraphAdaptor
    reduce_func : callable
    edge_tuples : tuple of utils.Index
    recv_nodes : utils.Index

    Returns
    -------
    var.FEAT_DICT
        The reduced feature dict.
    """
    src, dst, eid = edge_tuples
    rfunc = _standardize_func_usage(reduce_func, 'reduce')
    rfunc_is_list = utils.is_iterable(rfunc)
    # Create a tmp frame to hold the feature data.
    # The frame has the same size and schemes of the
    # node frame.
    # TODO(minjie): should replace this with an IR call to make the program
    # stateless.
    tmpframe = FrameRef(frame_like(graph.dstframe._frame, len(recv_nodes)))

    # vars
    var_msg = var.FEAT_DICT(graph.msgframe, 'msg')
    var_dst_nf = var.FEAT_DICT(graph.dstframe, 'nf')
    var_out = var.FEAT_DICT(data=tmpframe)

    if rfunc_is_list:
        adj, edge_map, nbits = spmv.build_gidx_and_mapping_uv(
            (src, dst, eid), graph.num_src(), graph.num_dst())
        # using edge map instead of message map because messages are in global
        # message frame
        var_out_map = _build_idx_map(recv_nodes, nbits)
        spmv.gen_e2v_spmv_schedule(graph=adj,
                                   rfunc=rfunc,
                                   message_frame=var_msg,
                                   out=var_out,
                                   out_size=len(recv_nodes),
                                   edge_map=edge_map,
                                   out_map=var_out_map)
        return var_out
    else:
        # gen degree bucketing schedule for UDF recv
        db.gen_degree_bucketing_schedule(rfunc, eid, dst, recv_nodes,
                                         var_dst_nf, var_msg, var_out,
                                         ntype=graph.canonical_etype[-1])
        return var_out

def _gen_send_reduce(
        src_node_frame,
        dst_node_frame,
        edge_frame,
        message_func,
        reduce_func,
        var_send_edges,
        var_reduce_nodes,
        uv_getter,
        adj_creator,
        out_map_creator,
        canonical_etype=(None, None, None)):
    """Generate send and reduce schedule.

    The function generates symbolic program for computing
    (1) message function on the given edges (var_send_edges).
    (2) reduce function on the given nodes (var_reduce_nodes).

    If both message_func and reduce_func are DGL builtin functions, the schedule
    will invoke fused message passing kernels (e.g. dgl.backend.binary_reduce) to
    avoid generating explicit edge messages.

    If message_func is UDF while reduce_func is DGL builtin function, the schedule
    first invokes UDF to generate explicit edge messages, and then invokes
    dgl.backend.copy_reduce to reduce messages on the destination nodes.

    If both message_func and reduce_func are UDFs, the schedule first invokes message
    UDF to generate explicit edge messages and then use degree-bucketing to invoke
    reduce UDF.

    Parameters
    ----------
    src_node_frame : NodeFrame
        The node frame of the source nodes.
    dst_node_frame : NodeFrame
        The node frame of the destination nodes.
    edge_frame : NodeFrame
        The frame for the edges between the source and destination nodes.
    message_func : callable, list of builtins
        The message func(s).
    reduce_func : callable, list of builtins
        The reduce func(s).
    var_send_edges : var.IDX
        The edges (ids) to perform send.
    var_reduce_nodes : var.IDX
        Unique and sorted nodes to perform reduce. This should include
        unique(v) + 0deg nodes.
    uv_getter : callable
        Function that returns a pair of var.IDX (u, v) for the triggered edges.
    adj_creator : callable
        Function that returns the adjmat, edge order of csr matrix, and
        bit-width.
    out_map_creator : callable
        A function that returns a mapping from reduce_nodes to relabeled
        consecutive ids
    canonical_etype : tuple[str, str, str], optional
        Canonical edge type if running on a heterograph.
        Default: (None, None, None), if running on a homogeneous graph.

    Returns
    -------
    var.FEAT_DICT
        The reduced feature dict.

    Notes
    -----
    Reduce_nodes are assumed to be in the *unique-ascending* order of the edge
    destination node ids. The returned reduced features will be batched
    following the order of reduce_nodes.
    """
    # NOTE: currently, this function requires all var.IDX to contain concrete
    # data.
    reduce_nodes = var_reduce_nodes.data

    # arg vars
    var_src_nf = var.FEAT_DICT(src_node_frame, name='src_frame')
    var_dst_nf = var.FEAT_DICT(dst_node_frame, name='dst_frame')
    var_ef = var.FEAT_DICT(edge_frame, name='edge_frame')
    var_eid = var_send_edges

    # format the input functions
    mfunc = _standardize_func_usage(message_func, 'message')
    rfunc = _standardize_func_usage(reduce_func, 'reduce')
    mfunc_is_list = utils.is_iterable(mfunc)
    rfunc_is_list = utils.is_iterable(rfunc)

    # Create a tmp frame to hold the feature data. The frame has the same size
    # and schemes of the node frame.
    # TODO(minjie): should replace this with an IR call to make the program
    # stateless.
    tmpframe = FrameRef(frame_like(dst_node_frame._frame, len(reduce_nodes)))
    var_out = var.FEAT_DICT(data=tmpframe)

    # 1. If either mfunc or rfunc is builtin, generate adjmat, edge mapping and
    # message mapping
    if mfunc_is_list or rfunc_is_list:
        adj, edge_map, nbits = adj_creator()

    # 2. If rfunc is builtin, generate a mapping from recv nodes to consecutive
    # output id
    if rfunc_is_list:
        out_map = out_map_creator(nbits)

    # 3. First try fused message and reduce function
    if mfunc_is_list and rfunc_is_list:
        # builtin message + builtin reducer
        spmv.gen_v2v_spmv_schedule(graph=adj,
                                   mfunc=mfunc,
                                   rfunc=rfunc,
                                   src_frame=var_src_nf,
                                   dst_frame=var_dst_nf,
                                   edge_frame=var_ef,
                                   out=var_out,
                                   out_size=len(reduce_nodes),
                                   edge_map=edge_map,
                                   out_map=out_map)
        return var_out

    var_u, var_v = uv_getter()

    # 4. Unable to fuse, then generate message
    if mfunc_is_list:
        # messages are builtin but reduce is UDF
        # Create a tmp frame to hold the message.
        # TODO: should replace this with an IR call to make the program
        # stateless.
        n_message = len(var_eid.data)
        tmp_msg_frame = FrameRef(frame_like(edge_frame._frame, n_message))
        var_mf = var.FEAT_DICT(data=tmp_msg_frame)
        spmv.gen_v2e_spmv_schedule(graph=adj,
                                   mfunc=mfunc,
                                   src_frame=var_src_nf,
                                   dst_frame=var_dst_nf,
                                   edge_frame=var_ef,
                                   out=var_mf,
                                   out_size=n_message,
                                   edge_map=edge_map)
    else:
        # generate UDF send schedule
        var_mf = _gen_udf_send(var_src_nf, var_dst_nf, var_ef, var_u,
                               var_v, var_eid, mfunc, canonical_etype=canonical_etype)

    # 6. Generate reduce
    if rfunc_is_list:
        # UDF message + builtin reducer
        spmv.gen_e2v_spmv_schedule(graph=adj,
                                   rfunc=rfunc,
                                   message_frame=var_mf,
                                   out=var_out,
                                   out_size=len(reduce_nodes),
                                   edge_map=None,  # messages are stored compactly
                                   out_map=out_map)
        return var_out
    else:
        # gen degree bucketing schedule for UDF recv
        mid = utils.toindex(slice(0, len(var_v.data)), var_v.data.dtype)
        db.gen_degree_bucketing_schedule(rfunc, mid, var_v.data,
                                         reduce_nodes, var_dst_nf, var_mf,
                                         var_out, ntype=canonical_etype[-1])
        return var_out

def _gen_udf_send(var_src_nf, var_dst_nf, var_ef, u, v, eid, mfunc,
                  canonical_etype=(None, None, None)):
    """Internal function to generate send schedule for UDF message function."""
    fdsrc = ir.READ_ROW(var_src_nf, u)
    fddst = ir.READ_ROW(var_dst_nf, v)
    fdedge = ir.READ_ROW(var_ef, eid)
    def _mfunc_wrapper(src_data, edge_data, dst_data):
        ebatch = EdgeBatch((u.data, v.data, eid.data),
                           src_data, edge_data, dst_data,
                           canonical_etype=canonical_etype)
        return mfunc(ebatch)
    _mfunc_wrapper = var.FUNC(_mfunc_wrapper)
    msg = ir.EDGE_UDF(_mfunc_wrapper, fdsrc, fdedge, fddst)
    return msg

def _gen_send(graph, u, v, eid, mfunc, var_src_nf, var_dst_nf, var_ef, block_id=None):
    """Internal function to generate send schedule"""
    mfunc = _standardize_func_usage(mfunc, 'message')
    mfunc_is_list = utils.is_iterable(mfunc)
    # vars
    var_u = var.IDX(u)
    var_v = var.IDX(v)
    var_eid = var.IDX(eid)

    if mfunc_is_list:
        if not hasattr(graph, 'num_edges'):
            # XXX(minjie): a temporary hack to detect Nodeflow object
            res = spmv.build_gidx_and_mapping_block(graph, block_id)
        elif eid.is_slice(0, graph.num_edges()):
            # full graph case
            res = spmv.build_gidx_and_mapping_graph(graph)
        else:
            res = spmv.build_gidx_and_mapping_uv(
                (u, v, eid), graph.num_src(), graph.num_dst())
        adj, edge_map, _ = res
        # create a tmp message frame
        tmp_mfr = FrameRef(frame_like(var_ef.data._frame, len(eid)))
        var_out = var.FEAT_DICT(data=tmp_mfr)
        spmv.gen_v2e_spmv_schedule(graph=adj,
                                   mfunc=mfunc,
                                   src_frame=var_src_nf,
                                   dst_frame=var_dst_nf,
                                   edge_frame=var_ef,
                                   out=var_out,
                                   out_size=len(eid),
                                   edge_map=edge_map)
    else:
        # UDF send
        var_out = _gen_udf_send(var_src_nf, var_dst_nf, var_ef, var_u,
                                var_v, var_eid, mfunc,
                                canonical_etype=graph.canonical_etype)
    return var_out

def _build_idx_map(idx, nbits):
    """Build a map from the input ids to continuous ids that starts from zero.
    And the number of bits data type of each integer in the mapping uses will
    be nbits

    Examples
    --------
    >>> x = [1, 5, 3, 6]
    >>> o2n = map_to_continuous(x)
    >>> o2n
    [n/a, 0, n/a, 2, n/a, 1, 3]

    "n/a" will be filled with 0

    Parameters
    ----------
    x : Index
        The input ids, assumed to be unique.
    nbits: int
        Number of bits each integer in the mapping should use, can be 32 or 64

    Returns
    -------
    old_to_new : CtxCachedObject
        The mapping from old id to new id. It is a vector of length MAX(x).
        One can use advanced indexing to convert an old id tensor to a
        new id tensor: new_id = old_to_new[old_id]
    """
    x = idx.tousertensor()
    map_len = int(F.asnumpy(F.max(x, dim=0))) + 1
    old_to_new = F.full_1d(map_len, -1, dtype=F.int64, ctx=F.cpu())
    # Use out-place update due to tensorflow compatibility
    old_to_new = F.scatter_row(old_to_new, x, F.arange(0, len(x)))
    old_to_new = utils.to_nbits_int(old_to_new, nbits)
    old_to_new = F.zerocopy_to_dgl_ndarray(old_to_new)
    return utils.CtxCachedObject(lambda ctx: nd.array(old_to_new, ctx=ctx))

_init_api("dgl._deprecate.runtime.scheduler")
