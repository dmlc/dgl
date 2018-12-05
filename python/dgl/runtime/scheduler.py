"""For different schedulers"""
from __future__ import absolute_import

from .. import utils
from .._ffi.function import _init_api
from ..base import ALL, DGLError, is_all
from .. import backend as F
from ..frame import frame_like, FrameRef
from ..function.base import BuiltinFunction, BundledFunction
from ..udf import EdgeBatch, NodeBatch

from . import ir
from .ir import var as var
from . import degree_bucketing as db
from . import spmv

__all__ = [
            "schedule_send",
            "schedule_recv",
            "schedule_update_all",
            "schedule_snr",
            "schedule_apply_nodes",
            "schedule_apply_edges",
            "schedule_push",
            "schedule_pull"
          ]

def schedule_send(graph, u, v, eid, message_func):
    """get send schedule

    Parameters
    ----------
    graph: DGLGraph
        The DGLGraph to use
    u : utils.Index
        Source nodes
    v : utils.Index
        Destination nodes
    eid : utils.Index
        Ids of sending edges
    message_func: callable or list of callable
        The message function
    """
    # TODO(minjie): support builtin message func
    message_func = _standardize_func_usage(message_func, 'message')
    # vars
    nf = var.FEAT_DICT(graph._node_frame)
    ef = var.FEAT_DICT(graph._edge_frame)
    mf = var.FEAT_DICT(graph._msg_frame)
    u = var.IDX(u)
    v = var.IDX(v)
    eid = var.IDX(eid)
    msg = _gen_send(graph, nf, ef, u, v, eid, message_func)
    # TODO: handle duplicate messages
    ir.APPEND_ROW_(mf, msg)

def schedule_recv(graph,
                  recv_nodes,
                  reduce_func,
                  apply_func,
                  inplace):
    """Schedule recv.

    Parameters
    ----------
    graph: DGLGraph
        The DGLGraph to use
    recv_nodes: utils.Index
        Nodes to recv.
    reduce_func: callable or list of callable
        The reduce function
    apply_func: callable
        The apply node function
    inplace: bool
        If True, the update will be done in place
    """
    src, dst, mid = graph._msg_graph.in_edges(recv_nodes)
    if len(mid) == 0:
        # All recv nodes are 0-degree nodes; downgrade to apply nodes.
        if apply_func is not None:
            schedule_apply_nodes(graph, recv_nodes, apply_func, inplace)
    else:
        var_nf = var.FEAT_DICT(graph._node_frame, name='nf')
        # sort and unique the argument
        recv_nodes, _ = F.sort_1d(F.unique(recv_nodes.tousertensor()))
        recv_nodes = utils.toindex(recv_nodes)
        var_recv_nodes = var.IDX(recv_nodes, name='recv_nodes')
        # reduce
        reduced_feat = _gen_reduce(graph, reduce_func, (src, dst, mid), recv_nodes)
        # apply
        final_feat = _apply_with_accum(graph, var_recv_nodes, var_nf, reduced_feat, apply_func)
        if inplace:
            ir.WRITE_ROW_INPLACE_(var_nf, var_recv_nodes, final_feat)
        else:
            ir.WRITE_ROW_(var_nf, var_recv_nodes, final_feat)

def schedule_snr(graph,
                 edge_tuples,
                 message_func,
                 reduce_func,
                 apply_func,
                 inplace):
    """Schedule send_and_recv.

    Parameters
    ----------
    graph: DGLGraph
        The DGLGraph to use
    edge_tuple: tuple
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
    """
    call_type = 'send_and_recv'
    u, v, eid = edge_tuples
    recv_nodes, _ = F.sort_1d(F.unique(v.tousertensor()))
    recv_nodes = utils.toindex(recv_nodes)
    # create vars
    var_nf = var.FEAT_DICT(graph._node_frame, name='nf')
    var_u = var.IDX(u)
    var_v = var.IDX(v)
    var_eid = var.IDX(eid)
    var_recv_nodes = var.IDX(recv_nodes, name='recv_nodes')
    # generate send and reduce schedule
    uv_getter = lambda : (var_u, var_v)
    adj_creator = lambda : spmv.build_adj_matrix_uv(graph, (u, v), recv_nodes)
    inc_creator = lambda : spmv.build_inc_matrix_dst(v, recv_nodes)
    reduced_feat = _gen_send_reduce(
            graph, message_func, reduce_func,
            var_eid, var_recv_nodes,
            uv_getter, adj_creator, inc_creator)
    # generate apply schedule
    final_feat = _apply_with_accum(graph, var_recv_nodes, var_nf, reduced_feat, apply_func)
    if inplace:
        ir.WRITE_ROW_INPLACE_(var_nf, var_recv_nodes, final_feat)
    else:
        ir.WRITE_ROW_(var_nf, var_recv_nodes, final_feat)

def schedule_update_all(graph,
                        message_func,
                        reduce_func,
                        apply_func):
    """get send and recv schedule

    Parameters
    ----------
    graph: DGLGraph
        The DGLGraph to use
    message_func: callable or list of callable
        The message function
    reduce_func: callable or list of callable
        The reduce function
    apply_func: callable
        The apply node function
    """
    if graph.number_of_edges() == 0:
        # All the nodes are zero degree; downgrade to apply nodes
        if apply_func is not None:
            nodes = utils.toindex(slice(0, graph.number_of_nodes()))
            schedule_apply_nodes(graph, nodes, apply_func, inplace=False)
    else:
        call_type = 'update_all'
        eid = utils.toindex(slice(0, graph.number_of_edges()))  # shortcut for ALL
        recv_nodes = utils.toindex(slice(0, graph.number_of_nodes()))  # shortcut for ALL
        # create vars
        var_nf = var.FEAT_DICT(graph._node_frame, name='nf')
        var_recv_nodes = var.IDX(recv_nodes, name='recv_nodes')
        var_eid = var.IDX(eid)
        # generate send + reduce
        def uv_getter():
            src, dst, _ = graph._graph.edges()
            return var.IDX(src), var.IDX(dst)
        adj_creator = lambda : spmv.build_adj_matrix_graph(graph)
        inc_creator = lambda : spmv.build_inc_matrix_graph(graph)
        reduced_feat = _gen_send_reduce(
                graph, message_func, reduce_func,
                var_eid, var_recv_nodes,
                uv_getter, adj_creator, inc_creator)
        # generate optional apply
        final_feat = _apply_with_accum(graph, var_recv_nodes, var_nf, reduced_feat, apply_func)
        ir.WRITE_DICT_(var_nf, final_feat)

def schedule_apply_nodes(graph,
                         v,
                         apply_func,
                         inplace):
    """get apply nodes schedule

    Parameters
    ----------
    graph: DGLGraph
        The DGLGraph to use
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
    var_nf = var.FEAT_DICT(graph._node_frame, name='nf')
    var_v = var.IDX(v)
    v_nf = ir.READ_ROW(var_nf, var_v)
    def _afunc_wrapper(node_data):
        nb = NodeBatch(graph, v, node_data)
        return apply_func(nb)
    afunc = var.FUNC(_afunc_wrapper)
    applied_feat = ir.NODE_UDF(afunc, v_nf)
    if inplace:
        ir.WRITE_ROW_INPLACE_(var_nf, var_v, applied_feat)
    else:
        ir.WRITE_ROW_(var_nf, var_v, applied_feat)

def schedule_apply_edges(graph,
                         u, v, eid,
                         apply_func,
                         inplace):
    """get apply edges schedule

    Parameters
    ----------
    graph: DGLGraph
        The DGLGraph to use
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
    var_nf = var.FEAT_DICT(graph._node_frame, name='nf')
    var_ef = var.FEAT_DICT(graph._edge_frame, name='ef')
    var_u = var.IDX(u)
    var_v = var.IDX(v)
    var_eid = var.IDX(eid)
    # schedule apply edges
    fdsrc = ir.READ_ROW(var_nf, var_u)
    fddst = ir.READ_ROW(var_nf, var_v)
    fdedge = ir.READ_ROW(var_ef, var_eid)
    def _efunc_wrapper(src_data, edge_data, dst_data):
        eb = EdgeBatch(graph, (u, v, eid),
                src_data, edge_data, dst_data)
        return apply_func(eb)
    _efunc = var.FUNC(_efunc_wrapper)
    new_fdedge = ir.EDGE_UDF(_efunc, fdsrc, fdedge, fddst)
    if inplace:
        ir.WRITE_ROW_INPLACE_(var_ef, var_eid, new_fdedge)
    else:
        ir.WRITE_ROW_(var_ef, var_eid, new_fdedge)

def schedule_push(graph,
                  u,
                  message_func,
                  reduce_func,
                  apply_func,
                  inplace):
    """get push schedule

    Parameters
    ----------
    graph: DGLGraph
        The DGLGraph to use
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
    """
    u, v, eid = graph._graph.out_edges(u)
    if len(eid) == 0:
        # All the pushing nodes have no out edges. No computation is scheduled.
        return
    schedule_snr(graph, (u, v, eid),
                 message_func, reduce_func, apply_func, inplace)

def schedule_pull(graph,
                  pull_nodes,
                  message_func,
                  reduce_func,
                  apply_func,
                  inplace):
    """get pull schedule

    Parameters
    ----------
    graph: DGLGraph
        The DGLGraph to use
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
    """
    # TODO(minjie): `in_edges` can be omitted if message and reduce func pairs
    #   can be specialized to SPMV. This needs support for creating adjmat
    #   directly from pull node frontier.
    u, v, eid = graph._graph.in_edges(pull_nodes)
    if len(eid) == 0:
        # All the nodes are 0deg; downgrades to apply.
        if apply_func is not None:
            schedule_apply_nodes(graph, pull_nodes, apply_func, inplace)
    else:
        call_type = 'send_and_recv'
        pull_nodes, _ = F.sort_1d(F.unique(pull_nodes.tousertensor()))
        pull_nodes = utils.toindex(pull_nodes)
        # create vars
        var_nf = var.FEAT_DICT(graph._node_frame, name='nf')
        var_pull_nodes = var.IDX(pull_nodes, name='pull_nodes')
        var_u = var.IDX(u)
        var_v = var.IDX(v)
        var_eid = var.IDX(eid)
        # generate send and reduce schedule
        uv_getter = lambda : (var_u, var_v)
        adj_creator = lambda : spmv.build_adj_matrix_uv(graph, (u, v), pull_nodes)
        inc_creator = lambda : spmv.build_inc_matrix_dst(v, pull_nodes)
        reduced_feat = _gen_send_reduce(
                graph, message_func, reduce_func,
                var_eid, var_pull_nodes,
                uv_getter, adj_creator, inc_creator)
        # generate optional apply
        final_feat = _apply_with_accum(graph, var_pull_nodes, var_nf, reduced_feat, apply_func)
        if inplace:
            ir.WRITE_ROW_INPLACE_(var_nf, var_pull_nodes, final_feat)
        else:
            ir.WRITE_ROW_(var_nf, var_pull_nodes, final_feat)

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

    This function checks if func meets the requirement, and merges last two cases
    by putting builtin function in case 2 into a list

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

def _apply_with_accum(graph, var_nodes, var_nf, var_accum, apply_func):
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
    """
    if apply_func:
        # To avoid writing reduced features back to node frame and reading
        # it again for apply phase. Instead, we first read the the node
        # features and "merge" it with the reduced features.
        v_nf = ir.READ_ROW(var_nf, var_nodes)
        v_nf = ir.UPDATE_DICT(v_nf, var_accum)
        def _afunc_wrapper(node_data):
            nb = NodeBatch(graph, var_nodes.data, node_data)
            return apply_func(nb)
        afunc = var.FUNC(_afunc_wrapper)
        applied_feat = ir.NODE_UDF(afunc, v_nf)
        final_feat = ir.UPDATE_DICT(var_accum, applied_feat)
    else:
        final_feat = var_accum
    return final_feat

def _gen_reduce(graph, reduce_func, edge_tuples, recv_nodes):
    """
    graph : DGLGraph
    reduce_func : callable
    edge_tuples : tuple of utils.Index
    recv_nodes : utils.Index
    """
    call_type = "recv"
    _, dst, mid = edge_tuples
    rfunc = _standardize_func_usage(reduce_func, 'reduce')
    rfunc_is_list = utils.is_iterable(rfunc)
    # Create a tmp frame to hold the feature data.
    # The frame has the same size and schemes of the
    # node frame.
    # TODO(minjie): should replace this with an IR call to make the program stateless.
    tmpframe = FrameRef(frame_like(graph._node_frame._frame, len(recv_nodes)))

    # vars
    msg = var.FEAT_DICT(graph._msg_frame, 'msg')
    nf = var.FEAT_DICT(graph._node_frame, 'nf')
    out = var.FEAT_DICT(data=tmpframe)

    if rfunc_is_list:
        # UDF message + builtin reducer
        # analyze e2v spmv
        spmv_rfunc, rfunc = spmv.analyze_e2v_spmv(graph, rfunc)
        # FIXME: refactor this when fixing the multi-recv bug
        inc = spmv.build_inc_matrix_eid(graph._msg_frame.num_rows, mid, dst, recv_nodes)
        spmv.gen_e2v_spmv_schedule(inc, spmv_rfunc, msg, out)

        if len(rfunc) == 0:
            # All mfunc and rfunc has been processed.
            return out

        # convert the remaining rfunc to UDFs
        rfunc = BundledFunction(rfunc)

    # gen degree bucketing schedule for UDF recv
    db.gen_degree_bucketing_schedule(graph, rfunc, mid, dst,
            recv_nodes, nf, msg, out)
    return out

def _gen_send_reduce(
        graph,
        message_func,
        reduce_func,
        var_send_edges,
        var_reduce_nodes,
        uv_getter,
        adj_creator,
        inc_creator):
    """Generate send and reduce schedule.

    This guarantees that the returned reduced features are batched
    in the *unique-ascending* order of the edge destination node ids.

    Parameters
    ----------
    graph : DGLGraph
        The graph
    message_func : callable, list of builtins
        The message func(s).
    reduce_func : callable, list of builtins
        The reduce func(s).
    var_send_edges : var.IDX
        The edges (ids) to perform send.
    var_reduce_nodes : var.IDX
        The nodes to perform reduce. This should include unique(v) + 0deg nodes.
    uv_getter : callable
        A function that returns a pair of var.IDX (u, v) for the triggered edges.
    adj_creator : callable
        A function that returns the adjmat and the shuffle index.
    inc_creator : callable
        A function that returns the incmat and the shuffle index.

    Returns
    -------
    var.FEAT_DICT
        The reduced feature dict.
    """
    # NOTE: currently, this function requires all var.IDX to contain concrete data.
    reduce_nodes = var_reduce_nodes.data

    # arg vars
    var_nf = var.FEAT_DICT(graph._node_frame, name='nf')
    var_ef = var.FEAT_DICT(graph._edge_frame, name='ef')
    var_eid = var_send_edges

    # format the input functions
    mfunc = _standardize_func_usage(message_func, 'message')
    rfunc = _standardize_func_usage(reduce_func, 'reduce')
    mfunc_is_list = utils.is_iterable(mfunc)
    rfunc_is_list = utils.is_iterable(rfunc)

    # Create a tmp frame to hold the feature data.
    # The frame has the same size and schemes of the
    # node frame.
    # TODO(minjie): should replace this with an IR call to make the program stateless.
    tmpframe = FrameRef(frame_like(graph._node_frame._frame, len(reduce_nodes)))
    var_out = var.FEAT_DICT(data=tmpframe)

    if mfunc_is_list and rfunc_is_list:
        # builtin message + builtin reducer
        # analyze v2v spmv
        spmv_pairs, mfunc, rfunc = spmv.analyze_v2v_spmv(graph, mfunc, rfunc)
        adj = adj_creator()
        spmv.gen_v2v_spmv_schedule(adj, spmv_pairs, var_nf, var_ef, var_eid, var_out)

        if len(mfunc) == 0:
            # All mfunc and rfunc have been converted to v2v spmv.
            return var_out

    if mfunc_is_list:
        # Two cases:
        #  - mfunc is builtin while rfunc is UDF.
        #  - mfunc and rfunc are both builtin but some combinations
        #    fall through from the v2v spmv analysis.
        # In both cases, convert the mfunc to UDF.
        mfunc = BundledFunction(mfunc)

    # generate UDF send schedule
    var_u, var_v = uv_getter()
    var_mf = _gen_send(graph, var_nf, var_ef, var_u, var_v, var_eid, mfunc)

    if rfunc_is_list:
        # UDF message + builtin reducer
        # analyze e2v spmv
        spmv_rfunc, rfunc = spmv.analyze_e2v_spmv(graph, rfunc)
        inc = inc_creator()
        spmv.gen_e2v_spmv_schedule(inc, spmv_rfunc, var_mf, var_out)

        if len(rfunc) == 0:
            # All mfunc and rfunc has been processed.
            return var_out

        # convert the remaining rfunc to UDFs
        rfunc = BundledFunction(rfunc)

    # gen degree bucketing schedule for UDF recv
    mid = utils.toindex(slice(0, len(var_v.data)))  # message id is from 0~|dst|
    db.gen_degree_bucketing_schedule(graph, rfunc,
            mid, var_v.data, reduce_nodes,
            var_nf, var_mf, var_out)
    return var_out

def _gen_send(graph, nf, ef, u, v, eid, mfunc):
    fdsrc = ir.READ_ROW(nf, u)
    fddst = ir.READ_ROW(nf, v)
    fdedge = ir.READ_ROW(ef, eid)
    def _mfunc_wrapper(src_data, edge_data, dst_data):
        eb = EdgeBatch(graph, (u.data, v.data, eid.data),
                src_data, edge_data, dst_data)
        return mfunc(eb)
    _mfunc_wrapper = var.FUNC(_mfunc_wrapper)
    msg = ir.EDGE_UDF(_mfunc_wrapper, fdsrc, fdedge, fddst)
    return msg

_init_api("dgl.runtime.scheduler")
