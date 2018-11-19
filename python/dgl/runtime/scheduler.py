"""For different schedulers"""
from __future__ import absolute_import

from .._ffi.function import _init_api
from ..base import ALL, DGLError, is_all
from .. import backend as F
from ..function.base import BuiltinFunction, BundledFunction
from .. import utils

from . import ir
from .ir import var as var
from . import degree_bucketing as db
from . import spmv

__all__ = [
            "get_send_schedule",
            "get_recv_schedule",
            "get_update_all_schedule",
            "get_snr_schedule",
            "get_apply_nodes_schedule",
            "get_apply_edges_schedule",
            "get_push_schedule",
            "get_pull_schedule"
          ]

def get_send_schedule(graph, u, v, eid, message_func):
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

    Returns
    -------
    A list of executors for DGL Runtime
    """
    with ir.prog() as prog:
        # vars
        nf = var.FEAT_DICT(graph._node_frame)
        ef = var.FEAT_DICT(graph._edge_frame)
        mf = var.FEAT_DICT(graph._msg_frame)
        u = var.IDX(u)
        v = var.IDX(v)
        eid = var.IDX(eid)
        mfunc = var.FUNC(message_func)
        msg = _gen_send(nf, ef, u, v, eid, mfunc)
        # TODO: handle duplicate messages
        ir.APPEND_ROW_(mf, msg)
        return prog

def get_recv_schedule(graph, nodes, reduce_func, apply_func):
    """get recv schedule

    Parameters
    ----------
    graph: DGLGraph
        The DGLGraph to use
    nodes : utils.Index
        Nodes to recv.
    reduce_func: callable or list of callable
        The reduce function
    apply_func: callable
        The apply node function

    Returns
    -------
    A list of executors for DGL Runtime
    """
    with ir.prog() as prog:
        nf = var.FEAT_DICT(graph._node_frame, name='nf')
        # sort and unique the argument
        nodes, _ = F.sort_1d(F.unique(nodes.tousertensor()))
        nodes = var.IDX(recv_nodes, name='recv_nodes')
        reduced_feat = _gen_reduce(graph, reduce_func, nodes)
        if apply_func:
            # To avoid writing reduced features back to node frame and reading
            # it again for apply phase. Instead, we first read the the node
            # features and "merge" it with the reduced features.
            v_nf = ir.READ_ROW(nf, nodes)
            v_nf = ir.UPDATE_DICT(v_nf, reduced_feat)
            # TODO: wrap afunc to the proper signature
            afunc = var.FUNC(apply_func)
            applied_feat = ir.CALL(afunc, [v_nf])
            final_feat = ir.UPDATE_DICT(applied_feat, reduced_feat)
        else:
            final_feat = reduced_feat
        ir.WRITE_ROW_(nf, nodes, final_feat)
        return prog

def _gen_reduce(graph, reduce_func, nodes):
    call_type = "recv"
    u, v, eid = graph._msg_graph.in_edges(nodes)
    msg = var.FEAT_DICT(graph._msg_frame, 'msg')
    nf = var.FEAT_DICT(graph._node_frame, 'nf')
    u = var.IDX(u)
    v = var.IDX(v)
    eid = var.IDX(eid)

    rfunc = _standardize_func_usage(reduce_func)
    rfunc_is_list = utils.is_iterable(rfunc)

    out = var.FEAT_DICT() 

    if rfunc_is_list:
        # UDF message + builtin reducer
        # analyze e2v spmv
        spmv_rfunc, rfunc = spmv.analyze_e2v_spmv(graph, rfunc)
        inc = spmv.build_inc_matrix(call_type, graph, eid.data, v.data)
        spmv.gen_e2v_spmv_schedule(inc, spmv_rfunc, msg, out)

        if len(rfunc) == 0:
            # All mfunc and rfunc has been processed.
            return out

        # convert the remaining rfunc to UDFs
        rfunc = BundledFunction(rfunc)

    # gen degree bucketing schedule for UDF recv
    rfunc = var.FUNC(rfunc)
    db.gen_degree_bucketing_schedule(call_type, nf, msg,
            rfunc, graph, v, out)
    return out

def get_snr_schedule(graph,
                     edge_tuples,
                     message_func,
                     reduce_func,
                     apply_func):
    call_type = 'send_and_recv'
    with ir.prog() as prog:
        nf = var.FEAT_DICT(graph._node_frame, name='nf')
        recv_nodes, _ = F.sort_1d(F.unique(edge_tuples[2].tousertensor()))
        recv_nodes = var.IDX(recv_nodes, name='recv_nodes')
        reduced_feat = _gen_send_reduce(call_type, graph,
                edge_tuples, message_func, reduce_func)
        if apply_func:
            # To avoid writing reduced features back to node frame and reading
            # it again for apply phase. Instead, we first read the the node
            # features and "merge" it with the reduced features.
            v_nf = ir.READ_ROW(nf, recv_nodes)
            v_nf = ir.UPDATE_DICT(v_nf, reduced_feat)
            # TODO: wrap afunc to the proper signature
            afunc = var.FUNC(apply_func)
            applied_feat = ir.CALL(afunc, [v_nf])
            final_feat = ir.UPDATE_DICT(applied_feat, reduced_feat)
        else:
            final_feat = reduced_feat
        ir.WRITE_ROW_(nf, recv_nodes, final_feat)
        return prog

def _gen_send_reduce(
        call_type,
        graph,
        edge_tuples,
        message_func,
        reduce_func):
    """Generate 

    This guarantees that the returned reduced features are batched
    in the *unique-ascending* order of the edge destination node ids.

    call_type : str
    graph : DGLGraph
    edge_tuples : (u, v, eid) tuple of utils.Index
    message_func : callable, list of builtins
    reduce_func : callable, list of builtins
    """
    # arg vars
    u, v, eid = edge_tuples
    nf = var.FEAT_DICT(graph._node_frame, name='nf')
    ef = var.FEAT_DICT(graph._edge_frame, name='ef')
    u = var.IDX(u, name='u')
    v = var.IDX(v, name='v')
    eid = var.IDX(eid, name='eid')

    # format the input functions
    mfunc = _standardize_func_usage(message_func)
    rfunc = _standardize_func_usage(reduce_func)
    mfunc_is_list = utils.is_iterable(mfunc)
    rfunc_is_list = utils.is_iterable(rfunc)

    out = var.FEAT_DICT() 

    if mfunc_is_list and rfunc_is_list:
        # builtin message + builtin reducer
        # analyze v2v spmv
        spmv_pairs, mfunc, rfunc = spmv.analyze_v2v_spmv(graph, mfunc, rfunc)
        adj = spmv.build_adj_matrix(call_type, graph, u.data, v.data)
        spmv.gen_v2v_spmv_schedule(adj, spmv_pairs, nf, ef, eid, out)

        if len(mfunc) == 0:
            # All mfunc and rfunc have been converted to v2v spmv.
            return out

        # converting remaining mfunc to UDFs
        mfunc = BundledFunction(mfunc)
    
    # generate UDF send schedule
    mfunc = var.FUNC(mfunc)
    msg = _gen_send(nf, ef, u, v, eid, mfunc)

    if rfunc_is_list:
        # UDF message + builtin reducer
        # analyze e2v spmv
        spmv_rfunc, rfunc = spmv.analyze_e2v_spmv(graph, rfunc)
        inc = spmv.build_inc_matrix(call_type, graph, eid.data, v.data)
        spmv.gen_e2v_spmv_schedule(inc, spmv_rfunc, msg, out)

        if len(rfunc) == 0:
            # All mfunc and rfunc has been processed.
            return out

        # convert the remaining rfunc to UDFs
        rfunc = BundledFunction(rfunc)

    # gen degree bucketing schedule for UDF recv
    rfunc = var.FUNC(rfunc)
    db.gen_degree_bucketing_schedule(call_type, nf, msg,
            rfunc, graph, v, out)
    return out

def _gen_send(nf, ef, u, v, eid, mfunc):
    fdsrc = ir.READ_ROW(nf, u)
    fddst = ir.READ_ROW(nf, v)
    fdedge = ir.READ_ROW(ef, eid)
    # TODO: wrap mfunc and change it to UDF signature.
    msg = ir.CALL(mfunc, [fdsrc, fdedge, fddst])
    return msg

def get_update_all_schedule(graph, message_func, reduce_func, apply_func):
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

    Returns
    -------
    A list of executors for DGL Runtime
    """
    call_type = 'update_all'
    edge_tuples = (ALL, ALL, ALL)
    with ir.prog() as prog:
        nf = var.FEAT_DICT(graph._node_frame, name='nf')
        recv_nodes = var.IDX(edge_tupes[2], name='recv_nodes')
        reduced_feat = _gen_send_reduce(call_type, graph,
                edge_tuples, message_func, reduce_func)
        if apply_func:
            # To avoid writing reduced features back to node frame and reading
            # it again for apply phase. Instead, we first read the the node
            # features and "merge" it with the reduced features.
            v_nf = ir.READ_ROW(nf, recv_nodes)
            v_nf = ir.UPDATE_DICT(v_nf, reduced_feat)
            # TODO: wrap afunc to the proper signature
            afunc = var.FUNC(apply_func)
            applied_feat = ir.CALL(afunc, [v_nf])
            final_feat = ir.UPDATE_DICT(applied_feat, reduced_feat)
        else:
            final_feat = reduced_feat
        ir.WRITE_ROW_(nf, recv_nodes, final_feat)
        return prog

def get_apply_nodes_schedule(graph, v, apply_func):
    """get apply nodes schedule

    Parameters
    ----------
    graph: DGLGraph
        The DGLGraph to use
    v : utils.Index
        Nodes to apply
    apply_func: callable
        The apply node function

    Returns
    -------
    A list of executors for DGL Runtime
    """
    assert False
    apply_exec, out_repr = build_node_executor(graph, v, apply_func)
    wb_exec = build_write_back_exec(graph, out_repr, v, "node")
    return apply_exec + wb_exec

def get_apply_edges_schedule(graph, u, v, eid, apply_func):
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

    Returns
    -------
    A list of executors for DGL Runtime
    """
    assert False
    apply_exec, out_repr = build_edge_executor(graph, u, v, eid, apply_func)
    wb_exec = build_write_back_exec(graph, out_repr, eid, "edge")
    return apply_exec + wb_exec

def get_push_schedule(graph, u, message_func, reduce_func, apply_func):
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

    Returns
    -------
    A list of executors for DGL Runtime
    """
    assert False
    # XXX: for now, use send_and_recv to implement push
    u, v, eid = graph._graph.out_edges(u)
    if len(eid) == 0:
        return []
    return get_snr_schedule(graph, u, v, eid,
                            message_func, reduce_func, apply_func)

def get_pull_schedule(graph, v, message_func, reduce_func, apply_func):
    """get pull schedule

    Parameters
    ----------
    graph: DGLGraph
        The DGLGraph to use
    v : utils.Index
        Destination nodes for pull
    message_func: callable or list of callable
        The message function
    reduce_func: callable or list of callable
        The reduce function
    apply_func: callable
        The apply node function

    Returns
    -------
    A list of executors for DGL Runtime
    """
    assert False
    # XXX: for now, use send_and_recv to implement pull
    u, v, eid = graph._graph.in_edges(v)
    if len(eid) == 0:
        return []
    return get_snr_schedule(graph, u, v, eid,
                            message_func, reduce_func, apply_func)

def _check_builtin_func_list(func_list):
    """Check whether func_list only contains builtin functions."""
    for fn in func_list:
        if not isinstance(fn, BuiltinFunction):
            raise DGLError("If specify multiple message/reduce functions, \
                           all of them must be builtin")

def _standardize_func_usage(func):
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
        # rfunc is a list of builtin
        _check_builtin_func_list(func)
        return func
    elif isinstance(func, BuiltinFunction):
        # func is one builtin-in
        return [func]
    else:
        # rfunc is one UDF
        return func

_init_api("dgl.runtime.scheduler")
