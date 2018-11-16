"""For different schedulers"""
from __future__ import absolute_import

from ..base import ALL, DGLError, is_all
from .. import utils
from .. import backend as F
from ..function.base import BuiltinFunction, BundledFunction
from .executor import *

from .._ffi.function import _init_api

from . import ir
from . degree_bucketing import gen_degree_bucketing_schedule
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
        nf = ir.Var.FEAT_DICT(graph._node_frame)
        ef = ir.Var.FEAT_DICT(graph._edge_frame)
        mf = ir.Var.FEAT_DICT(graph._msg_frame)
        u = ir.Var.IDX(u)
        v = ir.Var.IDX(v)
        eid = ir.Var.IDX(eid)
        mfunc = ir.Var.FUNC(message_func)
        msg = gen_udf_send_schedule(nf, ef, u, v, eid, mfunc)
        # TODO: handle duplicate messages
        ir.APPEND_ROW(mf, msg)
        return prog

def get_recv_schedule(graph, v, reduce_func, apply_func):
    """get recv schedule

    Parameters
    ----------
    graph: DGLGraph
        The DGLGraph to use
    v : utils.Index
        Destination nodes
    reduce_func: callable or list of callable
        The reduce function
    apply_func: callable
        The apply node function

    Returns
    -------
    A list of executors for DGL Runtime
    """
    execs, out_repr = build_recv_executor(graph, v, reduce_func)
    if apply_func:
        apply_exec, out_repr = build_node_executor(graph, v, apply_func,
                                                   reduce_accum=out_repr)
        execs += apply_exec
    wb_exec = build_write_back_exec(graph, out_repr, v, "node")
    execs += wb_exec
    return execs

def _get_snr_schedule(graph, u, v, eid, message_func, reduce_func, apply_func):
    """get send and recv schedule

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
    reduce_func: callable or list of callable
        The reduce function
    apply_func: callable
        The apply node function

    Returns
    -------
    A list of executors for DGL Runtime
    """
    call_type = "send_and_recv"
    execs, out_repr = build_send_and_recv_executor(graph, call_type, u, v, eid,
                                                   message_func, reduce_func)
    # XXX: unique_v is calculated multiple times...
    unique_v, _ = F.sort_1d(F.unique(v.tousertensor()))
    if apply_func:
        apply_exec, out_repr = build_node_executor(graph, unique_v, apply_func,
                                                   reduce_accum=out_repr)
        execs += apply_exec
    wb_exec = build_write_back_exec(graph, out_repr, unique_v, "node")
    execs += wb_exec
    return execs

def get_snr_schedule(graph,
                     u,
                     v,
                     eid,
                     message_func,
                     reduce_func,
                     apply_func):
    call_type = 'send_and_recv'
    with ir.prog() as prog:
        nf = ir.Var.FEAT_DICT(graph._node_frame, name='nf')
        unique_v, _ = F.sort_1d(F.unique(v.tousertensor()))
        unique_v = ir.Var.IDX(unique_v, name='unique_v')
        reduced_feat = _x_get_snr_schedule(
                call_type, graph, u, v, eid, message_func, reduce_func)
        if apply_func:
            # To avoid writing reduced features back to node frame and reading
            # it again for apply phase. Instead, we first read the the node
            # features and "merge" it with the reduced features.
            v_nf = ir.READ_ROW(nf, v)
            v_nf = ir.UPDATE_DICT(v_nf, reduced_feat)
            # TODO: wrap afunc to the proper signature
            afunc = ir.Var.FUNC(apply_func)
            applied_feat = ir.CALL(afunc, [v_nf])
            final_feat = ir.UPDATE_DICT(applied_feat, reduced_feat)
        else:
            final_feat = reduced_feat
        ir.WRITE_ROW_(nf, unique_v, final_feat)
        return prog

def _x_get_snr_schedule(call_type, graph, u, v, eid,
        message_func, reduce_func):
    # arg vars
    nf = ir.Var.FEAT_DICT(graph._node_frame, name='nf')
    ef = ir.Var.FEAT_DICT(graph._edge_frame, name='ef')
    u = ir.Var.IDX(u, name='u')
    v = ir.Var.IDX(v, name='v')
    eid = ir.Var.IDX(eid, name='eid')

    # format the input functions
    mfunc = _standardize_func_usage(message_func)
    rfunc = _standardize_func_usage(reduce_func)
    mfunc_is_list = utils.is_iterable(mfunc)
    rfunc_is_list = utils.is_iterable(rfunc)

    out = ir.Var.FEAT_DICT() 

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
    mfunc = ir.Var.FUNC(mfunc)
    msg = gen_udf_send_schedule(nf, ef, u, v, eid, mfunc)

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
    rfunc = ir.Var.FUNC(rfunc)
    gen_degree_bucketing_schedule(call_type, nf, msg,
            rfunc, graph, v, out)
    return out

def gen_udf_send_schedule(nf, ef, u, v, eid, mfunc):
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
    call_type = "update_all"
    u, v, eid = ALL, ALL, ALL
    execs, out_repr = build_send_and_recv_executor(graph, call_type, u, v, eid,
                                                   message_func, reduce_func)
    if apply_func:
        apply_exec, out_repr = build_node_executor(graph, ALL, apply_func,
                                                   reduce_accum=out_repr)
        execs += apply_exec
    wb_exec = build_write_back_exec(graph, out_repr, ALL, "node")
    execs += wb_exec
    return execs

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
    # XXX: for now, use send_and_recv to implement pull
    u, v, eid = graph._graph.in_edges(v)
    if len(eid) == 0:
        return []
    return get_snr_schedule(graph, u, v, eid,
                            message_func, reduce_func, apply_func)

def build_node_executor(graph, v, func, reduce_accum=None):
    execs = []
    if reduce_accum:
        # if has reduce phase, apply should update the output of reduce
        out_repr = reduce_accum
    else:
        out_repr = {}
    if func:
        exe = NodeExecutor(func, graph, v, out_repr, reduce_accum)
        execs.append(exe)
    return execs, out_repr

def build_edge_executor(graph, u, v, eid, func):
    execs = []
    out_repr = {}
    if func:
        if is_all(eid):
            # if edges is ALL, look up source and destination nodes
            u, v, _ = graph._graph.edges()
        exe = EdgeExecutor(func, graph, u, v, eid, out_repr)
        execs.append(exe)
    return execs, out_repr

def build_recv_executor(graph, v, rfunc):
    """Build executors for recv"""
    call_type = "recv"
    rfunc = _standardize_func_usage(rfunc)

    recv_execs = []

    out_repr = {}

    if utils.is_iterable(rfunc):
        # build e2v spmv
        message_repr = dict(graph._msg_frame)
        u, v, eid = graph._msg_graph.in_edges(v)
        rfunc = _analyze_e2v_spmv(recv_execs, out_repr, rfunc,
                                  graph, call_type, v, eid, message_repr)

    # build degree bucketing
    _degree_bucket_exec(recv_execs, out_repr, rfunc,
                        graph, call_type, graph._msg_frame, v)

    return recv_execs, out_repr

def build_send_and_recv_executor(graph, call_type, u, v, eid, mfunc, rfunc):
    """Build executors for send_and_recv"""
    mfunc = _standardize_func_usage(mfunc)
    rfunc = _standardize_func_usage(rfunc)

    mfunc_is_list = utils.is_iterable(mfunc)
    rfunc_is_list = utils.is_iterable(rfunc)

    recv_execs = []

    out_repr = {}

    # both message and reduce are a list of builtin
    if mfunc_is_list and rfunc_is_list:
        # pair mfunc with rfunc
        pairs = _pair_reduce_with_message(mfunc, rfunc)

        # build v2v spmv
        mfunc, rfunc = _analyze_v2v_spmv(recv_execs, out_repr, pairs,
                                         graph, call_type, u, v, eid)

    # build send executor
    send_execs, message_repr = build_edge_executor(graph, u, v, eid, mfunc)

    if rfunc_is_list:
        # build e2v spmv
        rfunc = _analyze_e2v_spmv(recv_execs, out_repr, rfunc,
                                  graph, call_type, v, eid, message_repr)

    # build degree bucketing
    _degree_bucket_exec(recv_execs, out_repr, rfunc,
                        graph, call_type, message_repr, v)

    return send_execs + recv_execs, out_repr

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

def _analyze_v2v_spmv(exec_list, out_repr, pairs, graph, call_type, u, v, eid):
    """Analyze if SPMV from node space to node space can be applied

    Parameters
    ----------
    exec_list: list
        A list where generated executor will be put in
    out_repr: dict
        A dictionary to be binded to the executor to store the execution output
        This dictionary is empty until Runtime executes and materialize results
    pairs: list of tuple
        A list of tuples, each tuple is a message and reduce function pair
    graph: DGLGraph
        DGLGraph to use
    call_type: str
        Call_type of current graph API, could be "update_all" or "send_and_recv"
    u: utils.Index
        Source nodes
    v: utils.Index
        Destination nodes
    eid: utils.Index
        Edge ids

    Returns:
    mfunc_left: list
        A list of message functions that can't use v2v spmv. In other
        words, these message functions need to be materialized
    rfunc_left: list
        A list of reduce functions that can't use v2v spmv
    """
    mfunc_left = []
    rfunc_left = []

    # cache adjmat or adj_idx_shape
    adjmat = None
    adj_idx_shape = None

    # node/edge repr for spmv
    node_repr = graph.get_n_repr()
    edge_repr = graph.get_e_repr(eid)

    for mfn, rfn in pairs:
        # XXX: should pre-compile a look up table
        if mfn.is_spmv_supported(graph) and rfn.is_spmv_supported():
            if mfn.use_edge_feature:
                if adj_idx_shape is None:
                    adj_idx_shape = _build_adj_matrix(graph, call_type, u, v,
                                                      indices_and_shape=True)
                exe = _v2v_spmv_exec(mfunc=mfn,
                                     rfunc=rfn,
                                     adjmat=adj_idx_shape,
                                     node_repr=node_repr,
                                     out_repr=out_repr,
                                     use_edge_feat=True,
                                     edge_repr=edge_repr)
            else:
                if adjmat is None:
                    adjmat = _build_adj_matrix(graph, call_type, u, v)
                exe = _v2v_spmv_exec(mfunc=mfn,
                                     rfunc=rfn,
                                     adjmat=adjmat,
                                     node_repr=node_repr,
                                     out_repr=out_repr,
                                     use_edge_feat=False)
            exec_list.append(exe)
        else:
            mfunc_left.append(mfn)
            rfunc_left.append(rfn)
    return mfunc_left, rfunc_left

def _analyze_e2v_spmv(exec_list, out_repr, rfunc,
                      graph, call_type, v, eid, message_repr):
    """Analyze if SPMV from edge space to node space can be applied

    Parameters
    ----------
    exec_list: list
        A list where generated executor will be put in
    out_repr: dict
        A dictionary to be binded to the executor to store the execution output
        This dictionary is empty until Runtime executes and materialize results
    rfunc: list
        A list of reduce functions to be analyzed
    graph: DGLGraph
        DGLGraph to use
    call_type: str
        Call_type of current graph API, could be "update_all" or "send_and_recv"
    v: utils.Index
        Destination nodes
    eid: utils.Index
        Edge ids
    message_repr: dict
        Message representations (generated by message function)

    Returns:
    rfunc_left: list
        A list of reduce functions that can't use e2v spmv
    """
    if not rfunc:
        return []

    rfunc_left = []
    icd_mat = None
    for rfn in rfunc:
        if rfn.is_spmv_supported():
            if icd_mat is None:
                icd_mat = _build_incidence_matrix(graph, call_type, v, eid)
                exe = _e2v_spmv_exec(rfunc=rfn,
                                     adjmat=icd_mat,
                                     message_repr=message_repr,
                                     out_repr=out_repr)
                exec_list.append(exe)
        else:
            rfunc_left.append(rfn)
    return rfunc_left

def _v2v_spmv_exec(mfunc,
                   rfunc,
                   adjmat_creator,
                   node_feat,
                   out_feat,
                   edge_feat):
    """Build v2v spmv executor"""
    #return SPMVExecutor(A_creator=adjmat_creator,
                        #A_store=edge_feat,
                        #A_field=
    if use_edge_feat:
        index, shape = adjmat
        return SPMVExecutor(src_field=mfunc.src_field,
                            src_repr=node_repr,
                            out_field=rfunc.out_field,
                            out_repr=out_repr,
                            adjmat = index,
                            use_edge_feat=True,
                            edge_field=mfunc.edge_field,
                            edge_repr=edge_repr,
                            dense_shape=shape)
    else:
        return SPMVExecutor(src_field=mfunc.src_field,
                            src_repr=node_repr,
                            out_field=rfunc.out_field,
                            out_repr=out_repr,
                            adjmat = adjmat,
                            use_edge_feat=False)

def _e2v_spmv_exec(rfunc, adjmat, message_repr, out_repr):
    """Build e2v spmv executor"""
    return SPMVExecutor(src_field=rfunc.msg_field,
                        src_repr=message_repr,
                        out_field=rfunc.out_field,
                        out_repr=out_repr,
                        adjmat=adjmat,
                        use_edge_feat=False)

def build_write_back_exec(graph, new_repr, ids, target):
    return [WriteBackExecutor(graph, new_repr, ids, target)]

_init_api("dgl.runtime.scheduler")
