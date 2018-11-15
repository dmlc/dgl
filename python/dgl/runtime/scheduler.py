"""For different schedulers"""
from __future__ import absolute_import

from ..base import ALL, DGLError, is_all
from .. import utils
from .. import backend as F
from ..function.base import BuiltinFunction, BundledFunction
from .executor import *
from ..immutable_graph_index import ImmutableGraphIndex

from .._ffi.function import _init_api

from . import ir

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
    send_exec, out_repr = build_edge_executor(graph, u, v, eid, message_func)
    wb_exec = build_write_back_exec(graph, out_repr, None, "message")
    return send_exec + wb_exec

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
    """
    env={nf, u, v, e, inc, unique_v, mfunc, rfunc
        vb1, vb2, ..., eb1, eb2, ...}
    FeatDict t1 = READ_ROW(nf, u)
    FeatDict t2 = READ_ROW(nf, v)
    FeatDict t3 = READ_ROW(ef, e)
    FeatDict t10 = CALL(mfunc, t1, t2, t3)

    FeatDict fdvb1 = READ_ROW(nf, vb1)
    FeatDict fdeb1 = READ_ROW(t10, eb1)
    FeatDict fdvb1 = CALL(rfunc, fdvb1, fdeb1)  # bkt1

    FeatDict fdvb2 = READ_ROW(nf, vb2)
    FeatDict fdeb2 = READ_ROW(t10, eb2)
    FeatDict fdvb2 = CALL(rfunc, fdvb2, fdeb2)  # bkt2

    FeatDict fdvb3 = READ_ROW(nf, vb3)
    FeatDict fdeb3 = READ_ROW(t10, eb3)
    FeatDict fdvb3 = CALL(rfunc, fdvb3, fdeb3)  # bkt3

    FeatDict t15 = MERGE(vb1, fdvb1,
                         vb2, fdvb2,
                         vb3, fdvb3,
                         unique_v)

    WRITE(nf, unique_v, t15)
    """
    prog = []
    # env
    nf = ir.Var.FEAT_DICT(graph._node_frame)
    ef = ir.Var.FEAT_DICT(graph._edge_frame)
    u = ir.Var.IDX(u)
    v = ir.Var.IDX(v)
    eid = ir.Var.IDX(eid)
    mfunc = ir.Var.FUNC(message_func)
    afunc = ir.Var.FUNC(apply_func)
    with ir.prog() as prog:
        # send
        fdsrc = ir.READ_ROW(nf, u)
        fddst = ir.READ_ROW(nf, v)
        fdedge = ir.READ_ROW(ef, eid)
        msg = ir.CALL(mfunc, [fdsrc, fdedge, fddst])

        # recv (degree bucketing)
        buckets = _degree_bucketing_for_edges(v.data)
        _, degs, dsts, msg_ids, zero_deg_nodes = buckets
        # loop over each bucket
        idx_list = []
        fd_list = []
        for deg, vb, mid in zip(degs, dsts, msg_ids):
            fdvb = ir.Var.FEAT_DICT()
            fdeb = ir.Var.FEAT_DICT()
            vb = ir.Var.IDX(vb)
            mid = ir.Var.IDX(mid)
            # TODO: wrap reshape into it
            rfunc = ir.Var.FUNC(reduce_func)
            fdvb = ir.READ_ROW(nf, vb)
            fdeb = ir.READ_ROW(msg, mid)
            fdvb = ir.CALL(rfunc, [fdvb, fdeb], ret=fdvb)  # reuse var
            # save for merge
            idx_list.append(vb)
            fd_list.append(fdvb)
        # merge buckets
        unique_v = ir.Var.IDX(utils.toindex(F.unique(v.data.tousertensor())))
        new_feat = ir.MERGE(idx_list, fd_list)
        ir.WRITE_ROW(nf, unique_v, new_feat)
        # TODO: apply func
        prog.pprint()

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

def _pair_reduce_with_message(mfunc, rfunc):
    """Look up message function for reduce function based on the message field
    """
    mfunc = {fn.out_field: fn for fn in mfunc}
    func_list = []
    for rfn in rfunc:
        mfn = mfunc.get(rfn.msg_field, None)
        if mfn:
            func_list.append((mfn, rfn))
        else:
            raise DGLError("Cannot find message function that \
                           generates field %s." % rfn.msg_field)
    return func_list

def _build_adj_matrix(g, call_type, u, v, indices_and_shape=False):
    """Build sparse adjacency matrix based on the call type
    If indices_and_shape is True, return packed indices and shape instead
    """
    if call_type == "update_all":
        # full graph case
        if indices_and_shape:
            return g._graph.adjacency_matrix_indices_and_shape()
        else:
            return g._graph.adjacency_matrix()
    elif call_type == "send_and_recv":
        # partial graph case
        new2old, old2new = utils.build_relabel_map(v)
        nnz = len(u)
        u = u.tousertensor()
        v = v.tousertensor()
        new_v = old2new[v]
        n = g.number_of_nodes()
        m = len(new2old)
        if indices_and_shape:
            idx = F.cat([F.unsqueeze(new_v, 0), F.unsqueeze(u, 0)], dim=0)
            cached_idx = utils.CtxCachedObject(lambda ctx: F.copy_to(idx, ctx))
            return cached_idx, (m, n)
        else:
            mat = utils.build_sparse_matrix(new_v, u, [m, n], nnz)
            return utils.CtxCachedObject(lambda ctx: F.copy_to(mat, ctx))
    else:
        raise DGLError("Unsupported call type when build adjmat: %s"
                       % call_type)

def _build_incidence_matrix(g, call_type, v, eid):
    """Build incidence matrix based on call type"""
    if call_type == "update_all":
        # full graph case
        return g._graph.in_edge_incidence_matrix()
    else:
        # partial graph case
        if call_type == "send_and_recv":
            m = len(v)
            eid = F.arannge(m)
        elif call_type == "recv":
            _, v, eid = g._msg_graph.in_edges(v)
            m = len(eid)
        else:
            raise DGLError("Unsupported call type when build incidence matrix:\
                           %s" % call_type)

        new2old, old2new = utils.build_relabel_map(v)
        v = v.tousertensor()
        eid = eid.tousertensor()
        new_v = old2new[v]
        n = len(new2old)
        mat = utils.build_sparse_matrix(new_v, eid, [n, m], m)
        return utils.CtxCachedObject(lambda ctx: F.to_context(mat, ctx))

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

def _process_buckets(buckets):
    """read bucketing auxiliary data

    Returns
    -------
    unique_v: utils.Index
        unqiue destination nodes
    degrees: utils.Index
        A list of degree for each bucket
    v_bkt: list of utils.Index
        A list of node id buckets, nodes in each bucket have the same degree
    msg_ids: list of utils.Index
        A list of message id buckets, each node in the ith node id bucket has
        degree[i] messages in the ith message id bucket
    """
    # get back results
    degs = utils.toindex(buckets(0))
    v = utils.toindex(buckets(1))
    # XXX: convert directly from ndarary to python list?
    v_section = buckets(2).asnumpy().tolist()
    msg_ids = utils.toindex(buckets(3))
    msg_section = buckets(4).asnumpy().tolist()

    # split buckets
    unique_v = v.tousertensor()
    msg_ids = msg_ids.tousertensor()
    dsts = F.split(unique_v, v_section, 0)
    msg_ids = F.split(msg_ids, msg_section, 0)

    # convert to utils.Index
    #unique_v = utils.toindex(unique_v)
    dsts = [utils.toindex(dst) for dst in dsts]
    msg_ids = [utils.toindex(msg_id) for msg_id in msg_ids]

    # handle zero deg
    degs = degs.tolist()
    if degs[-1] == 0:
        degs = degs[:-1]
        zero_deg_nodes = dsts[-1]
        dsts = dsts[:-1]
    else:
        zero_deg_nodes = None

    return unique_v, degs, dsts, msg_ids, zero_deg_nodes

def _degree_bucketing_schedule(mids, dsts, v):
    """Return the bucketing by degree scheduling for destination nodes of
    messages

    Parameters
    ----------
    mids: utils.Index
        edge id for each message
    dsts: utils.Index
        destination node for each message
    v: utils.Index
        all receiving nodes (for checking zero degree nodes)
    """

    buckets = _CAPI_DGLDegreeBucketing(mids.todgltensor(), dsts.todgltensor(),
                                       v.todgltensor())
    return _process_buckets(buckets)

def _degree_bucketing_for_edges(dsts):
    """Return the bucketing by degree scheduling for destination nodes of
    messages

    Parameters
    ----------
    dsts: utils.Index
        destination node for each message
    """

    buckets = _CAPI_DGLDegreeBucketingForEdges(dsts.todgltensor())
    return _process_buckets(buckets)

def _degree_bucketing_for_graph(graph, v=ALL):
    """Return the bucketing by degree scheduling given graph index and optional
    dst nodes

    Parameters:
    graph: GraphIndex
        DGLGraph Index (update all case) or message graph index (recv cases)
    v: utils.Index
        Destination nodes (recv cases)
    """

    if is_all(v):
        buckets = _CAPI_DGLDegreeBucketingForFullGraph(graph._handle)
    else:
        buckets = _CAPI_DGLDegreeBucketingForRecvNodes(graph._handle,
                                                       v.todgltensor())
    return _process_buckets(buckets)

def _degree_bucket_exec(exec_list, out_repr, rfunc,
                        graph, call_type, message_repr, v=None):
    """Create degree bucketing schedule

    Parameters
    ----------
    exec_list: list
        A list where generated executor will be put in
    out_repr: dict
        A dictionary to be binded to the executor to store the execution output
        This dictionary is empty until Runtime executes and materialize results
    rfunc: list
        A list of reduce functions to use degree bucketing
    graph: DGLGraph
        DGLGraph to use
    call_type: str
        Call_type of current graph API, could be "update_all" or "send_and_recv"
    message_repr: dict or Frame
        Message representations (generated by message function)
    v: utils.Index
        Optional Receiving nodes

    """
    if not rfunc:
        return

    if utils.is_iterable(rfunc):
        rfunc = BundledFunction(rfunc)

    # get degree bucketing schedule
    if isinstance(graph._graph, ImmutableGraphIndex):
        # immutable graph case (no c++ support)
        if call_type == "send_and_recv":
            mids = utils.toindex(range(0, len(v)))
            dsts = v
        elif call_type == "update_all":
            _, dsts, mids = graph._graph.edges()
            v = utils.toindex(range(graph._graph.number_of_nodes()))
        elif call_type == "recv":
            _, dsts, mids = graph._msg_graph.in_edges(v)
        else:
            raise DGLError("Unsupported call type for degree bucketing: %s"
                           % call_type)
        buckets = _degree_bucketing_schedule(mids, dsts, v)
    else:
        # mutable graph case
        if call_type == "send_and_recv":
            buckets = _degree_bucketing_for_edges(v)
        elif call_type == "update_all":
            buckets = _degree_bucketing_for_graph(graph._graph)
        elif call_type == "recv":
            buckets = _degree_bucketing_for_graph(graph._msg_graph, v)
        else:
            raise DGLError("Unsupported call type for degree bucketing: %s"
                           % call_type)

    exe = DegreeBucketingExecutor(graph, rfunc, message_repr, out_repr, buckets)
    exec_list.append(exe)

def build_write_back_exec(graph, new_repr, ids, target):
    return [WriteBackExecutor(graph, new_repr, ids, target)]


_init_api("dgl.runtime.scheduler")
