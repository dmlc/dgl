"""For different schedulers"""
from __future__ import absolute_import

from ..base import ALL, DGLError, is_all
from .. import utils
from .. import backend as F
from ..function.function import BuiltinFunction, BundledFunction
from .executor import SPMVExecutor, DegreeBucketingExecutor, EdgeExecutor, NodeExecutor
from .frame import FrameRef, Frame
from collections import Iterable

from .._fi.function import _init_api

# TODO(lingfan)
# 1. handle 0 degree in c++ (done)
# 2. adjmat index case (done)
# 3. double check on multi-edge in graph.py (need clean up)
# 4. remove graph store (done)
# 5. push and pull schedule
# 6. doc string
# 7. reorder arguments
# 8. message graph

# Attention:
# 1. recv v could become different after query in_edge
# 2. unique_v is calculated multiple times

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
    # TODO (lingfan): doc string
    call_type = "send"
    execs, out_repr = build_edge_executor(graph, call_type, u, v, eid, message_func)
    return execs, out_repr

def get_recv_schedule(graph, v, reduce_func, apply_func):
    # TODO (lingfan): doc string
    call_type = "recv"
    execs, out_repr = build_recv_executor(graph, call_type, v, reduce_func)
    if apply_func:
        apply_exec, out_repr = build_node_executor(graph, v, apply_func, reduce_accum=out_repr)
        execs += apply_exec
    return execs, out_repr, v

def get_snr_schedule(graph, u, v, eid, message_func, reduce_func, apply_func):
    # TODO (lingfan): doc string
    call_type = "send_and_recv"
    execs, out_repr = build_send_and_recv_executor(graph, call_type, u, v, eid, message_func, reduce_func)
    unique_v = F.unique(v.tousertensor())
    if apply_func:
        apply_exec, out_repr = build_node_executor(graph, unique_v, apply_func, reduce_accum=out_repr)
        execs += apply_exec
    return execs, out_repr, unique_v

def get_update_all_schedule(graph, message_func, reduce_func, apply_func):
    # TODO (lingfan): doc string
    call_type = "update_all"
    execs, out_repr = build_send_and_recv_executor(graph, call_type, ALL, ALL, ALL, message_func, reduce_func)
    if apply_func:
        apply_exec, out_repr = build_node_executor(graph, ALL, apply_func, reduce_accum=out_repr)
        execs += apply_exec
    return execs, out_repr, ALL

def get_apply_nodes_schedule(graph, v, apply_func):
    # TODO (lingfan): doc string
    return build_node_executor(graph, v, apply_func)

def get_apply_edges_schedule(graph, u, v, eid, apply_func):
    # TODO (lingfan): doc string
    return build_edge_executor(graph, u, v, eid, apply_func)

def get_push_schedule(graph, u, message_func, reduce_func):
    # TODO (lingfan): doc string
    pass

def get_pull_schedule(graph, v, message_func, reduce_func):
    # TODO (lingfan): doc string
    pass

def build_node_executor(graph, v, func, reduce_accum=None):
    execs = []
    if reduce_accum:
        out_repr = reduce_accum
    else:
        out_repr = {}
    _node_exec(execs, out_repr, func, graph, v)
    return execs, out_repr

def build_edge_executor(graph, u, v, eid, func):
    execs = []
    out_repr = {}
    _edge_exec(execs, out_repr, func, graph, u, v, eid)
    return execs, out_repr

def build_recv_executor(rfunc, graph, call_type, v, eid):
    rfunc = _standardize_func_usage(rfunc)

    recv_execs = []

    out_repr = {}

    if _is_iterable(rfunc):
        # build e2v spmv
        message_repr = dict(graph._msg_frame)
        rfunc = _analyze_e2v_spmv(recv_execs, out_repr, rfunc, graph, call_type, v, eid, message_repr)

    # build degree bucketing
    _degree_bucket_exec(recv_execs, out_repr, rfunc, graph, call_type, v, message_repr)

    return recv_execs, out_repr

def build_send_and_recv_executor(graph, call_type, u, v, eid, mfunc, rfunc):
    mfunc = _standardize_func_usage(mfunc)
    rfunc = _standardize_func_usage(rfunc)

    mfunc_is_list = _is_iterable(mfunc)
    rfunc_is_list = _is_iterable(rfunc)

    send_execs = []
    recv_execs = []

    message_repr = {}
    out_repr = {}

    if mfunc_is_list and rfunc_is_list:
        # pair mfunc with rfunc
        pairs = _pair_reduce_with_message(mfunc, rfunc)

        # build v2v spmv
        mfunc, rfunc = _analyze_v2v_spmv(recv_execs, out_repr, pairs, graph, call_type, u, v, eid)

    # build send executor
    _edge_exec(send_execs, message_repr, mfunc, graph, u, v, eid)

    if rfunc_is_list:
        # build e2v spmv
        rfunc = _analyze_e2v_spmv(recv_execs, out_repr, rfunc, graph, call_type, v, eid, message_repr)

    # build degree bucketing
    _degree_bucket_exec(recv_execs, out_repr, rfunc, graph, call_type, v, message_repr)

    return send_execs + recv_execs, out_repr

def _is_iterable(x):
    return isinstance(x, Iterable)

def _check_builtin_func_list(func_list):
    for fn in func_list:
        if not isinstance(fn, BuiltinFunction):
            raise DGLError("If specify multiple message/reduce functions, all of them must be builtin")

def _standardize_func_usage(func):
    if _is_iterable(func):
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
    mfunc = {fn.out_field: fn for fn in mfunc}
    func_list = []
    for rfn in rfunc:
        mfn = mfunc.get(rfn.msg_field, None)
        if mfn:
            func_list.append((mfn, rfn))
        else:
            raise DGLError("Cannot find message function that generates field %s." % rfn.msg_field)
    return func_list

def _build_adj_matrix(g, call_type, u, v, indices_and_shape=False):
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
            idx = utils.CtxCachedObject(lambda ctx: F.to_context(idx, ctx))
            return idx, (m, n)
        else:
            mat = utils.build_sparse_matrix(new_v, u, [m, n], nnz)
            return utils.CtxCachedObject(lambda ctx: F.to_context(mat, ctx))
    else:
        raise DGLError("Unsupported call type when build adjmat: %s" % call_type)

def _build_incidence_matrix(g, call_type, v, eid):
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
            raise DGLError("Unsupported call type when build incidence matrix: %s" % call_type)

        new2old, old2new = utils.build_relabel_map(v)
        v = v.tousertensor()
        eid = eid.tousertensor()
        new_v = old2new[v]
        n = len(new2old)
        mat = utils.build_sparse_matrix(new_v, eid, [n, m], m)
        return utils.CtxCachedObject(lambda ctx: F.to_context(mat, ctx))

def _analyze_v2v_spmv(exec_list, out_repr, pairs, graph, call_type, u, v, eid):
    mfunc_left = []
    rfunc_left = []
    adjmat = None
    adj_idx_shape = None
    node_repr = graph.get_n_repr(u)
    edge_repr = graph.get_e_repr(eid)
    for mfn, rfn in pairs:
        if mfn.is_spmv_supported() and rfn.is_spmv_supported():
            use_edge_feat = mfn.use_edge_feature()
            if use_edge_feat:
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
                                     use_edge_feat=False)
            exec_list.append(exe)
        else:
            mfunc_left.append(mfn)
            rfunc_left.append(rfn)
    return mfunc_left, rfunc_left

def _analyze_e2v_spmv(exec_list, out_repr, rfunc, graph, call_type, v, eid, message_repr):
    if not rfunc:
        return []

    rfunc_left = []
    incidence_mat = None
    for rfn in rfunc:
        if rfunc.is_spmv_supported():
            if incidence_mat is None:
                incidence_mat = _build_incidence_matrix(graph, call_type, v, eid)
                exe = _e2v_spmv_exec(rfunc=rfn,
                                     adjmat=incidence_mat,
                                     message_repr=message_repr,
                                     out_repr=out_repr)
                exec_list.append(exe)
        else:
            rfunc_left.append(rfn)
    return rfunc_left

def _v2v_spmv_exec(mfunc, rfunc, adjmat, node_repr, out_repr,
                   use_edge_feat=False, edge_repr=None):
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
    return SPMVExecutor(src_field=rfunc.msg_field,
                        src_repr=message_repr,
                        out_field=rfunc.out_field,
                        out_repr=out_repr,
                        adjmat=adjmat,
                        use_edge_feat=False)

def _node_exec(exec_list, out_repr, func, graph, u, reduce_accum):
    if func:
        if _is_iterable(func):
            func = BundledFunction(func)
        exec_list.append(NodeExecutor(func, graph, u, out_repr, reduce_accum))

def _edge_exec(exec_list, out_repr, func, graph, u, v, eid):
    if func:
        if _is_iterable(func):
            func = BundledFunction(func)
        exec_list.append(EdgeExecutor(func, graph, u, v, eid, out_repr))

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
    # TODO: convert directly from ndarary to python list?
    v_section = buckets(2).asnumpy().tolist()
    msg_ids = utils.toindex(buckets(3))
    msg_section = buckets(4).asnumpy().tolist()

    # split buckets
    unique_v = v.tousertensor()
    msg_ids = msg_ids.tousertensor()
    dsts = F.unpack(unique_v, v_section)
    msg_ids = F.unpack(msg_ids, msg_section)

    # convert to utils.Index
    unique_v = utils.toindex(unique_v)
    dsts = [utils.toindex(dst) for dst in dsts]
    msg_ids = [utils.toindex(msg_id) for msg_id in msg_ids]

    # handle zero deg
    degs = degs.tolist()
    if degs[-1] == 0:
        degs = degs[:-1]
        zero_deg_nodes = dsts[-1]
        dsts = dsts[-1]
    else:
        zero_deg_nodes = None

    return unique_v, degs, dsts, msg_ids, zero_deg_nodes

def _degree_bucketing_for_edges(dst):
    """Return the bucketing by degree scheduling for destination nodes of messages

    Parameters
    ----------
    dst: utils.Index
        destionation node for each message
    """

    buckets = _CAPI_DGLDegreeBucketingForEdges(dst.todgltensor())
    return _process_buckets(buckets)

def _degree_bucketing_for_graph(graph, v=ALL):
    """Return the bucketing by degree scheduling given graph index and option dst nodes

    Parameters:
    graph: GraphIndex
        DGLGraph Index (update all case) or message graph index (recv cases)
    v: utils.Index
        Destination nodes (recv cases)
    """

    if is_all(v):
        buckets = _CAPI_DGLDegreeBucketingForFullGraph(graph._handle)
    else:
        buckets = _CAPI_DGLDegreeBucketingForRecvNodes(graph._handle, v)
    return _process_buckets(buckets)

def _degree_bucket_exec(exec_list, out_repr, rfunc, g, call_type, message_repr, v=None):
    if not rfunc:
        return

    message_frame = FrameRef(Frame(message_repr))

    if _is_iterable(rfunc):
        rfunc = BundledFunction(rfunc)

    # get degree bucketing schedule
    if call_type == "send_and_recv":
        buckets = _degree_bucketing_for_edges(v)
    elif call_type == "update_all":
        buckets = _degree_bucketing_for_graph(g._graph)
    elif call_type == "recv":
        buckets = _degree_bucketing_for_graph(g._msg_graph, v)
    else:
        raise DGLError("Unsupported call type for degree bucketing: %s" % call_type)

    # TODO(lingfan): check zero degree in C++

    exe = DegreeBucketingExecutor(rfunc, g, rfunc, message_frame, out_repr, buckets)
    exec_list.append(exe)

_init_api("dgl.scheduler")
