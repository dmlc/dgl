"""For different schedulers"""
from __future__ import absolute_import

from ..base import ALL, DGLError, is_all
from .. import utils
from .. import backend as F
from ..function import message as fmsg
from ..function import reducer as fred
from .executor import *
from .frame import FrameRef

from .._fi.function import _init_api

# TODO(lingfan)
__all__ = []

# XXX: multi-edge?

def _get_edge_idx(g, u, v, eid):
    if is_all(u) and is_all(v) and eid is None:
        u, v, eid = g._graph.edges()
    elif eid is not None:
        eid = utils.toindex(eid)
        u, v, _ = g._graph.find_edges(eid)
    else:
        u = utils.toindex(u)
        v = utils.toindex(v)
        u, v, eid = g._graph.edge_ids(u, v)
    return u, v, eid

def get_send_schedule(graph, u, v, eid, message_func):
    # FIXME: need to check message frame first
    # TODO (lingfan): doc string
    u, v, eid = _get_edge_idx(graph, u, v, eid)
    if isinstance(message_func, (list, tuple)):
        message_func = fmsg.BundledMessageFunction(message_func)
    send_exe = _build_edge_executor(message_func, graph._node_frame, graph._edge_frame, graph._msg_frame, u, v, eid)
    graph._msg_graph.add_edges(u, v)

def get_recv_schedule(graph, v, reduce_func):
    # TODO (lingfan): doc string
    e2v_spmv, to_deg_bucket = _get_reduce_plan(rfunc)

    exec_plan = ExecutionPlan()
    reduce_execs = []
    message_frame = graph._msg_frame
    node_frame = graph._node_frame

    # incidence matrix spmv
    if e2v_spmv:
        graph_data = GraphData()
        spmv_out_frame = FrameRef()
        key = _build_incidence_matrix(graph, call_type, graph_data, **kwargs)
        for rfn in e2v_spmv:
            exe = _build_incidence_spmv_exec(graph_data, key, rfn, message_frame, spmv_out_frame)
            reduce_execs.append(exe)

    # degree bucketing
    if to_deg_bucket:
        exec_list = _build_degree_bucketing_executors(graph, call_type, node_frame, message_frame)
        recv_exes.extend(exec_list)

    exec_plan.add_stage(recv_exes)

def get_update_all_schedule(graph, message_func, reduce_func):
    # TODO (lingfan): doc string
    mfunc, v2v_spmv, e2v_spmv, to_deg_bucket = _get_multi_stage_plan(message_func, reduce_func)

    # adjmat kv store
    exec_plan = ExecutionPlan()
    node_frame = graph._node_frame
    edge_frame = graph._edge_frame
    graph_data = GraphData()
    message_frame = FrameRef()

    # send
    if mfunc:
        u, v, eid = _get_edge_info(ALL, ALL, None)
        send_exe = _build_edge_executor(graph, mfunc, node_frame, edge_frame, message_frame, u, v, eid):
        exec_plan.add_stage([send_exe])

    recv_exes = []
    spmv_out_frame = FrameRef()

    # fused spmv
    if v2v_spmv:
        key = _build_adj_matrix(graph, call_type, graph_data, edges=edges)
        for mfn, rfn in v2v_spmv:
            exe = _build_adj_spmv_exec(graph_data, key, mfn, rfn, node_frame, edge_frame, spmv_out_frame)
            recv_exes.append(exe)

    # incidence matrix spmv
    if e2v_spmv:
        key = _build_incidence_matrix(graph, call_type, graph_data, **kwargs)
        for rfn in e2v_spmv:
            exe = _build_incidence_spmv_exec(graph_data, key, rfn, message_frame, spmv_out_frame)
            reduce_execs.append(exe)

    # degree bucketing
    if to_deg_bucket:
        exec_list = _build_degree_bucketing_executors(graph, call_type, node_frame, message_frame)
        recv_exes.extend(exec_list)

    exec_plan.add_stage(recv_exes)
    return exec_plan

def get_send_and_recv_schedule(graph, u, v, eid, message_func, reduce_func):
    # TODO (lingfan): doc string
    u, v, eid = _get_edge_idx(graph, u, v, eid)

    mfunc, v2v_spmv, e2v_spmv, to_deg_bucket = _get_multi_stage_plan(message_func, reduce_func)

    # adjmat kv store
    exec_plan = ExecutionPlan()
    node_frame = graph._node_frame
    edge_frame = graph._edge_frame
    graph_data = GraphData()
    message_frame = FrameRef()

    # send
    if mfunc:
        send_exe = _build_edge_executor(graph, mfunc, node_frame, edge_frame, message_frame, u, v, eid):
        exec_plan.add_stage([send_exe])

    recv_exes = []
    spmv_out_frame = FrameRef()

    # fused spmv
    if v2v_spmv:
        key = _build_adj_matrix(graph, call_type, graph_data, edges=edges)
        for mfn, rfn in v2v_spmv:
            exe = _build_adj_spmv_exec(graph_data, key, mfn, rfn, node_frame, edge_frame, spmv_out_frame)
            recv_exes.append(exe)

    # incidence matrix spmv
    if e2v_spmv:
        key = _build_incidence_matrix(graph, call_type, graph_data, **kwargs)
        for rfn in e2v_spmv:
            exe = _build_incidence_spmv_exec(graph_data, key, rfn, message_frame, spmv_out_frame)
            reduce_execs.append(exe)

    # degree bucketing
    if to_deg_bucket:
        exec_list = _build_degree_bucketing_executors(graph, call_type, node_frame, message_frame)
        recv_exes.extend(exec_list)

    exec_plan.add_stage(recv_exes)
    return exec_plan

def get_push_schedule(graph, u, message_func, reduce_func):
    # TODO (lingfan): doc string
    pass

def get_pull_schedule(graph, v, message_func, reduce_func):
    # TODO (lingfan): doc string
    pass

def get_update_edge_schedule(graph, u, v, eid, apply_edge_func):
    # TODO (lingfan): doc string
    u, v, eid = _get_edge_idx(graph, u, v, eid)
    if isinstance(apply_edge_func, (list, tuple)):
        apply_edge_func = fmsg.BundledMessageFunction(apply_edge_func)
    exec_plan = ExecutionPlan()
    send_exe = _build_edge_executor(apply_edge_func, graph._node_frame, graph._edge_frame, graph._edge_frame, u, v, eid)
    return exec_plan

def _build_adj_spmv_exec(graph_data, key, mfunc, rfunc, node_frame, edge_frame, out_frame):
    if isinstance(mfunc, fmsg.SrcMulEdgeMessageFunction):
        return _build_spmv_executors(graph_data=graph_data,
                                     key=key,
                                     src_field=mfunc.src_field,
                                     src_frame=node_frame,
                                     out_field=rfunc.out_field,
                                     out_frame=out_frame,
                                     use_edge_feat=True,
                                     edge_field=mfunc.edge_field,
                                     edge_frame=edge_frame)
    else:
        return _build_spmv_executors(graph_data=graph_data,
                                     key=key,
                                     src_field=mfunc.src_field,
                                     src_frame=node_frame,
                                     out_field=rfunc.out_field,
                                     out_frame=out_frame,
                                     use_edge_feat=False)

def _build_incidence_spmv_exec(graph_data, key, rfunc, message_frame, out_frame):
    return _build_spmv_executors(graph_data=graph_data,
                                 key=key,
                                 src_field=rfunc.msg_field,
                                 src_frame=message_frame,
                                 out_field=rfunc.out_field,
                                 out_frame=out_frame,
                                 use_edge_feat=False)

def _build_spmv_executors(graph_data, key, src_field, src_frame, out_field, out_frame, use_edge_feat=False, edge_field=None, edge_frame=None):
    exe = SPMVExecutor(graph_data, use_edge_feat)
    exe.set_graph_key(key)
    exe.set_node_input(src_frame, fields=src_field)
    if use_edge_feat:
        exe.set_edge_input(edge_frame, fields=edge_field)
    exe.set_node_output(out_frame, fields=rfn.out_field)
    return exe

def _build_edge_executors(mfunc, node_frame, edge_frame, out_frame, u, v, eid):
    exe = EdgeExecutor(mfunc)
    exe.set_node_input(node_frame, u, v)
    exe.set_edge_input(edge_frame, eid)
    exe.set_edge_output(out_frame)
    return exe

def _build_degree_bucketing_executors(g, call_type, rfunc, node_frame, message_frame, v=None, edges=None):
    exec_list = []

    # get degree bucketing schedule
    if call_type == "send_and_recv":
        v = edges[1]
        uniq_v, degs, dsts, msg_ids = degree_bucketing_for_edges(v)
    elif call_type == "update_all":
        uniq_v, degs, dsts, msg_ids = degree_bucketing_for_graph(g._graph)
    elif call_type == "recv":
        unqi_v, degs, dsts, msg_ids = degree_bucketing_for_graph(g._msg_graph, v)
    else:
        raise DGLError("Unsupported call type for degree bucketing: %s" % call_type)

    # TODO(lingfan): check zero degree in C++

    # create UDF node executor
    for deg, vv, mid in zip(degs, dsts, msg_ids):
        exe = NodeExecutor(rfunc)
        exe.set_node_input(node_frame, ids=vv)
        exe.set_edge_input(message_frame, ids=mid)
        exe.set_node_output(FrameRef(), ids=vv)
        exec_list.append(exe)
    return exec_list

def _build_adj_matrix(g, call_type, graph_data, **kwargs):
    key = "adj_" + call_type
    if graph_data.get(key, None):
        # key already exists
        return key

    if call_type == "update_all":
        # full graph case
        mat = g._handle.in_edge_incidence_matrix()
    else:
        # partial graph case
        if call_type != "send_and_recv":
            raise DGLError("Unsupported call type when build adjmat: %s" % call_type)

        u, v = kwargs['edges']
        new2old, old2new = utils.build_relabel_map(v)
        nnz = len(u)
        u = u.tousertensor()
        v = v.tousertensor()
        new_v = old2new[v]
        n = g.number_of_nodes()
        m = len(new2old)
        mat = utils.build_sparse_matrix(new_v, u, [m, n], nnz)
        mat = utils.CtxCachedObject(lambda ctx: F.to_context(mat, ctx))

    graph_data[key] = mat
    return key

def _build_incidence_matrix(g, call_type, graph_data, **kwargs):
    key = "inc_" + call_type
    if graph_data.get(key, None):
        # key already exists
        return key

    if call_type == "update_all":
        # full graph case
        mat = g._graph.in_edge_incidence_matrix()
    else:
        # partial graph case
        if call_type == "send_and_recv":
            v = kwargs['edges'][1]
            m = len(v)
            eid = F.arannge(m)
        elif call_type == "recv":
            _, v, eid = g._msg_graph.in_edges(kwargs['v'])
            m = len(eid)
        else:
            raise DGLError("Unsupported call type when build incidence matrix: %s" % call_type)

        new2old, old2new = utils.build_relabel_map(v)
        v = v.tousertensor()
        new_v = old2new[v]
        n = len(new2old)
        mat = utils.build_sparse_matrix(new_v, eid, [n, m], m)
        mat = utils.CtxCachedObject(lambda ctx: F.to_context(mat, ctx))

    graph_data[key] = mat
    return key

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

    return unique_v, degs, dsts, msg_ids

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


"""Check if reduce functions are legal. The legal cases are:
1. a list of builtin reduce function
2. a single builtin/UDF reduce function
Mixed use of builtin and UDF reduce is not allowed.
This function returns a list of builtin reduce function or None if the rfunc is UDF
"""
def _sanity_check_on_rfunc(rfunc):
    if isinstance(rfunc, fred.ReduceFunction):
        # reduce is one single builtin
        # convert it to a list
        return [rfunc]
    elif isinstance(rfunc, (tuple, list)):
        # reduce is a list of func
        # check if all rfunc in the list are builtin
        for fn in rfunc:
            assert isinstance(fn, fred.ReduceFunction), \
                    "Mixed use of builtin reduce and UDF is forbidden"
        return rfunc
    else:
        return None

"""
Generate exec plan for API that involves only reduce func
This function return two components:
    1. reduce func for Edge-to-Node space SPMV
    2. reduce func for degree bucketing
"""
def _get_reduce_plan(rfunc):
    builtin_list = _sanity_check_on_rfunc(rfunc)
    if builtin_list is None:
        # reduce is a single UDF
        # all messages needed bundle them together and we are done
        return None, rfunc

    # rfn evaluated by e2v spmv whose mfn is materialized
    e2v_spmv = []
    # rfn evaluated by degree bucketing
    deg_bucket = []

    for rfn in builtin_list:
        if rfn.is_spmv_supported():
            mfunc_to_materialize.append(mfn)
            e2v_spmv.append(rfn)
        else:
            deg_bucket.append(rfn)

    n_deg_bucket = len(deg_bucket)
    if n_deg_bucket == 0:
        deg_bucket = None
    elif n_deg_bucket == 1:
        deg_bucket = deg_bucket[0]
    else:
        deg_bucket = fred.BundledReduceFunction(deg_bucket)

    return e2v_spmv, deg_bucket

"""
Generate exec plan for API that involves both message and reduce func
This function return four components:
    1. Message func that needs to be materialized
    2. message and reduce pairs for Node-to-Node space SPMV
    3. reduce func for Edge-to-Node space SPMV
    4. reduce func for degree bucketing
"""
def _get_multi_stage_plan(mfunc, rfunc):
    builtin_list = _sanity_check_on_rfunc(rfunc)
    if builtin_list is None:
        # reduce is a single UDF
        # all messages needed, bundle them together and we are done
        if isinstance(mfunc, (list, tuple)):
            mfunc = fmsg.BundledMessageFunction(mfunc)
        return mfunc, None, None, rfunc

    # if scheduler reaches this point, then reduce function must all be builtin
    # now try to pair builtin reduce with message function
    if not isinstance(mfunc, (list, tuple)):
        mfunc = [mfunc]

    # build msg out field to msg func dict
    out2mfunc = {}
    udf_mfunc = []
    for fn in mfunc:
        if isinstance(fn, fmsg.MessageFunction):
            out2mfunc[fn.out_field] = fn
        else:
            udf_mfunc.append(fn)

    # for each msg/red pair, decide whether to use spmv or degree-bucketing
    use_udf_msg = False

    # builtin msg that needs to be materialized
    mfunc_to_materialize = []
    # pairs of mfn, rfn that can be fused to v2v spmv
    v2v_spmv_pairs = []
    # rfn evaluated by e2v spmv whose mfn is materialized
    e2v_spmv = []
    # rfn evaluated by degree bucketing
    deg_bucket = []

    for rfn in builtin_list:
        mfn = out2mfunc.get(rfn.msg_field, None)
        if mfn is None:
            use_udf_msg = True # assume msg field generated by udf
        if rfn.is_spmv_supported():
            if mfn is not None:
                if mfn.is_spmv_supported():
                    v2v_spmv_pairs.append((mfn, rfn))
                else:
                    mfunc_to_materialize.append(mfn)
                    e2v_spmv.append(rfn)
            else:
                e2v_spmv_append(rfn)
        else:
            if mfn is not None:
                mfunc_to_materialize.append(mfn)
            deg_bucket.append(rfn)

    if use_udf_msg:
        if len(udf_mfunc) == 0:
            raise DGLError("Some reduce func needs message fields that are not generated by any message func.")
        else:
            mfunc_to_materialize.extend(udf_mfunc)

    n_mfunc = len(mfunc_to_materialize)
    if n_mfunc == 0:
        mfunc = None
    elif n_mfunc == 1:
        mfunc = mfunc_to_materizalize[0]
    else:
        mfunc = fmsg.BundledMessageFunction(mfunc_to_materialize)

    n_deg_bucket = len(deg_bucket)
    if n_deg_bucket == 0:
        deg_bucket = None
    elif n_deg_bucket == 1:
        deg_bucket = deg_bucket[0]
    else:
        deg_bucket = fred.BundledReduceFunction(deg_bucket)

    return mfunc, v2v_spmv_pairs, e2v_spmv, deg_bucket

_init_api("dgl.scheduler")
