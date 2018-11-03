"""For different schedulers"""

from ..base import ALL, DGLError, is_all
from .. import backend as F
from ..function import message as fmsg
from ..function import reducer as fred

def get_send_plan(graph, message_func):
    # TODO (lingfan): doc string
    pass

def get_recv_plan(graph, v, reduce_func):
    # TODO (lingfan): doc string
    pass

def get_update_all_plan(graph, message_func, reduce_func):
    # TODO (lingfan): doc string
    pass

def get_send_and_recv_plan(graph, edges, message_func, reduce_func):
    # TODO (lingfan): doc string

    pass

def get_push_plan(graph, u, message_func, reduce_func):
    # TODO (lingfan): doc string
    pass

def get_pull_plan(graph, v, message_func, reduce_func):
    # TODO (lingfan): doc string
    pass

# XXX: multi-edge?
def get_update_edge_plan(graph, eids):
    # TODO (lingfan): doc string
    pass

def generate_degree_bucketing_executors():
    pass

def generate_spmv_executors():
    pass

def _generate_graph_key(prefix, call_type):
    if call_type == "update_all":
        key = prefix + "all"
    elif call_type == "send_and_recv":
        key = prefix + "edges"
    elif call_type == "recv":
        key = prefix + "message"
    else:
        raise DGLError("Unsupported call type: %s" % (call_type))
    return key

def _prepare_adjmat(g, call_type, graph_store, mfunc, **kwargs):
    key = _generate_graph_key("adjmat_")
    if isinstance(mfunc, fmsg.SrcMulEdgeMessageFunction):
        key += "_" + mfunc.edge_field
    if graph_store.get(key, None):
        # key already exists
        return

def _build_adj_matrix(g, call_type, edges=None):
    if call_type == "update_all":
        mat = self.g._handle.in_edge_incidence_matrix()
        return mat

    # partial graph case
    if call_type != "send_and_recv":
        raise DGLError("Unsupported call type when build adjmat: %s" % call_type)

    u, v = edges
    new2old, old2new = utils.build_relabel_map(v)
    nnz = len(u)
    u = u.tousertensor()
    v = v.tousertensor()
    new_v = old2new[v]
    n = self.g.number_of_nodes()
    m = len(new2old)
    mat = utils.build_sparse_matrix(new_v, u, [m, n], nnz)
    return utils.CtxCachedObject(lambda ctx: F.to_context(mat, ctx))

def _build_incidence_matrix(g, call_type, v=None, edges=None):
    if call_type == "update_all":
        mat = self.g._graph.in_edge_incidence_matrix()
        return mat

    # partial graph case
    if call_type == "send_and_recv":
        v = edges[1]
        m = len(v)
        eid = F.arannge(m)
    elif call_type == "recv":
        _, v, eid = self._msg_graph.in_edges(v)
        m = len(eid)
    else:
        raise DGLError("Unsupported call type when build incidence matrix: %s" % call_type)

    new2old, old2new = utils.build_relabel_map(v)
    v = v.tousertensor()
    new_v = old2new[v]
    n = len(new2old)
    mat = utils.build_sparse_matrix(new_v, eid, [n, m], m)
    return utils.CtxCachedObject(lambda ctx: F.to_context(mat, ctx))

def _get_exec_plan(g, call_type, mfunc=None, rfunc=None, **kwargs):
    v2v_spmv = None
    e2v_spmv = None
    to_deg_bucket = None
    if rfunc is None:
        # send and update_edge case
        if not isinstance(mfunc, (list, tuple)):
            mfunc = [mfunc]
    elif mfunc is None:
        # recv case
        e2v_spmv, to_deg_bucket = _get_reduce_plan(rfunc)
    else:
        mfunc, v2v_spmv, e2v_spmv, to_deg_bucket = _get_multi_stage_plan(mfunc, rfunc)

    graph_store = {}

    # send
    if mfunc:
        if len(mfunc) == 1:
            mfunc = mfunc[0]
        else:
            mfunc = fmsg.BundledMessageFunction(mfunc)
        # TODO(lingfan): create send executor

    # fused spmv
    if v2v_spmv:
        if call_type == "send_and_recv":
            # build partial
            pass
        elif call_type == "update_all":
            # build full
        else:
            assert(0)

        # TODO(lingfan): build adjmat
        for mfn, rfn in v2v_spmv:
            # TODO(lingfan): create spmv executor using adjmat
            pass

    # incidence matrix spmv
    if e2v_spmv:
        if call_type == "send_and_recv":
        else:
            assert(0)
        # TODO(lingfan): build incidence mat
        for rfn in e2v_spmv:
            # TODO(lingfan): create spmv executor using incidence mat
            pass

    # degree bucketing
    if to_deg_bucket:
        if len(to_deg_bucket) == 1:
            rfunc = to_deg_bucket[0]
        else:
            rfunc = fred.BundledMessageFunction(to_deg_bucket)
        # TODO(lingfan): get degree bucketing schedule
        # TODO(lingfan): create degree bucketing executor

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
Generate exec plan for API that involves both message and reduce func
This function return four components:
    3. reduce func for Edge-to-Node space SPMV
    4. reduce func for degree bucketing
Note: each component is either a list or None
"""
def _get_reduce_plan(rfunc):
    builtin_list = _sanity_check_on_rfunc(rfunc)
    if builtin_list is None:
        # reduce is a single UDF
        # all messages needed bundle them together and we are done
        return None, [rfunc]

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

"""
Generate exec plan for API that involves both message and reduce func
This function return four components:
    1. Message func that needs to be materialized
    2. message and reduce pairs for Node-to-Node space SPMV
    3. reduce func for Edge-to-Node space SPMV
    4. reduce func for degree bucketing
Note: each component is either a list or None
"""
def _get_multi_stage_plan(mfunc, rfunc):
    if not isinstance(mfunc, (list, tuple)):
        mfunc = [mfunc]

    builtin_list = _sanity_check_on_rfunc(rfunc)
    if builtin_list is None:
        # reduce is a single UDF
        # all messages needed bundle them together and we are done
        return mfunc, None, None, [rfunc]

    # if scheduler reaches this point, then reduce function must all be builtin
    # now try to pair builtin reduce with message function

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

    return mfunc_to_materialize, v2v_spmv_pairs, e2v_spmv, deg_bucket

