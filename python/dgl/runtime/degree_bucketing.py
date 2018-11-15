"""Module for degree bucketing schedulers"""
from __future__ import absolute_import

from .._ffi.function import _init_api
from .. import utils
from .. import backend as F
from ..immutable_graph_index import ImmutableGraphIndex
from . import ir

def gen_degree_bucketing_schedule(
        call_type,
        nf,
        msg,
        reduce_func,
        graph,
        v=None):
    """Create degree bucketing schedule

    Parameters
    ----------
    call_type: str
        Call_type of current graph API, could be 'update_all', 'send_and_recv', 'recv'
    nf : ir.Var
        The variable for node features.
    msg : ir.Var
        The variable for messages.
    rfunc: callable
        The UDF to reduce messages.
    graph: DGLGraph
        DGLGraph to use
    v: utils.Index
        Optional Receiving nodes

    Returns
    -------
    ir.Var
        The variable for the reduced node ids.
    ir.Var
        The variable for the reduced node features.
    """
    #if not rfunc:
    #    return

    #if utils.is_iterable(rfunc):
    #    rfunc = BundledFunction(rfunc)

    # Get degree bucketing schedule
    if isinstance(graph._graph, ImmutableGraphIndex):
        # NOTE: this is a workaround because immutable graph does not have (c++ support)
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
    # generate schedule
    unique_v, degs, dsts, msg_ids, zero_deg_nodes = buckets
    # loop over each bucket
    idx_list = []
    fd_list = []
    for deg, vb, mid in zip(degs, dsts, msg_ids):
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
    reduced_feat = ir.MERGE(idx_list, fd_list)
    # node id
    reduced_v, _ = F.sort_1d(unique_v)
    reduced_v = ir.Var.IDX(utils.toindex(reduced_v))
    return reduced_v, reduced_feat

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

def _degree_bucketing_for_graph(graph, v):
    """Return the bucketing by degree scheduling given graph index and optional
    dst nodes

    Parameters:
    -----------
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

def _process_buckets(buckets):
    """read bucketing auxiliary data

    Returns
    -------
    unique_v: utils.Index
        unqiue destination nodes
    degrees: numpy.ndarray
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

_init_api("dgl.runtime.degree_bucketing")
