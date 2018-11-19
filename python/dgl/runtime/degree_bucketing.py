"""Module for degree bucketing schedulers"""
from __future__ import absolute_import

from .._ffi.function import _init_api
from .. import utils
from .. import backend as F
from ..immutable_graph_index import ImmutableGraphIndex
from ..udf import EdgeBatch, NodeBatch

from . import ir
from .ir import var as var

def gen_degree_bucketing_schedule(
        call_type,
        nf,
        msg,
        reduce_udf,
        graph,
        edge_dst,
        out):
    """Create degree bucketing schedule.

    The messages will be divided by their receivers into buckets. Each bucket
    contains nodes that have the same in-degree. The reduce UDF will be applied
    on each bucket. The per-bucket result will be merged according to the
    *unique-ascending order* of the recv node ids. The order is important to
    be compatible with other reduce scheduler such as v2v_spmv.

    Parameters
    ----------
    call_type: str
        Call_type of current graph API, could be 'update_all', 'send_and_recv', 'recv'
    nf : var.Var
        The variable for node features.
    msg : var.Var
        The variable for messages.
    reduce_udf: callable
        The UDF to reduce messages.
    graph: DGLGraph
        DGLGraph to use
    edge_dst: var.Var
        The destination nodes of the edges. The number
        should be equal to the number of triggered edges.
    out : var.Var
        The variable for output feature dicts.
    """
    v = edge_dst.data
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
    unique_dst, degs, buckets, msg_ids, zero_deg_nodes = buckets
    # loop over each bucket
    idx_list = []
    fd_list = []
    for deg, vb, mid in zip(degs, buckets, msg_ids):
        # create per-bkt rfunc
        rfunc = _create_per_bkt_rfunc(graph, reduce_udf, deg, vb)
        # vars
        vb = var.IDX(vb)
        mid = var.IDX(mid)
        rfunc = var.FUNC(rfunc)
        # logic
        fdvb = ir.READ_ROW(nf, vb)
        fdmail = ir.READ_ROW(msg, mid)
        fdvb = ir.NODE_UDF(rfunc, fdvb, fdmail, ret=fdvb)  # reuse var
        # save for merge
        idx_list.append(vb)
        fd_list.append(fdvb)
    # zero-degree feats
    if zero_deg_nodes is not None:
        zero_deg_nodes = var.IDX(zero_deg_nodes)
        zero_deg_feat = ir.READ_ROW(nf, zero_deg_nodes)
        idx_list.append(zero_deg_nodes)
        fd_list.append(zero_deg_feat)
    # merge buckets according to the ascending order of the node ids.
    order = F.sort_1d(unique_dst.tousertensor())[0]
    order_var = var.IDX(utils.toindex(order))
    reduced_feat = ir.MERGE_ROW(order_var, idx_list, fd_list)
    ir.WRITE_DICT_(out, reduced_feat)

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
    zero_deg_nodes : utils.Index
        The zero-degree nodes
    """
    # get back results
    degs = utils.toindex(buckets(0))
    v = utils.toindex(buckets(1))
    # XXX: convert directly from ndarary to python list?
    v_section = buckets(2).asnumpy().tolist()
    msg_ids = utils.toindex(buckets(3))
    msg_section = buckets(4).asnumpy().tolist()

    # split buckets
    msg_ids = msg_ids.tousertensor()
    dsts = F.split(v.tousertensor(), v_section, 0)
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

    return v, degs, dsts, msg_ids, zero_deg_nodes

def _create_per_bkt_rfunc(graph, reduce_udf, deg, vb):
    def _rfunc_wrapper(node_data, mail_data):
        def _reshaped_getter(key):
            msg = mail_data[key]
            new_shape = (len(vb), deg) + F.shape(msg)[1:]
            return F.reshape(msg, new_shape)
        reshaped_mail_data = utils.LazyDict(_reshaped_getter, mail_data.keys())
        nb = NodeBatch(graph, vb, node_data, reshaped_mail_data)
        return reduce_udf(nb)
    return _rfunc_wrapper

_init_api("dgl.runtime.degree_bucketing")
