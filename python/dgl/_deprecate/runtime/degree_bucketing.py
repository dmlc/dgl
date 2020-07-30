"""Module for degree bucketing schedulers."""
from __future__ import absolute_import
from functools import partial

from ..._ffi.function import _init_api
from ... import backend as F
from ..udf import NodeBatch, EdgeBatch
from ... import utils

from . import ir
from .ir import var

def gen_degree_bucketing_schedule(
        reduce_udf,
        message_ids,
        dst_nodes,
        recv_nodes,
        var_nf,
        var_mf,
        var_out,
        ntype=None):
    """Create degree bucketing schedule.

    The messages will be divided by their receivers into buckets. Each bucket
    contains nodes that have the same in-degree. The reduce UDF will be applied
    on each bucket. The per-bucket result will be merged according to the
    *unique-ascending order* of the recv node ids. The order is important to
    be compatible with other reduce scheduler such as v2v_spmv.

    Parameters
    ----------
    reduce_udf : callable
        The UDF to reduce messages.
    message_ids : utils.Index
        The variable for message ids.
        Invariant: len(message_ids) == len(dst_nodes)
    dst_nodes : utils.Index
        The variable for dst node of each message.
        Invariant: len(message_ids) == len(dst_nodes)
    recv_nodes : utils.Index
        The unique nodes that perform recv.
        Invariant: recv_nodes = sort(unique(dst_nodes))
    var_nf : var.FEAT_DICT
        The variable for node feature frame.
    var_mf : var.FEAT_DICT
        The variable for message frame.
    var_out : var.FEAT_DICT
        The variable for output feature dicts.
    ntype : str, optional
        The node type, if running on a heterograph.
        If None, assuming it's running on a homogeneous graph.
    """
    buckets = _degree_bucketing_schedule(message_ids, dst_nodes, recv_nodes)
    # generate schedule
    _, degs, buckets, msg_ids, zero_deg_nodes = buckets
    # loop over each bucket
    idx_list = []
    fd_list = []
    for deg, vbkt, mid in zip(degs, buckets, msg_ids):
        # create per-bkt rfunc
        rfunc = _create_per_bkt_rfunc(reduce_udf, deg, vbkt, ntype=ntype)
        # vars
        vbkt = var.IDX(vbkt)
        mid = var.IDX(mid)
        rfunc = var.FUNC(rfunc)
        # recv on each bucket
        fdvb = ir.READ_ROW(var_nf, vbkt)
        fdmail = ir.READ_ROW(var_mf, mid)
        fdvb = ir.NODE_UDF(rfunc, fdvb, fdmail, ret=fdvb)  # reuse var
        # save for merge
        idx_list.append(vbkt)
        fd_list.append(fdvb)
    if zero_deg_nodes is not None:
        # NOTE: there must be at least one non-zero-deg node; otherwise,
        #   degree bucketing should not be called.
        var_0deg = var.IDX(zero_deg_nodes)
        zero_feat = ir.NEW_DICT(var_out, var_0deg, fd_list[0])
        idx_list.append(var_0deg)
        fd_list.append(zero_feat)
    # merge buckets according to the ascending order of the node ids.
    all_idx = F.cat([idx.data.tousertensor() for idx in idx_list], dim=0)
    _, order = F.sort_1d(all_idx)
    var_order = var.IDX(utils.toindex(order))
    reduced_feat = ir.MERGE_ROW(var_order, fd_list)
    ir.WRITE_DICT_(var_out, reduced_feat)

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
    return _process_node_buckets(buckets)

def _process_node_buckets(buckets):
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
    dtype = buckets(0).dtype
    degs = utils.toindex(buckets(0), dtype)
    v = utils.toindex(buckets(1), dtype)
    # XXX: convert directly from ndarary to python list?
    v_section = buckets(2).asnumpy().tolist()
    msg_ids = utils.toindex(buckets(3), dtype)
    msg_section = buckets(4).asnumpy().tolist()

    # split buckets
    msg_ids = msg_ids.tousertensor()
    dsts = F.split(v.tousertensor(), v_section, 0)
    msg_ids = F.split(msg_ids, msg_section, 0)

    # convert to utils.Index
    dsts = [utils.toindex(dst, dtype) for dst in dsts]
    msg_ids = [utils.toindex(msg_id, dtype) for msg_id in msg_ids]

    # handle zero deg
    degs = degs.tonumpy()
    if degs[-1] == 0:
        degs = degs[:-1]
        zero_deg_nodes = dsts[-1]
        dsts = dsts[:-1]
    else:
        zero_deg_nodes = None

    return v, degs, dsts, msg_ids, zero_deg_nodes

def _create_per_bkt_rfunc(reduce_udf, deg, vbkt, ntype=None):
    """Internal function to generate the per degree bucket node UDF."""
    def _rfunc_wrapper(node_data, mail_data):
        def _reshaped_getter(key):
            msg = mail_data[key]
            new_shape = (len(vbkt), deg) + F.shape(msg)[1:]
            return F.reshape(msg, new_shape)
        reshaped_mail_data = utils.LazyDict(_reshaped_getter, mail_data.keys())
        nbatch = NodeBatch(vbkt, node_data, reshaped_mail_data, ntype=ntype)
        return reduce_udf(nbatch)
    return _rfunc_wrapper

def gen_group_apply_edge_schedule(
        apply_func,
        u, v, eid,
        group_by,
        var_src_nf,
        var_dst_nf,
        var_ef,
        var_out,
        canonical_etype=(None, None, None)):
    """Create degree bucketing schedule for group_apply_edge

    Edges will be grouped by either its source node or destination node
    specified by 'group_by', and will be divided into buckets in which
    'group_by' nodes have the same degree. The apply_func UDF will be applied
    to each bucket. The per-bucket result will be merged according to the
    *unique-ascending order* of the edge ids.

    Parameters
    ----------
    apply_func: callable
        The edge_apply_func UDF
    u: utils.Index
        Source nodes of edges to apply
    v: utils.Index
        Destination nodes of edges to apply
    eid: utils.Index
        Edges to apply
    group_by: str
        If "src", group by u. If "dst", group by v
    var_src_nf : var.FEAT_DICT
        The variable for source feature frame.
    var_dst_nf : var.FEAT_DICT
        The variable for destination feature frame.
    var_ef : var.FEAT_DICT
        The variable for edge frame.
    var_out : var.FEAT_DICT
        The variable for output feature dicts.
    canonical_etype : tuple[str, str, str], optional
        Canonical edge type if running on a heterograph.
        Default: (None, None, None), if running on a homogeneous graph.
    """
    if group_by == "src":
        buckets = _degree_bucketing_for_edge_grouping(u, v, eid)
        degs, uids, vids, eids = buckets
    elif group_by == "dst":
        buckets = _degree_bucketing_for_edge_grouping(v, u, eid)
        degs, vids, uids, eids = buckets
    else:
        raise DGLError("group_apply_edge must be grouped by either src or dst")

    idx_list = []
    fd_list = []
    for deg, u_bkt, v_bkt, eid_bkt in zip(degs, uids, vids, eids):
        # create per-bkt efunc
        _efunc = var.FUNC(_create_per_bkt_efunc(apply_func, deg,
                                                u_bkt, v_bkt, eid_bkt,
                                                canonical_etype=canonical_etype))
        # vars
        var_u = var.IDX(u_bkt)
        var_v = var.IDX(v_bkt)
        var_eid = var.IDX(eid_bkt)
        # apply edge UDF on each bucket
        fdsrc = ir.READ_ROW(var_src_nf, var_u)
        fddst = ir.READ_ROW(var_dst_nf, var_v)
        fdedge = ir.READ_ROW(var_ef, var_eid)
        fdedge = ir.EDGE_UDF(_efunc, fdsrc, fdedge, fddst, ret=fdedge)  # reuse var
        # save for merge
        idx_list.append(var_eid)
        fd_list.append(fdedge)

    # merge buckets according to the ascending order of the edge ids.
    all_idx = F.cat([idx.data.tousertensor() for idx in idx_list], dim=0)
    _, order = F.sort_1d(all_idx)
    var_order = var.IDX(utils.toindex(order))
    ir.MERGE_ROW(var_order, fd_list, ret=var_out)

def _degree_bucketing_for_edge_grouping(uids, vids, eids):
    """Return the edge buckets by degree and grouped nodes for group_apply_edge

    Parameters
    ----------
    degree
    uids: utils.Index
        node id of one end of eids, based on which edges are grouped
    vids: utils.Index
        node id of the other end of eids
    eids: utils.Index
        edge id for each edge
    """
    buckets = _CAPI_DGLGroupEdgeByNodeDegree(uids.todgltensor(),
                                             vids.todgltensor(),
                                             eids.todgltensor())
    return _process_edge_buckets(buckets)

def _process_edge_buckets(buckets):
    """read bucketing auxiliary data for group_apply_edge buckets

    Returns
    -------
    degrees: numpy.ndarray
        A list of degree for each bucket
    uids: list of utils.Index
        A list of node id buckets, nodes in each bucket have the same degree
    vids: list of utils.Index
        A list of node id buckets
    eids: list of utils.Index
        A list of edge id buckets
    """
    # get back results
    dtype = buckets(0).dtype
    degs = buckets(0).asnumpy()
    uids = utils.toindex(buckets(1), dtype)
    vids = utils.toindex(buckets(2), dtype)
    eids = utils.toindex(buckets(3), dtype)
    # XXX: convert directly from ndarary to python list?
    sections = buckets(4).asnumpy().tolist()

    # split buckets and convert to index
    def split(to_split):
        res = F.split(to_split.tousertensor(), sections, 0)
        return map(partial(utils.toindex, dtype=dtype), res)

    uids = split(uids)
    vids = split(vids)
    eids = split(eids)
    return degs, uids, vids, eids

def _create_per_bkt_efunc(apply_func, deg, u, v, eid, canonical_etype=(None, None, None)):
    """Internal function to generate the per degree bucket edge UDF."""
    batch_size = len(u) // deg
    def _efunc_wrapper(src_data, edge_data, dst_data):
        def _reshape_func(data):
            def _reshaped_getter(key):
                feat = data[key]
                new_shape = (batch_size, deg) + F.shape(feat)[1:]
                return F.reshape(feat, new_shape)
            return _reshaped_getter

        def _reshape_back(data):
            shape = F.shape(data)[2:]
            new_shape = (batch_size * deg,) + shape
            return F.reshape(data, new_shape)

        reshaped_src_data = utils.LazyDict(_reshape_func(src_data),
                                           src_data.keys())
        reshaped_edge_data = utils.LazyDict(_reshape_func(edge_data),
                                            edge_data.keys())
        reshaped_dst_data = utils.LazyDict(_reshape_func(dst_data),
                                           dst_data.keys())
        ebatch = EdgeBatch((u, v, eid), reshaped_src_data,
                           reshaped_edge_data, reshaped_dst_data,
                           canonical_etype=canonical_etype)
        return {k: _reshape_back(v) for k, v in apply_func(ebatch).items()}
    return _efunc_wrapper

_init_api("dgl._deprecate.runtime.degree_bucketing")
