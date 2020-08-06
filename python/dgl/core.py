"""Implementation for core graph computation."""

from .base import DGLError, is_all, NID, EID, ALL
from . import backend as F
from . import function as fn
from .frame import Frame
from .udf import NodeBatch, EdgeBatch
from . import ops

def is_builtin(func):
    """Return true if the function is a DGL builtin function."""
    return isinstance(func, fn.BuiltinFunction)

def invoke_node_udf(graph, ntype, func, ndata, *, orig_nid=None):
    """Invoke user-defined node function on all the nodes in the graph.

    Parameters
    ----------
    graph : DGLGraph
        The input graph.
    ntype : str
        Node type.
    func : callable
        The user-defined function.
    ndata : dict[str, Tensor]
        Node feature data.
    orig_nid : Tensor, optional
        Original node IDs. Useful if the input graph is an extracted subgraph.

    Returns
    -------
    dict[str, Tensor]
        Results from running the UDF.
    """
    if orig_nid is None:
        orig_nid = graph.nodes(ntype=ntype)
    nbatch = NodeBatch(graph, orig_nid, ntype, ndata)
    return func(nbatch)

def invoke_edge_udf(graph, eid, etype, func, *, orig_eid=None):
    etid = graph.get_etype_id(etype)
    stid, dtid = graph._graph.metagraph.find_edge(etid)
    if is_all(eid):
        u, v, eid = graph.edges(form='all')
        edata = graph._edge_frames[etid]
    else:
        u, v = graph.find_edges(eid)
        edata = graph._edge_frames[etid].subframe(eid)
    srcdata = graph._node_frames[stid].subframe(u)
    dstdata = graph._node_frames[dtid].subframe(v)
    ebatch = EdgeBatch(graph, eid if orig_eid is None else orig_eid,
                       etype, srcdata, edata, dstdata)
    return func(ebatch)

def invoke_udf_reduce(graph, func, msgdata, orig_nid=None):
    degs = graph.in_degrees()
    nodes = graph.dstnodes()
    if orig_nid is None:
        orig_nid = nodes
    ntype = graph.dsttypes[0]
    ntid = graph.get_ntype_id_from_dst(ntype)
    dstdata = graph._node_frames[ntid]
    msgdata = Frame(msgdata)

    # degree bucketing
    unique_degs, bucketor = _bucketing(degs)
    bkt_rsts = []
    bkt_nodes = []
    for deg, node_bkt, orig_nid_bkt in zip(unique_degs, bucketor(nodes), bucketor(orig_nid)):
        if deg == 0:
            # skip reduce function for zero-degree nodes
            continue
        bkt_nodes.append(node_bkt)
        ndata_bkt = dstdata.subframe(node_bkt)
        eid_bkt = graph.in_edges(node_bkt, form='eid')
        assert len(eid_bkt) == deg * len(node_bkt)
        msgdata_bkt = msgdata.subframe(eid_bkt)
        # reshape all msg tensors to (num_nodes_bkt, degree, feat_size)
        maildata = {}
        for k, msg in msgdata_bkt.items():
            newshape = (len(node_bkt), deg) + F.shape(msg)[1:]
            maildata[k] = F.reshape(msg, newshape)
        # invoke udf
        nbatch = NodeBatch(graph, orig_nid_bkt, ntype, ndata_bkt, msgs=maildata)
        bkt_rsts.append(func(nbatch))

    # prepare a result frame
    retf = Frame(num_rows=len(nodes))
    retf._initializers = dstdata._initializers
    retf._default_initializer = dstdata._default_initializer

    # merge bucket results and write to the result frame
    if len(bkt_rsts) != 0:  # if all the nodes have zero degree, no need to merge results.
        merged_rst = {}
        for k in bkt_rsts[0].keys():
            merged_rst[k] = F.cat([rst[k] for rst in bkt_rsts], dim=0)
        merged_nodes = F.cat(bkt_nodes, dim=0)
        retf.update_row(merged_nodes, merged_rst)

    return retf

def _bucketing(val):
    sorted_val, idx = F.sort_1d(val)
    unique_val = F.asnumpy(F.unique(sorted_val))
    bkt_idx = []
    for v in unique_val:
        eqidx = F.nonzero_1d(F.equal(sorted_val, v))
        bkt_idx.append(F.gather_row(idx, eqidx))
    def bucketor(data):
        bkts = [F.gather_row(data, idx) for idx in bkt_idx]
        return bkts
    return unique_val, bucketor

def invoke_gsddmm(graph, func):
    alldata = [graph.srcdata, graph.dstdata, graph.edata]
    if isinstance(func, fn.BinaryMessageFunction):
        x = alldata[func.lhs][func.lhs_field]
        y = alldata[func.rhs][func.rhs_field]
        op = getattr(ops, func.name)
        z = op(graph, x, y)
    else:
        x = alldata[func.target][func.in_field]
        op = getattr(ops, func.name)
        z = op(graph, x)
    return {func.out_field : z}

def invoke_gspmm(graph, mfunc, rfunc, *, srcdata=None, dstdata=None, edata=None):
    # sanity check
    if mfunc.out_field != rfunc.msg_field:
        raise DGLError('Invalid message ({}) and reduce ({}) function pairs.'
                       ' The output field of the message function must be equal to the'
                       ' message field of the reduce function.'.format(mfunc, rfunc))
    if edata is None:
        edata = graph.edata
    if srcdata is None:
        srcdata = graph.srcdata
    if dstdata is None:
        dstdata = graph.dstdata
    alldata = [srcdata, dstdata, edata]

    if isinstance(mfunc, fn.BinaryMessageFunction):
        x = alldata[mfunc.lhs][mfunc.lhs_field]
        y = alldata[mfunc.rhs][mfunc.rhs_field]
        op = getattr(ops, '{}_{}'.format(mfunc.name, rfunc.name))
        z = op(graph, x, y)
    else:
        x = alldata[mfunc.target][mfunc.in_field]
        op = getattr(ops, '{}_{}'.format(mfunc.name, rfunc.name))
        z = op(graph, x)
    return {rfunc.out_field : z}

def message_passing(g, mfunc, rfunc, afunc):
    if g.number_of_edges() == 0:
        # No message passing is triggered.
        ndata = {}
    elif (is_builtin(mfunc) and is_builtin(rfunc) and
            getattr(ops, '{}_{}'.format(mfunc.name, rfunc.name), None) is not None):
        # invoke fused message passing
        ndata = invoke_gspmm(g, mfunc, rfunc)
    else:
        # invoke message passing in two separate steps
        # message phase
        if is_builtin(mfunc):
            msgdata = invoke_gsddmm(g, mfunc)
        else:
            orig_eid = g.edata.get(EID, None)
            msgdata = invoke_edge_udf(g, ALL, g.canonical_etypes[0], mfunc, orig_eid=orig_eid)
        # reduce phase
        if is_builtin(rfunc):
            msg = rfunc.msg_field
            ndata = invoke_gspmm(g, fn.copy_e(msg, msg), rfunc, edata=msgdata)
        else:
            orig_nid = g.dstdata.get(NID, None)
            ndata = invoke_udf_reduce(g, rfunc, msgdata, orig_nid=orig_nid)
    # apply phase
    if afunc is not None:
        for k, v in g.dstdata.items():   # include original node features
            if k not in ndata:
                ndata[k] = v
        orig_nid = g.dstdata.get(NID, None)
        ndata = invoke_node_udf(g, g.dsttypes[0], afunc, ndata, orig_nid=orig_nid)
    return ndata
