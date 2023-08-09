"""Implementation for core graph computation."""
# pylint: disable=not-callable
import numpy as np

from . import backend as F, function as fn, ops
from .base import ALL, dgl_warning, DGLError, EID, is_all, NID
from .frame import Frame
from .udf import EdgeBatch, NodeBatch


def is_builtin(func):
    """Return true if the function is a DGL builtin function."""
    return isinstance(func, fn.BuiltinFunction)


def invoke_node_udf(graph, nid, ntype, func, *, ndata=None, orig_nid=None):
    """Invoke user-defined node function on the given nodes.

    Parameters
    ----------
    graph : DGLGraph
        The input graph.
    nid : Tensor
        The IDs of the nodes to invoke UDF on.
    ntype : str
        Node type.
    func : callable
        The user-defined function.
    ndata : dict[str, Tensor], optional
        If provided, apply the UDF on this ndata instead of the ndata of the graph.
    orig_nid : Tensor, optional
        Original node IDs. Useful if the input graph is an extracted subgraph.

    Returns
    -------
    dict[str, Tensor]
        Results from running the UDF.
    """
    ntid = graph.get_ntype_id(ntype)
    if ndata is None:
        if is_all(nid):
            ndata = graph._node_frames[ntid]
            nid = graph.nodes(ntype=ntype)
        else:
            ndata = graph._node_frames[ntid].subframe(nid)
    nbatch = NodeBatch(
        graph, nid if orig_nid is None else orig_nid, ntype, ndata
    )
    return func(nbatch)


def invoke_edge_udf(graph, eid, etype, func, *, orig_eid=None):
    """Invoke user-defined edge function on the given edges.

    Parameters
    ----------
    graph : DGLGraph
        The input graph.
    eid : Tensor
        The IDs of the edges to invoke UDF on.
    etype : (str, str, str)
        Edge type.
    func : callable
        The user-defined function.
    orig_eid : Tensor, optional
        Original edge IDs. Useful if the input graph is an extracted subgraph.

    Returns
    -------
    dict[str, Tensor]
        Results from running the UDF.
    """
    etid = graph.get_etype_id(etype)
    stid, dtid = graph._graph.metagraph.find_edge(etid)
    if is_all(eid):
        u, v, eid = graph.edges(form="all")
        edata = graph._edge_frames[etid]
    else:
        u, v = graph.find_edges(eid)
        edata = graph._edge_frames[etid].subframe(eid)
    if len(u) == 0:
        dgl_warning(
            "The input graph for the user-defined edge function "
            "does not contain valid edges"
        )
    srcdata = graph._node_frames[stid].subframe(u)
    dstdata = graph._node_frames[dtid].subframe(v)
    ebatch = EdgeBatch(
        graph,
        eid if orig_eid is None else orig_eid,
        etype,
        srcdata,
        edata,
        dstdata,
    )
    return func(ebatch)


def invoke_udf_reduce(graph, func, msgdata, *, orig_nid=None):
    """Invoke user-defined reduce function on all the nodes in the graph.

    It analyzes the graph, groups nodes by their degrees and applies the UDF on each
    group -- a strategy called *degree-bucketing*.

    Parameters
    ----------
    graph : DGLGraph
        The input graph.
    func : callable
        The user-defined function.
    msgdata : dict[str, Tensor]
        Message data.
    orig_nid : Tensor, optional
        Original node IDs. Useful if the input graph is an extracted subgraph.

    Returns
    -------
    dict[str, Tensor]
        Results from running the UDF.
    """
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
    for deg, node_bkt, orig_nid_bkt in zip(
        unique_degs, bucketor(nodes), bucketor(orig_nid)
    ):
        if deg == 0:
            # skip reduce function for zero-degree nodes
            continue
        bkt_nodes.append(node_bkt)
        ndata_bkt = dstdata.subframe(node_bkt)

        # order the incoming edges per node by edge ID
        eid_bkt = F.zerocopy_to_numpy(graph.in_edges(node_bkt, form="eid"))
        assert len(eid_bkt) == deg * len(node_bkt)
        eid_bkt = np.sort(eid_bkt.reshape((len(node_bkt), deg)), 1)
        eid_bkt = F.zerocopy_from_numpy(eid_bkt.flatten())

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
    if (
        len(bkt_rsts) != 0
    ):  # if all the nodes have zero degree, no need to merge results.
        merged_rst = {}
        for k in bkt_rsts[0].keys():
            merged_rst[k] = F.cat([rst[k] for rst in bkt_rsts], dim=0)
        merged_nodes = F.cat(bkt_nodes, dim=0)
        retf.update_row(merged_nodes, merged_rst)

    return retf


def _bucketing(val):
    """Internal function to create groups on the values.

    Parameters
    ----------
    val : Tensor
        Value tensor.

    Returns
    -------
    unique_val : Tensor
        Unique values.
    bucketor : callable[Tensor -> list[Tensor]]
        A bucketing function that splits the given tensor data as the same
        way of how the :attr:`val` tensor is grouped.
    """
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


def data_dict_to_list(graph, data_dict, func, target):
    """Get node or edge feature data of the given name for all the types.

    Parameters
    -------------
    graph :  DGLGraph
        The input graph.
    data_dict : dict[str, Tensor] or dict[(str, str, str), Tensor]] or Tensor
        Node or edge data stored in DGLGraph. The key of the dictionary
        is the node type name or edge type name. If there is only single source
        node type, data_dict is the value of feature(a Tensor) not a dict.
    func : dgl.function.BaseMessageFunction
        Built-in message function.
    target : 'u', 'v' or 'e'
        The target of the lhs or rhs data

    Returns
    --------
    data_list : list(Tensor)
        Feature data stored in a list of tensors. The i^th tensor stores the feature
        data of type ``types[i]``.
    """
    if isinstance(func, fn.BinaryMessageFunction):
        if target in ["u", "v"]:
            output_list = [None] * graph._graph.number_of_ntypes()
            # If there is only single source node type, data_dict should be the value of
            # feature, namely, a tensor.
            if not isinstance(data_dict, dict):
                src_id, dst_id = graph._graph.metagraph.find_edge(0)
                if target == "u":
                    output_list[src_id] = data_dict
                else:
                    output_list[dst_id] = data_dict
            else:
                for srctype, _, dsttype in graph.canonical_etypes:
                    if target == "u":
                        src_id = graph.get_ntype_id(srctype)
                        output_list[src_id] = data_dict[srctype]
                    else:
                        dst_id = graph.get_ntype_id(dsttype)
                        output_list[dst_id] = data_dict[dsttype]
        else:  # target == 'e'
            output_list = [None] * graph._graph.number_of_etypes()
            for rel in graph.canonical_etypes:
                etid = graph.get_etype_id(rel)
                output_list[etid] = data_dict[rel]
        return output_list
    else:
        if target == "u":
            lhs_list = [None] * graph._graph.number_of_ntypes()
            if not isinstance(data_dict, dict):
                src_id, _ = graph._graph.metagraph.find_edge(0)
                lhs_list[src_id] = data_dict
            else:
                for srctype, _, _ in graph.canonical_etypes:
                    src_id = graph.get_ntype_id(srctype)
                    lhs_list[src_id] = data_dict[srctype]
            return lhs_list
        else:  # target == 'e':
            rhs_list = [None] * graph._graph.number_of_etypes()
            for rel in graph.canonical_etypes:
                etid = graph.get_etype_id(rel)
                rhs_list[etid] = data_dict[rel]
            return rhs_list


def invoke_gsddmm(graph, func):
    """Invoke g-SDDMM computation on the graph.

    Parameters
    ----------
    graph :  DGLGraph
        The input graph.
    func : dgl.function.BaseMessageFunction
        Built-in message function.

    Returns
    -------
    dict[str, Tensor]
        Results from the g-SDDMM computation.
    """
    alldata = [graph.srcdata, graph.dstdata, graph.edata]
    if isinstance(func, fn.BinaryMessageFunction):
        x = alldata[func.lhs][func.lhs_field]
        y = alldata[func.rhs][func.rhs_field]
        op = getattr(ops, func.name)
        if graph._graph.number_of_etypes() > 1:
            lhs_target, _, rhs_target = func.name.split("_", 2)
            x = data_dict_to_list(graph, x, func, lhs_target)
            y = data_dict_to_list(graph, y, func, rhs_target)
        z = op(graph, x, y)
    else:
        x = alldata[func.target][func.in_field]
        op = getattr(ops, func.name)
        if graph._graph.number_of_etypes() > 1:
            # Convert to list as dict is unordered.
            if func.name == "copy_u":
                x = data_dict_to_list(graph, x, func, "u")
            else:  # "copy_e"
                x = data_dict_to_list(graph, x, func, "e")
        z = op(graph, x)
    return {func.out_field: z}


def invoke_gspmm(
    graph, mfunc, rfunc, *, srcdata=None, dstdata=None, edata=None
):
    """Invoke g-SPMM computation on the graph.

    Parameters
    ----------
    graph :  DGLGraph
        The input graph.
    mfunc : dgl.function.BaseMessageFunction
        Built-in message function.
    rfunc : dgl.function.BaseReduceFunction
        Built-in reduce function.
    srcdata : dict[str, Tensor], optional
        Source node feature data. If not provided, it use ``graph.srcdata``.
    dstdata : dict[str, Tensor], optional
        Destination node feature data. If not provided, it use ``graph.dstdata``.
    edata : dict[str, Tensor], optional
        Edge feature data. If not provided, it use ``graph.edata``.

    Returns
    -------
    dict[str, Tensor]
        Results from the g-SPMM computation.
    """
    # sanity check
    if mfunc.out_field != rfunc.msg_field:
        raise DGLError(
            "Invalid message ({}) and reduce ({}) function pairs."
            " The output field of the message function must be equal to the"
            " message field of the reduce function.".format(mfunc, rfunc)
        )
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
        op = getattr(ops, "{}_{}".format(mfunc.name, rfunc.name))
        if graph._graph.number_of_etypes() > 1:
            lhs_target, _, rhs_target = mfunc.name.split("_", 2)
            x = data_dict_to_list(graph, x, mfunc, lhs_target)
            y = data_dict_to_list(graph, y, mfunc, rhs_target)
        z = op(graph, x, y)
    else:
        x = alldata[mfunc.target][mfunc.in_field]
        op = getattr(ops, "{}_{}".format(mfunc.name, rfunc.name))
        if graph._graph.number_of_etypes() > 1 and not isinstance(x, tuple):
            if mfunc.name == "copy_u":
                x = data_dict_to_list(graph, x, mfunc, "u")
            else:  # "copy_e"
                x = data_dict_to_list(graph, x, mfunc, "e")
        z = op(graph, x)
    return {rfunc.out_field: z}


def message_passing(g, mfunc, rfunc, afunc):
    """Invoke message passing computation on the whole graph.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    mfunc : callable or dgl.function.BuiltinFunction
        Message function.
    rfunc : callable or dgl.function.BuiltinFunction
        Reduce function.
    afunc : callable or dgl.function.BuiltinFunction
        Apply function.

    Returns
    -------
    dict[str, Tensor]
        Results from the message passing computation.
    """
    if (
        is_builtin(mfunc)
        and is_builtin(rfunc)
        and getattr(ops, "{}_{}".format(mfunc.name, rfunc.name), None)
        is not None
    ):
        # invoke fused message passing
        ndata = invoke_gspmm(g, mfunc, rfunc)
    else:
        # invoke message passing in two separate steps
        # message phase
        if is_builtin(mfunc):
            msgdata = invoke_gsddmm(g, mfunc)
        else:
            orig_eid = g.edata.get(EID, None)
            msgdata = invoke_edge_udf(
                g, ALL, g.canonical_etypes[0], mfunc, orig_eid=orig_eid
            )
        # reduce phase
        if is_builtin(rfunc):
            msg = rfunc.msg_field
            ndata = invoke_gspmm(g, fn.copy_e(msg, msg), rfunc, edata=msgdata)
        else:
            orig_nid = g.dstdata.get(NID, None)
            ndata = invoke_udf_reduce(g, rfunc, msgdata, orig_nid=orig_nid)
    # apply phase
    if afunc is not None:
        for k, v in g.dstdata.items():  # include original node features
            if k not in ndata:
                ndata[k] = v
        orig_nid = g.dstdata.get(NID, None)
        ndata = invoke_node_udf(
            g, ALL, g.dsttypes[0], afunc, ndata=ndata, orig_nid=orig_nid
        )
    return ndata
