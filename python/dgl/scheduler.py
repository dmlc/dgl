"""Schedule policies for graph computation."""
from __future__ import absolute_import

import numpy as np

import dgl.backend as F
import dgl.function.message as fmsg
import dgl.function.reducer as fred
import dgl.utils as utils

__all__ = ["degree_bucketing", "get_executor"]

def degree_bucketing(cached_graph, v):
    """Create degree bucketing scheduling policy.

    Parameters
    ----------
    cached_graph : dgl.cached_graph.CachedGraph
        the graph
    v : dgl.utils.Index
        the nodes to gather messages

    Returns
    -------
    unique_degrees : list of int
        list of unique degrees
    v_bkt : list of dgl.utils.Index
        list of node id buckets; nodes belong to the same bucket have
        the same degree
    """
    degrees = F.asnumpy(cached_graph.in_degrees(v).totensor())
    unique_degrees = list(np.unique(degrees))
    v_np = np.array(v.tolist())
    v_bkt = []
    for deg in unique_degrees:
        idx = np.where(degrees == deg)
        v_bkt.append(utils.Index(v_np[idx]))
    #print('degree-bucketing:', unique_degrees, [len(b) for b in v_bkt])
    return unique_degrees, v_bkt

class Executor(object):
    def run(self, graph):
        raise NotImplementedError

class UpdateAllSPMVExecutor(Executor):
    def __init__(self, graph, src_field, dst_field, edge_field, use_adj):
        self.graph = graph
        self.src_field = src_field
        self.dst_field = dst_field
        self.edge_field = edge_field
        self.use_adj = use_adj

    def run(self):
        g = self.graph
        if self.src_field is None:
            srccol = g.get_n_repr()
        else:
            srccol = g.get_n_repr()[self.src_field]
        ctx = F.get_context(srccol)
        if self.use_adj:
            adjmat = g.cached_graph.adjmat().get(ctx)
        else:
            if self.edge_field is None:
                dat = g.get_e_repr()
            else:
                dat = g.get_e_repr()[self.edge_field]
            dat = F.squeeze(dat)
            # TODO(minjie): should not directly use _indices
            idx = g.cached_graph.adjmat().get(ctx)._indices()
            n = g.number_of_nodes()
            adjmat = F.sparse_tensor(idx, dat, [n, n])
        # spmm
        if len(F.shape(srccol)) == 1:
            srccol = F.unsqueeze(srccol, 1)
            dstcol = F.spmm(adjmat, srccol)
            dstcol = F.squeeze(dstcol)
        else:
            dstcol = F.spmm(adjmat, srccol)
        if self.dst_field is None:
            g.set_n_repr(dstcol)
        else:
            g.set_n_repr({self.dst_field : dstcol})

class SendRecvSPMVExecutor(Executor):
    def __init__(self, graph, src, dst, src_field, dst_field, edge_field, use_edge_dat):
        self.graph = graph
        self.src = src
        self.dst = dst
        self.src_field = src_field
        self.dst_field = dst_field
        self.edge_field = edge_field
        self.use_edge_dat = use_edge_dat

    def run(self):
        # get src col
        g = self.graph
        if self.src_field is None:
            srccol = g.get_n_repr()
        else:
            srccol = g.get_n_repr()[self.src_field]
        ctx = F.get_context(srccol)

        # build adjmat
        # build adjmat dat
        u, v = utils.edge_broadcasting(self.src, self.dst)
        if self.use_edge_dat:
            if self.edge_field is None:
                dat = g.get_e_repr(u, v)
            else:
                dat = g.get_e_repr(u, v)[self.edge_field]
            dat = F.squeeze(dat)
        else:
            dat = F.ones((len(u),))
        # build adjmat index
        new2old, old2new = utils.build_relabel_map(v)
        u = u.totensor()
        v = v.totensor()
        # TODO(minjie): should not directly use []
        new_v = old2new[v]
        idx = F.pack([F.unsqueeze(new_v, 0), F.unsqueeze(u, 0)])
        n = g.number_of_nodes()
        m = len(new2old)
        adjmat = F.sparse_tensor(idx, dat, [m, n])
        adjmat = F.to_context(adjmat, ctx)
        # spmm
        if len(F.shape(srccol)) == 1:
            srccol = F.unsqueeze(srccol, 1)
            dstcol = F.spmm(adjmat, srccol)
            dstcol = F.squeeze(dstcol)
        else:
            dstcol = F.spmm(adjmat, srccol)
        if self.dst_field is None:
            g.set_n_repr(dstcol, new2old)
        else:
            g.set_n_repr({self.dst_field : dstcol}, new2old)

def _is_spmv_supported_node_feat(g, field):
    if field is None:
        feat = g.get_n_repr()
    else:
        feat = g.get_n_repr()[field]
    shape = F.shape(feat)
    return (len(shape) == 1 or len(shape) == 2)

def _is_spmv_supported_edge_feat(g, field):
    # check shape, only scalar edge feature can be optimized at the moment.
    if field is None:
        feat = g.get_e_repr()
    else:
        feat = g.get_e_repr()[field]
    shape = F.shape(feat)
    return len(shape) == 1 or (len(shape) == 2 and shape[1] == 1)

def _create_update_all_exec(graph, **kwargs):
    mfunc = kwargs.pop('message_func')
    rfunc = kwargs.pop('reduce_func')
    if (isinstance(mfunc, fmsg.CopySrcMessageFunction)
            and isinstance(rfunc, fred.SumReducerFunction)
            and _is_spmv_supported_node_feat(graph, mfunc.src_field)):
        # TODO(minjie): more sanity check on field names
        return UpdateAllSPMVExecutor(graph,
                                     src_field=mfunc.src_field,
                                     dst_field=rfunc.out_field,
                                     edge_field=None,
                                     use_adj=True)
    elif (isinstance(mfunc, fmsg.SrcMulEdgeMessageFunction)
            and isinstance(rfunc, fred.SumReducerFunction)
            and _is_spmv_supported_node_feat(graph, mfunc.src_field)
            and _is_spmv_supported_edge_feat(graph, mfunc.edge_field)):
        return UpdateAllSPMVExecutor(graph,
                                     src_field=mfunc.src_field,
                                     dst_field=rfunc.out_field,
                                     edge_field=mfunc.edge_field,
                                     use_adj=False)
    elif (isinstance(mfunc, fmsg.CopyEdgeMessageFunction)
            and isinstance(rfunc, fred.SumReducerFunction)):
        return None
    else:
        return None

def _create_send_and_recv_exec(graph, **kwargs):
    src = kwargs.pop('src')
    dst = kwargs.pop('dst')
    mfunc = kwargs.pop('message_func')
    rfunc = kwargs.pop('reduce_func')
    if (isinstance(mfunc, fmsg.CopySrcMessageFunction)
            and isinstance(rfunc, fred.SumReducerFunction)
            and _is_spmv_supported_node_feat(graph, mfunc.src_field)):
        # TODO(minjie): more sanity check on field names
        return SendRecvSPMVExecutor(graph,
                                    src=src,
                                    dst=dst,
                                    src_field=mfunc.src_field,
                                    dst_field=rfunc.out_field,
                                    edge_field=None,
                                    use_edge_dat=False)
    elif (isinstance(mfunc, fmsg.SrcMulEdgeMessageFunction)
            and isinstance(rfunc, fred.SumReducerFunction)
            and _is_spmv_supported_node_feat(graph, mfunc.src_field)
            and _is_spmv_supported_edge_feat(graph, mfunc.edge_field)):
        return SendRecvSPMVExecutor(graph,
                                    src=src,
                                    dst=dst,
                                    src_field=mfunc.src_field,
                                    dst_field=rfunc.out_field,
                                    edge_field=mfunc.edge_field,
                                    use_edge_dat=True)
    else:
        return None

def get_executor(call_type, graph, **kwargs):
    if call_type == "update_all":
        return _create_update_all_exec(graph, **kwargs)
    elif call_type == "send_and_recv":
        return _create_send_and_recv_exec(graph, **kwargs)
    else:
        return None
