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

    def _to_list_of_fields(self, src_field, dst_field, edge_field):
        src_field = src_field if isinstance(src_field, (tuple, list)) else [src_field]
        dst_field = dst_field if isinstance(dst_field, (tuple, list)) else [dst_field]
        edge_field = edge_field if isinstance(edge_field, (tuple, list)) else [edge_field]
        n_src = len(src_field)
        n_dst = len(dst_field)
        n_edge = len(edge_field)
        # only 1-1-1, N-1-N, 1-N-N, N-N-N supported
        assert n_src == n_dst and (n_edge == n_src or n_edge == 1) or \
                n_edge == n_dst and (n_src == n_edge or n_src == 1)
        return src_field, dst_field, edge_field


class UpdateAllSPMVExecutor(Executor):
    def __init__(self, graph, src_field, dst_field, edge_field, use_adj):
        self.graph = graph
        self.src_field, self.dst_field, self.edge_field = \
                self._to_list_of_fields(src_field, dst_field, edge_field)
        self.use_adj = use_adj

    def run(self):
        g = self.graph
        n = g.number_of_nodes()
        adjmat_cache = g.cached_graph.adjmat()

        # cache src
        if len(self.src_field) == 1:
            src_field = self.src_field[0]
            if src_field is None:
                srccol_cache = g.get_n_repr()
            else:
                srccol_cache = g.get_n_repr()[src_field]
            ctx = F.get_context(srccol_cache)
        else:
            srccol_cache = None

        # cache edge
        if len(self.edge_field) == 1:
            edge_field = self.edge_field[0]
            # build adjmat dat
            if not self.use_adj:
                if edge_field is None:
                    edgecol = g.get_e_repr()
                else:
                    edgecol = g.get_e_repr()[edge_field]
                edgecol = F.squeeze(edgecol)
        else:
            edgecol = None

        # loop over fields
        for idx, dst_field in enumerate(self.dst_field):
            if srccol_cache is None:
                src_field = self.src_field[idx]
                if src_field is None:
                    srccol = g.get_n_repr()
                else:
                    srccol = g.get_n_repr()[src_field]
                ctx = F.get_context(srccol)
            else:
                srccol = srccol_cache

            adjmat = adjmat_cache.get(ctx)
            if not self.use_adj:
                if edgecol is None:
                    edge_field = self.edge_field[idx]
                    if edge_field is None:
                        dat = g.get_e_repr()
                    else:
                        dat = g.get_e_repr()[edge_field]
                    dat = F.squeeze(dat)
                else:
                    dat = edgecol
                # TODO(minjie): should not directly use _indices
                adjmat = F.sparse_tensor(adjmat._indices(), dat, [n, n])
            # spmm
            if len(F.shape(srccol)) == 1:
                srccol = F.unsqueeze(srccol, 1)
                dstcol = F.spmm(adjmat, srccol)
                dstcol = F.squeeze(dstcol)
            else:
                dstcol = F.spmm(adjmat, srccol)
            if dst_field is None:
                g.set_n_repr(dstcol)
            else:
                g.set_n_repr({dst_field : dstcol})

class SendRecvSPMVExecutor(Executor):
    def __init__(self, graph, src, dst, src_field, dst_field, edge_field, use_edge_dat):
        self.graph = graph
        self.src = src
        self.dst = dst
        self.src_field, self.dst_field, self.edge_field = \
                self._to_list_of_fields(src_field, dst_field, edge_field)
        self.use_edge_dat = use_edge_dat

    def run(self):
        g = self.graph

        # build adjmat index
        u, v = utils.edge_broadcasting(self.src, self.dst)
        new2old, old2new = utils.build_relabel_map(v)
        u = u.totensor()
        v = v.totensor()
        # TODO(minjie): should not directly use []
        new_v = old2new[v]
        idx = F.pack([F.unsqueeze(new_v, 0), F.unsqueeze(u, 0)])
        n = g.number_of_nodes()
        m = len(new2old)

        # cache edge frame lazy dict
        edge_feat = g.get_e_repr(u, v)

        # cache src
        if len(self.src_field) == 1:
            src_field = self.src_field[0]
            if src_field is None:
                srccol_cache = g.get_n_repr()
            else:
                srccol_cache = g.get_n_repr()[src_field]
            ctx = F.get_context(srccol_cache)
        else:
            srccol_cache = None

        # cache edge
        if len(self.edge_field) == 1:
            edge_field = self.edge_field[0]
            # build adjmat dat
            if self.use_edge_dat:
                if edge_field is None:
                    dat = edge_feat
                else:
                    dat = edge_feat[edge_field]
                dat = F.squeeze(dat)
            else:
                dat = F.ones((len(u),))
            adjmat_cache = F.sparse_tensor(idx, dat, [m, n])
        else:
            adjmat_cache = None

        # loop over fields
        for idx, dst_field in enumerate(self.dst_field):
            # get src col
            if srccol_cache is None:
                src_field = self.src_field[idx]
                if src_field is None:
                    srccol = g.get_n_repr()
                else:
                    srccol = g.get_n_repr()[src_field]
                ctx = F.get_context(srccol)
            else:
                srccol = srccol_cache

            # get adjmat
            if adjmat_cache is None:
                edge_field = self.edge_field[idx]
                # build adjmat dat
                if self.use_edge_dat:
                    if edge_field is None:
                        dat = edge_feat
                    else:
                        dat = edge_feat[edge_field]
                    dat = F.squeeze(dat)
                else:
                    dat = F.ones((len(u),))
                # build adjmat
                adjmat = F.sparse_tensor(idx, dat, [m, n])
            else:
                adjmat = adjmat_cache

            # convert to context
            adjmat = F.to_context(adjmat, ctx)
            # spmm
            if len(F.shape(srccol)) == 1:
                srccol = F.unsqueeze(srccol, 1)
                dstcol = F.spmm(adjmat, srccol)
                dstcol = F.squeeze(dstcol)
            else:
                dstcol = F.spmm(adjmat, srccol)
            if dst_field is None:
                g.set_n_repr(dstcol, new2old)
            else:
                g.set_n_repr({dst_field : dstcol}, new2old)

def _is_spmv_supported_node_feat(g, field):
    features = g.get_n_repr()
    if field is None:
        pass
    elif isinstance(field, str):
        features = features[field]
    else: # iterable
        features = [features[f] for f in field]
    if isinstance(features, list):
        for feat in features:
            shape = F.shape(feat)
            if len(shape) != 1 and len(shape) != 2:
                return False
        return True
    else:
        shape = F.shape(features)
        return len(shape) == 1 or len(shape) == 2

def _is_spmv_supported_edge_feat(g, field):
    # check shape, only scalar edge feature can be optimized at the moment.
    features = g.get_e_repr()
    if field is None:
        pass
    elif isinstance(field, str):
        features = features[field]
    else: # iterable
        features = [features[f] for f in field]
    if isinstance(features, list):
        for feat in features:
            shape = F.shape(feat)
            if not (len(shape) == 1 or (len(shape) == 2 and shape[1] == 1)):
                return False
        return True
    else:
        shape = F.shape(features)
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
