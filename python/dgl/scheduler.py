"""Schedule policies for graph computation."""
from __future__ import absolute_import

import numpy as np

import dgl.backend as F
import dgl.function.message as fmsg
import dgl.function.reducer as fred
import dgl.utils as utils
from dgl.base import ALL

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
    def run(self):
        raise NotImplementedError


class SPMVExecutor(Executor):
    def __init__(self, src_field, edge_field, dst_field, use_edge_feat,
                 node_repr, adj_build_fn):
        self.src_field = src_field
        self.edge_field = edge_field
        self.dst_field = dst_field
        self.use_edge_feat = use_edge_feat
        self.node_repr = node_repr
        self.adj_build_fn = adj_build_fn

    def run(self):
        # get src col
        if self.src_field is None:
            srccol = self.node_repr
        else:
            srccol = self.node_repr[self.src_field]
        ctx = F.get_context(srccol)

        # build adjmat
        adjmat = self.adj_build_fn(self.edge_field, ctx, self.use_edge_feat)

        # spmm
        if len(F.shape(srccol)) == 1:
            srccol = F.unsqueeze(srccol, 1)
            dstcol = F.spmm(adjmat, srccol)
            dstcol = F.squeeze(dstcol)
        else:
            dstcol = F.spmm(adjmat, srccol)
        if self.dst_field is None:
            return dstcol
        else:
            return {self.dst_field : dstcol}


class BundledExecutor(Executor):
    """
    Base class for Bundled execution
    All shared structure like graph index should be cached in this class or its subclass
    BundledUpdateAllExecutor and BundledSendRecvExecutor should subclass BundledExecutor
    """
    def __init__(self, graph, mfunc, rfunc):
        self.g = graph
        func_pairs = self._match_message_with_reduce(mfunc, rfunc)
        # create all executors
        self._build_executors(func_pairs)

    @property
    def node_repr(self):
        raise NotImplementedError

    @property
    def edge_repr(self):
        raise NotImplementedError

    def _adj_build_fn(self, edge_field, ctx, use_edge_feat):
        raise NotImplementedError

    def _build_executors(self, func_pairs):
        self.executors = []
        for mfunc, rfunc in func_pairs:
            if isinstance(mfunc, fmsg.CopySrcMessageFunction):
                exe = SPMVExecutor(src_field=mfunc.src_field,
                                   edge_field=None,
                                   dst_field=rfunc.out_field,
                                   use_edge_feat=False,
                                   node_repr=self.node_repr,
                                   adj_build_fn=self._adj_build_fn)
            elif isinstance(mfunc, fmsg.SrcMulEdgeMessageFunction):
                exe = SPMVExecutor(src_field=mfunc.src_field,
                                   edge_field=mfunc.edge_field,
                                   dst_field=rfunc.out_field,
                                   use_edge_feat=True,
                                   node_repr=self.node_repr,
                                   adj_build_fn=self._adj_build_fn)
            else:
                # degree bucketing here
                raise NotImplementedError
            self.executors.append(exe)

    def _match_message_with_reduce(self, mfunc, rfunc):
        out2mfunc = {fn.out_field: fn for fn in mfunc.fn_list}
        func_pairs = []
        for rfn in rfunc.fn_list:
            mfn = out2mfunc.get(rfn.msg_field, None)
            # field check
            assert mfn is not None, \
                    "cannot match message func with reduce func"
            func_pairs.append((mfn, rfn))
        return func_pairs

    def run(self):
        new_attr = {}
        for exe in self.executors:
            attr = exe.run()
            if isinstance(attr, dict):
                new_attr.update(attr)
            else:
                self.g.set_n_repr(attr, self.graph_mapping)
        if len(new_attr) > 0:
            self.g.set_n_repr(new_attr, self.graph_mapping)


class BundledUpdateAllExecutor(BundledExecutor):
    def __init__(self, graph, mfunc, rfunc):
        self._node_repr = None
        self._edge_repr = None
        super(BundledUpdateAllExecutor, self).__init__(graph, mfunc, rfunc)
        self._graph_idx = None
        self._graph_shape = None
        self._graph_mapping = None

    @property
    def graph_idx(self):
        if self._graph_idx is None:
            self._graph_idx = self.g.cached_graph.adjmat()
        return self._graph_idx

    @property
    def graph_shape(self):
        if self._graph_shape is None:
            n = self.g.number_of_nodes()
            self._graph_shape = [n, n]
        return self._graph_shape

    @property
    def graph_mapping(self):
        return ALL

    @property
    def node_repr(self):
        if self._node_repr is None:
            self._node_repr = self.g.get_n_repr()
        return self._node_repr

    @property
    def edge_repr(self):
        if self._edge_repr is None:
            self._edge_repr = self.g.get_e_repr()
        return self._edge_repr

    def _adj_build_fn(self, edge_field, ctx, use_edge_feat):
        if use_edge_feat:
            if edge_field is None:
                dat = self.edge_repr
            else:
                dat = self.edge_repr[edge_field]
            dat = F.squeeze(dat)
            # TODO(minjie): should not directly use _indices
            idx = self.graph_idx.get(ctx)._indices()
            adjmat = F.sparse_tensor(idx, dat, self.graph_shape)
        else:
            adjmat = self.graph_idx.get(ctx)
        return adjmat


class BundledSendRecvExecutor(BundledExecutor):
    def __init__(self, graph, src, dst, mfunc, rfunc):
        self.u, self.v = utils.edge_broadcasting(src, dst)
        self._node_repr = None
        self._edge_repr = None
        super(BundledSendRecvExecutor, self).__init__(graph, mfunc, rfunc)
        self._graph_idx = None
        self._graph_shape = None
        self._graph_mapping = None

    @property
    def graph_idx(self):
        if self._graph_idx is None:
            self._build_adjmat()
        return self._graph_idx

    @property
    def graph_shape(self):
        if self._graph_shape is None:
            self._build_adjmat()
        return self._graph_shape

    @property
    def graph_mapping(self):
        if self._graph_mapping is None:
            self._build_adjmat()
        return self._graph_mapping

    @property
    def node_repr(self):
        if self._node_repr is None:
            self._node_repr = self.g.get_n_repr()
        return self._node_repr

    @property
    def edge_repr(self):
        if self._edge_repr is None:
            self._edge_repr = self.g.get_e_repr(self.u, self.v)
        return self._edge_repr

    def _build_adjmat(self):
        # handle graph index
        new2old, old2new = utils.build_relabel_map(self.v)
        u = self.u.totensor()
        v = self.v.totensor()
        # TODO(minjie): should not directly use []
        new_v = old2new[v]
        n = self.g.number_of_nodes()
        m = len(new2old)
        self._graph_idx = F.pack([F.unsqueeze(new_v, 0), F.unsqueeze(u, 0)])
        self._graph_shape = [m, n]
        self._graph_mapping = new2old

    def _adj_build_fn(self, edge_field, ctx, use_edge_feat):
        if use_edge_feat:
            if edge_field is None:
                dat = self.edge_repr
            else:
                dat = self.edge_repr[edge_field]
            dat = F.squeeze(dat)
        else:
            dat = F.ones((len(self.u), ))
        adjmat = F.sparse_tensor(self.graph_idx, dat, self.graph_shape)
        return F.to_context(adjmat, ctx)


def _create_update_all_exec(graph, **kwargs):
    mfunc = kwargs.pop('message_func')
    rfunc = kwargs.pop('reduce_func')
    mfunc = fmsg.BundledMessageFunction(mfunc)
    rfunc = fred.BundledReduceFunction(rfunc)
    if mfunc.is_spmv_supported(graph) and rfunc.is_spmv_supported():
        return BundledUpdateAllExecutor(graph, mfunc=mfunc, rfunc=rfunc)
    else:
        return None

def _create_send_and_recv_exec(graph, **kwargs):
    src = kwargs.pop('src')
    dst = kwargs.pop('dst')
    mfunc = kwargs.pop('message_func')
    rfunc = kwargs.pop('reduce_func')
    mfunc = fmsg.BundledMessageFunction(mfunc)
    rfunc = fred.BundledReduceFunction(rfunc)
    if mfunc.is_spmv_supported(graph) and rfunc.is_spmv_supported():
        return BundledSendRecvExecutor(graph, src=src, dst=dst, mfunc=mfunc, rfunc=rfunc)
    else:
        return None

def get_executor(call_type, graph, **kwargs):
    if call_type == "update_all":
        return _create_update_all_exec(graph, **kwargs)
    elif call_type == "send_and_recv":
        return _create_send_and_recv_exec(graph, **kwargs)
    else:
        return None
