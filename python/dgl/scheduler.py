"""Schedule policies for graph computation."""
from __future__ import absolute_import

import numpy as np

from .base import ALL, __MSG__, __REPR__
from . import backend as F
from .function import message as fmsg
from .function import reducer as fred
from . import utils
from collections import defaultdict as ddict

from ._ffi.function import _init_api

__all__ = ["degree_bucketing", "get_recv_executor", "get_executor"]

def degree_bucketing(graph, v):
    """Create degree bucketing scheduling policy.

    Parameters
    ----------
    graph : dgl.graph_index.GraphIndex
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
    degrees = np.array(graph.in_degrees(v).tolist())
    unique_degrees = list(np.unique(degrees))
    v_np = np.array(v.tolist())
    v_bkt = []
    for deg in unique_degrees:
        idx = np.where(degrees == deg)
        v_bkt.append(utils.Index(v_np[idx]))
    #print('degree-bucketing:', unique_degrees, [len(b) for b in v_bkt])
    return unique_degrees, v_bkt

def _process_buckets(buckets):
    """read bucketing auxiliary data"""
    # get back results
    degs = utils.toindex(buckets(0))
    v = utils.toindex(buckets(1))
    # FIXME: how to convert directly from ndarary to python list?
    v_section = buckets(2).asnumpy().tolist()
    msg_ids = utils.toindex(buckets(3))
    msg_section = buckets(4).asnumpy().tolist()

    # split buckets
    unique_v = v.tousertensor()
    #v_section = v_section.tolist().tolist() # pytorch split only accepts list, tuple
    msg_ids = msg_ids.tousertensor()
    #msg_section = msg_section.tolist().tolist()
    dsts = F.unpack(unique_v, v_section)
    msg_ids = F.unpack(msg_ids, msg_section)

    # convert to utils.Index
    unique_v = utils.toindex(unique_v)
    dsts = [utils.toindex(dst) for dst in dsts]
    msg_ids = [utils.toindex(msg_id) for msg_id in msg_ids]

    return unique_v, degs, dsts, msg_ids

def light_degree_bucketing(v):
    """Return the bucketing by degree scheduling for destination nodes of messages

    Parameters
    ----------
    v: utils.Index
        destionation node for each message

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
    buckets = _CAPI_DGLDegreeBucketing(v.todgltensor())
    return _process_buckets(buckets)

def light_degree_bucketing_for_graph(graph):
    """Return the bucketing by degree scheduling for the entire graph

    Parameters:
        graph: GraphIndex

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
    buckets = _CAPI_DGLDegreeBucketingFromGraph(self._handle)
    return _process_buckets(buckets)


class Executor(object):
    def run(self):
        raise NotImplementedError

class SPMVOperator(Executor):
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


# FIXME: refactorize in scheduler/executor redesign
class DegreeBucketingExecutor(Executor):
    def __init__(self, g, rfunc, message_frame, edges=None):
        self.g = g
        self.rfunc = rfunc
        self.msg_frame = message_frame

        # calc degree bucketing schedule
        if edges is not None:
            unique_v, degs, dsts, msg_ids = light_degree_bucketing(edges[1])
        else:
            unique_v, degs, dsts, msg_ids = light_degree_bucketing_for_graph(g._graph)
        self._recv_nodes = unique_v
        self.degrees = degs
        self.dsts = dsts
        self.msg_ids = msg_ids

    @property
    def recv_nodes(self):
        return self._recv_nodes

    def run(self):
        new_reprs = []
        # loop over each bucket
        # FIXME (lingfan): handle zero-degree case
        for deg, vv, msg_id in zip(self.degrees, self.dsts, self.msg_ids):
            dst_reprs = self.g.get_n_repr(vv)
            in_msgs = self.msg_frame.select_rows(msg_id)
            def _reshape_fn(msg):
                msg_shape = F.shape(msg)
                new_shape = (len(vv), deg) + msg_shape[1:]
                return F.reshape(msg, new_shape)
            if len(in_msgs) == 1 and __MSG__ in in_msgs:
                reshaped_in_msgs = _reshape_fn(in_msgs[__MSG__])
            else:
                reshaped_in_msgs = utils.LazyDict(
                        lambda key: _reshape_fn(in_msgs[key]), self.msg_frame.schemes)
            new_reprs.append(self.rfunc(dst_reprs, reshaped_in_msgs))

        # Pack all reducer results together
        if utils.is_dict_like(new_reprs[0]):
            keys = new_reprs[0].keys()
            new_reprs = {key : F.pack([repr[key] for repr in new_reprs])
                         for key in keys}
        else:
            new_reprs = {__REPR__ : F.pack(new_reprs)}
        return new_reprs


class BasicExecutor(Executor):
    def __init__(self, graph, mfunc, rfunc):
        self.g = graph
        self.exe = self._build_exec(mfunc, rfunc)

    @property
    def node_repr(self):
        raise NotImplementedError

    @property
    def edge_repr(self):
        raise NotImplementedError

    @property
    def graph_mapping(self):
        raise NotImplementedError

    def _build_exec(self, mfunc, rfunc):
        if isinstance(mfunc, fmsg.CopySrcMessageFunction):
            exe = SPMVOperator(src_field=mfunc.src_field,
                               edge_field=None,
                               dst_field=rfunc.out_field,
                               use_edge_feat=False,
                               node_repr=self.node_repr,
                               adj_build_fn=self._adj_build_fn)
        elif isinstance(mfunc, fmsg.SrcMulEdgeMessageFunction):
            exe = SPMVOperator(src_field=mfunc.src_field,
                               edge_field=mfunc.edge_field,
                               dst_field=rfunc.out_field,
                               use_edge_feat=True,
                               node_repr=self.node_repr,
                               adj_build_fn=self._adj_build_fn)
        else:
            raise NotImplementedError("message func type {}".format(type(mfunc)))
        return exe

    def run(self):
        return self.exe.run()
        # self.g.set_n_repr(attr, self.graph_mapping)


class UpdateAllExecutor(BasicExecutor):
    def __init__(self, graph, mfunc, rfunc):
        self._init_state()
        super(UpdateAllExecutor, self).__init__(graph, mfunc, rfunc)

    def _init_state(self):
        self._node_repr = None
        self._edge_repr = None
        self._graph_idx = None
        self._graph_shape = None
        self._graph_mapping = None

    @property
    def graph_idx(self):
        if self._graph_idx is None:
            self._graph_idx = self.g._graph.adjacency_matrix()
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


class SendRecvExecutor(BasicExecutor):
    def __init__(self, graph, src, dst, mfunc, rfunc):
        self._init_state(src, dst)
        super(SendRecvExecutor, self).__init__(graph, mfunc, rfunc)

    def _init_state(self, src, dst):
        self.u, self.v = utils.edge_broadcasting(src, dst)
        self._node_repr = None
        self._edge_repr = None
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
        u = self.u.tousertensor()
        v = self.v.tousertensor()
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


class BundledExecutor(BasicExecutor):
    """
    Base class for Bundled execution
    All shared structure like graph index should be cached in this class or its subclass
    BundledUpdateAllExecutor and BundledSendRecvExecutor should subclass BundledExecutor
    """
    def __init__(self, graph, mfunc, rfunc):
        self.g = graph
        func_pairs = self._match_message_with_reduce(mfunc, rfunc)
        # create all executors
        self.executors = self._build_executors(func_pairs)

    def _build_executors(self, func_pairs):
        executors = []
        for mfunc, rfunc in func_pairs:
            exe = self._build_exec(mfunc, rfunc)
            executors.append(exe)
        return executors

    def _match_message_with_reduce(self, mfunc, rfunc):
        out2mfunc = {fn.out_field: fn for fn in mfunc.fn_list}
        func_pairs = []
        for rfn in rfunc.fn_list:
            mfn = out2mfunc.get(rfn.msg_field, None)
            # field check
            assert mfn is not None, \
                    "cannot find message func for reduce func in-field {}".format(rfn.msg_field)
            func_pairs.append((mfn, rfn))
        return func_pairs

    def run(self):
        attr = None
        for exe in self.executors:
            res = exe.run()
            if attr is None:
                attr = res
            else:
                # attr and res must be dict
                attr.update(res)
        return attr


class BundledUpdateAllExecutor(BundledExecutor, UpdateAllExecutor):
    def __init__(self, graph, mfunc, rfunc):
        self._init_state()
        BundledExecutor.__init__(self, graph, mfunc, rfunc)


class BundledSendRecvExecutor(BundledExecutor, SendRecvExecutor):
    def __init__(self, graph, src, dst, mfunc, rfunc):
        self._init_state(src, dst)
        BundledExecutor.__init__(self, graph, mfunc, rfunc)

def _is_spmv_supported(fn, graph=None):
    if isinstance(fn, fmsg.MessageFunction):
        return fn.is_spmv_supported(graph)
    elif isinstance(fn, fred.ReduceFunction):
        return fn.is_spmv_supported()
    else:
        return False

def _create_update_all_exec(graph, **kwargs):
    mfunc = kwargs.pop('message_func')
    rfunc = kwargs.pop('reduce_func')
    if isinstance(mfunc, (list, tuple)) or isinstance(rfunc, (list, tuple)):
        mfunc = fmsg.BundledMessageFunction(mfunc)
        rfunc = fred.BundledReduceFunction(rfunc)
        exec_cls = BundledUpdateAllExecutor
    else:
        exec_cls = UpdateAllExecutor
    if _is_spmv_supported(mfunc, graph) and _is_spmv_supported(rfunc):
        return exec_cls(graph, mfunc=mfunc, rfunc=rfunc)
    else:
        return None

def _create_send_and_recv_exec(graph, **kwargs):
    src = kwargs.pop('src')
    dst = kwargs.pop('dst')
    mfunc = kwargs.pop('message_func')
    rfunc = kwargs.pop('reduce_func')
    if isinstance(mfunc, (list, tuple)) or isinstance(rfunc, (list, tuple)):
        mfunc = fmsg.BundledMessageFunction(mfunc)
        rfunc = fred.BundledReduceFunction(rfunc)
        exec_cls = BundledSendRecvExecutor
    else:
        exec_cls = SendRecvExecutor
    if _is_spmv_supported(mfunc, graph) and _is_spmv_supported(rfunc):
        return exec_cls(graph, src=src, dst=dst, mfunc=mfunc, rfunc=rfunc)
    else:
        return None

def get_executor(call_type, graph, **kwargs):
    if call_type == "update_all":
        return _create_update_all_exec(graph, **kwargs)
    elif call_type == "send_and_recv":
        return _create_send_and_recv_exec(graph, **kwargs)
    else:
        return None

def get_recv_executor(graph, reduce_func, message_frame, edges=None):
    """Create executor for recv phase

    Parameters
    ----------
    graph: DGLGraph
        DGLGraph on which to perform recv
    reduce_func: callable
        The reduce function
    message_frame: FrameRef
        Message frame
    edges: tuple/list of utils.Index
        src and dst Index representing edges along which messages are sent
        If not specified, all edges of graph are used instead
    """

    # FIXME: handle builtin spmv executor case
    return DegreeBucketingExecutor(graph, reduce_func, message_frame, edges)

_init_api("dgl.scheduler")
