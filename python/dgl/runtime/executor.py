from __future__ import absolute_import

from .. import backend as F
from ..udf import NodeBatch, EdgeBatch
from .. import utils
from ..frame import Frame, FrameRef

__all__ = [
           "SPMVExecutor",
           "DegreeBucketingExecutor",
           "EdgeExecutor",
           "NodeExecutor",
           "WriteBackExecutor"
          ]

class Executor(object):
    def run(self):
        raise NotImplementedError

class SPMVExecutor(Executor):
    """Executor to perform SPMV for one field"""
    def __init__(self, src_field, src_repr, out_field, out_repr, adjmat,
                 use_edge_feat=False, edge_field=None, edge_repr=None,
                 dense_shape=None):
        self.src_field = src_field
        self.src_repr = src_repr
        self.out_field = out_field
        self.out_repr = out_repr
        self.use_edge_feat = use_edge_feat
        self.adjmat = adjmat
        if self.use_edge_feat:
            self.edge_field = edge_field
            self.edge_repr = edge_repr
            self.dense_shape = dense_shape

    def run(self):
        # get src col
        srccol = self.src_repr[self.src_field]

        # move to context
        self.adjmat = self.adjmat.get(F.context(srccol))

        # build adjmat
        if self.use_edge_feat:
            dat = self.edge_repr[self.edge_field]
            if len(F.shape(dat)) > 1:
                # The edge feature is of shape (N, 1)
                dat = F.squeeze(dat, 1)
            self.adjmat = F.sparse_matrix(dat, ('coo', self.adjmat),
                                          self.dense_shape)

        # spmm
        if len(F.shape(srccol)) == 1:
            srccol = F.unsqueeze(srccol, 1)
            dstcol = F.spmm(self.adjmat, srccol)
            dstcol = F.squeeze(dstcol, 1)
        else:
            dstcol = F.spmm(self.adjmat, srccol)

        # update repr
        self.out_repr[self.out_field] = dstcol

class DegreeBucketingExecutor(Executor):
    """Executor for degree bucketing schedule"""
    def __init__(self, g, rfunc, message_frame, out_repr, buckets, reorder=True):
        self.g = g
        self.rfunc = rfunc
        self.msg_frame = message_frame
        self.v, self.degs, self.dsts, self.msg_ids, self.zero_deg_nodes = buckets
        self.reorder = reorder
        self.out_repr = out_repr

    def run(self):
        if isinstance(self.msg_frame, dict):
            # only at this point we are should messages have been materialized
            self.msg_frame = FrameRef(Frame(self.msg_frame))
        new_repr = []
        # loop over each bucket
        for deg, vv, msg_id in zip(self.degs, self.dsts, self.msg_ids):
            v_data = self.g.get_n_repr(vv)
            in_msgs = self.msg_frame.select_rows(msg_id)
            def _reshape_fn(msg):
                msg_shape = F.shape(msg)
                new_shape = (len(vv), deg) + msg_shape[1:]
                return F.reshape(msg, new_shape)
            reshaped_in_msgs = utils.LazyDict(
                    lambda key: _reshape_fn(in_msgs[key]),
                                self.msg_frame.schemes)
            nb = NodeBatch(self.g, vv, v_data, reshaped_in_msgs)
            new_repr.append(self.rfunc(nb))

        # Pack all reducer results together
        keys = new_repr[0].keys()
        if self.zero_deg_nodes:
            new_repr.append(self.g.get_n_repr(self.zero_deg_nodes))
        new_repr = {key : F.cat([repr[key] for repr in new_repr], dim=0)
                     for key in keys}
        if self.reorder:
            _, indices = F.sort_1d(self.v)
            indices = utils.toindex(indices)
            new_repr = utils.reorder(new_repr, indices)
        self.out_repr.update(new_repr)

class NodeExecutor(Executor):
    """Executor to perform apply_nodes"""
    def __init__(self, func, graph, u, out_repr, reduce_accum=None):
        self.func = func
        self.graph = graph
        self.u = u
        self.out_repr = out_repr
        self.reduce_accum = reduce_accum

    def run(self):
        node_data = self.graph.get_n_repr(self.u)
        if self.reduce_accum:
            node_data = utils.HybridDict(self.reduce_accum, node_data)
        nb = NodeBatch(self.graph, self.u, node_data)
        self.out_repr.update(self.func(nb))

class EdgeExecutor(Executor):
    """Executor to perform edge related computation like send and apply_edges"""
    def __init__(self, func, graph, u, v, eid, out_repr):
        self.func = func
        self.graph = graph
        self.u = u
        self.v = v
        self.eid = eid
        self.out_repr = out_repr

    def run(self):
        src_data = self.graph.get_n_repr(self.u)
        edge_data = self.graph.get_e_repr(self.eid)
        dst_data = self.graph.get_n_repr(self.v)
        eb = EdgeBatch(self.graph, (self.u, self.v, self.eid),
                    src_data, edge_data, dst_data)
        self.out_repr.update(self.func(eb))

class WriteBackExecutor(Executor):
    """Executor to write results back to frame storage"""
    def __init__(self, graph, new_repr, ids, target):
        self.graph = graph
        self.new_repr = new_repr
        self.ids = ids
        self.target = target

    def run(self):
        if self.target == "node":
            self.graph.set_n_repr(self.new_repr, self.ids)
        elif self.target == "edge":
            self.graph.set_e_repr(self.new_repr, self.ids)
        elif self.target == "message":
            self.graph._msg_frame.append(self.new_repr)
        else:
            raise RuntimeError("Write back target %s not supported."
                               % self.target)

