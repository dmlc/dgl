from __future__ import absolute_import

from .. import backend as F
from ..udf import NodeBatch, EdgeBatch
from .. import utils

__all__ = [
           "SPMVExecutor",
           "DegreeBucketingExecutor",
           "EdgeExecutor",
           "NodeExecutor"
          ]

class Executor(object):
    def run(self):
        raise NotImplementedError

class SPMVExecutor(Executor):
    def __init__(self, src_field, src_repr, out_field, out_repr, adjmat, use_edge_feat=False, edge_field=None, edge_repr=None, dense_shape=None):
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
            self.adjmat = F.sparse_matrix(dat, self.adjmat, self.dense_shape)

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
    def __init__(self, g, rfunc, message_frame, out_repr, buckets, zero_deg_nodes=None, reorder=True):
        self.g = g
        self.rfunc = rfunc
        self.msg_frame = message_frame
        self.degrees, self.dsts, self.msg_ids = buckets
        self.zero_deg_nodes = zero_deg_nodes
        self.reorder = reorder
        self.out_repr = out_repr

    def run(self):
        new_reprs = []
        # loop over each bucket
        # FIXME (lingfan): handle zero-degree case
        for deg, vv, msg_id in zip(self.degrees, self.dsts, self.msg_ids):
            v_data = self.g.get_n_repr(vv)
            in_msgs = self.msg_frame.select_rows(msg_id)
            def _reshape_fn(msg):
                msg_shape = F.shape(msg)
                new_shape = (len(vv), deg) + msg_shape[1:]
                return F.reshape(msg, new_shape)
            reshaped_in_msgs = utils.LazyDict(
                    lambda key: _reshape_fn(in_msgs[key]), self.msg_frame.schemes)
            nb = NodeBatch(self.g, vv, v_data, reshaped_in_msgs)
            new_reprs.append(self.rfunc(nb))

        # Pack all reducer results together
        keys = new_reprs[0].keys()
        new_reprs = {key : F.cat([repr[key] for repr in new_reprs], dim=0)
                     for key in keys}
        self.out_repr.update(new_reprs)

class NodeExecutor(Executor):
    def __init__(self, func, graph, u, out_repr, reduce_accum=None):
        self.func = func
        node_data = graph.get_n_repr(u)
        if reduce_accum:
            node_data = utils.HybridDict(reduce_accum, node_data)
        self.nb = NodeBatch(graph, u, node_data)
        self.out_repr = out_repr

    def run(self):
        self.out_repr.update(self.func(self.nb))

class EdgeExecutor(Executor):
    def __init__(self, func, graph, u, v, eid, out_repr):
        self.func = func
        src_data = graph.get_n_repr(u)
        edge_data = graph.get_e_repr(eid)
        dst_data = graph.get_n_repr(v)
        self.eb = EdgeBatch(graph, (u, v, eid),
                    src_data, edge_data, dst_data)
        self.out_repr = out_repr

    def run(self):
        self.out_repr.update(self.func(self.eb))
