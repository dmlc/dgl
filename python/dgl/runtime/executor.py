"""Module for executor classes."""
from __future__ import absolute_import

from .. import backend as F
from ..frame import Frame, FrameRef
from ..function.base import BuiltinFunction, BundledFunction
from ..udf import NodeBatch, EdgeBatch
from .. import utils

__all__ = [
           "SPMVExecutor",
           "DegreeBucketingExecutor",
           "EdgeExecutor",
           "NodeExecutor",
           "WriteBackExecutor"
          ]

class Executor(object):
    """Base executor class."""
    def run(self):
        """The function to run this executor."""
        raise NotImplementedError

class SPMVExecutor(Executor):
    """Executor to perform SPMV.

    The SPMV is a sparse matrix operation as follows:
    ::

        C := A op B

    , where A is a sparse matrix, B is a dense matrix, C is a dense matrix.
    The op can be any *diverse semi-ring*. However, only (*,+) is supported
    at the moment.

    When running this executor, B's data will be fetched by the given field;
    A is created using the given sparse index and the data fetched by the
    given field; C is then write to the given output field.

    Note that the sparse index of A is not directly given to the constructor
    of this executor because the device context is unknown until the B matrix
    is computed.

    Parameters
    ----------
    A_creator : callable
        The function to create the sparse matrix A. The function should
        comply with following signature:
        def fn(data_store : dict, key : str, ctx : context): -> SparseTensor
    A_store : dict
        The k-v store of A's data.
    A_field : str
        Field name of A's data.
    B_store : dict
        The k-v store of B's data.
    B_field : str
        Field name of B's data.
    C_store : dict
        The k-v store of C's data.
    C_field : str
        Field name of C's data.

    See Also
    --------
    dgl.backend.sparse_matrix
    dgl.backend.sparse_matrix_indices
    """
    def __init__(self, A_creator, A_store, A_field,
                 B_store, B_field, C_store, C_field):
        self.A_creator = A_creator
        self.A_store = A_store
        self.A_field = A_field
        self.B_store = B_store
        self.B_field = B_field
        self.C_store = C_store
        self.C_field = C_field

    def run(self):
        B = self.B_store[self.B_field]
        ctx = F.context(B)
        A = self.A_creator(self.A_store, self.A_field, ctx)
        # spmv
        if F.ndim(B) == 1:
            # B is a vector, append a (1,) dim at the end
            B = F.unsqueeze(B, 1)
            C = F.spmm(A, B)
            C = F.squeeze(C, 1)
        else:
            C = F.spmm(A, B)
        self.C_store[self.C_field] = C

    '''
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
    '''


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
    """Executor to perform apply_nodes.

    Parameters
    ----------
    func : callable, or a list of callable
        The node UDF(s).
    graph : DGLGraph
        The graph.
    u : utils.Index
        The nodes to be applied.
    out_repr : dict
        The dict for writing outputs.
    reduce_accum : dict
        The dict containing reduced results.
    """
    def __init__(self, func, graph, u, out_repr, reduce_accum=None):
        self.func = BundledFunction(func) if utils.is_iterable(func) else func
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
    """Executor to perform edge related computation like send and apply_edges.

    Parameters
    ----------
    func : callable, or a list of callable
        The edge UDF(s).
    graph : DGLGraph
        The graph.
    u : utils.Index
        The src nodes.
    v : utils.Index
        The dst nodes.
    eid : utils.Index
        The edge ids.
    out_repr : dict
        The dict for writing outputs.
    """
    def __init__(self, func, graph, u, v, eid, out_repr):
        self.func = BundledFunction(func) if utils.is_iterable(func) else func
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
