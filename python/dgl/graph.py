"""Base graph class specialized for neural networks on graphs.
"""
from __future__ import absolute_import

import networkx as nx
from networkx.classes.digraph import DiGraph

import dgl
from dgl.base import ALL, is_all
import dgl.backend as F
from dgl.backend import Tensor
import dgl.builtin as builtin
from dgl.cached_graph import CachedGraph, create_cached_graph
import dgl.context as context
from dgl.frame import FrameRef, merge_frames
from dgl.nx_adapt import nx_init
import dgl.scheduler as scheduler
import dgl.utils as utils

__MSG__ = "__MSG__"
__REPR__ = "__REPR__"

class DGLGraph(DiGraph):
    """Base graph class specialized for neural networks on graphs.

    TODO(minjie): document of batching semantics
    TODO(minjie): document of __REPR__ semantics

    Parameters
    ----------
    graph_data : graph data
        Data to initialize graph. Same as networkx's semantics.
    node_frame : dgl.frame.Frame
        Node feature storage.
    edge_frame : dgl.frame.Frame
        Edge feature storage.
    attr : keyword arguments, optional
        Attributes to add to graph as key=value pairs.
    """
    def __init__(self,
                 graph_data=None,
                 node_frame=None,
                 edge_frame=None,
                 **attr):
        # TODO(minjie): maintaining node/edge list is costly when graph is large.
        self._edge_list = []
        nx_init(self,
                self._add_node_callback,
                self._add_edge_callback,
                self._del_node_callback,
                self._del_edge_callback,
                graph_data,
                **attr)
        # cached graph and storage
        self._cached_graph = None
        self._node_frame = node_frame if node_frame is not None else FrameRef()
        self._edge_frame = edge_frame if edge_frame is not None else FrameRef()
        # other class members
        self._msg_graph = None
        self._msg_frame = FrameRef()
        self._message_func = None
        self._reduce_func = None
        self._update_func = None
        self._edge_func = None

    def node_attr_schemes(self):
        return self._node_frame.schemes

    def edge_attr_schemes(self):
        return self._edge_frame.schemes

    def set_n_repr(self, hu, u=ALL):
        """Set node(s) representation.

        To set multiple node representations at once, pass `u` with a tensor or
        a supported container of node ids. In this case, `hu` must be a tensor
        of shape (B, D1, D2, ...), where B is the number of the nodes and
        (D1, D2, ...) is the shape of the node representation tensor.

        Dictionary type is also supported for `hu`. In this case, each item
        will be treated as separate attribute of the nodes.

        Parameters
        ----------
        hu : tensor or dict of tensor
          Node representation.
        u : node, container or tensor
          The node(s).
        """
        # sanity check
        if is_all(u):
            num_nodes = self.number_of_nodes()
        else:
            u = utils.toindex(u)
            num_nodes = len(u)
        if isinstance(hu, dict):
            for key, val in hu.items():
                assert F.shape(val)[0] == num_nodes
        else:
            assert F.shape(hu)[0] == num_nodes
        # set
        if is_all(u):
            if isinstance(hu, dict):
                for key, val in hu.items():
                    self._node_frame[key] = val
            else:
                self._node_frame[__REPR__] = hu
        else:
            if isinstance(hu, dict):
                self._node_frame[u] = hu
            else:
                self._node_frame[u] = {__REPR__ : hu}

    def get_n_repr(self, u=ALL):
        """Get node(s) representation.

        Parameters
        ----------
        u : node, container or tensor
          The node(s).
        """
        if is_all(u):
            if len(self._node_frame) == 1 and __REPR__ in self._node_frame:
                return self._node_frame[__REPR__]
            else:
                return dict(self._node_frame)
        else:
            u = utils.toindex(u)
            if len(self._node_frame) == 1 and __REPR__ in self._node_frame:
                return self._node_frame.select_rows(u)[__REPR__]
            else:
                return self._node_frame.select_rows(u)

    def pop_n_repr(self, key=__REPR__):
        """Get and remove the specified node repr.

        Parameters
        ----------
        key : str
          The attribute name.
        """
        return self._node_frame.pop(key)

    def set_e_repr(self, h_uv, u=ALL, v=ALL):
        """Set edge(s) representation.

        To set multiple edge representations at once, pass `u` and `v` with tensors or
        supported containers of node ids. In this case, `h_uv` must be a tensor
        of shape (B, D1, D2, ...), where B is the number of the edges and
        (D1, D2, ...) is the shape of the edge representation tensor.

        Dictionary type is also supported for `h_uv`. In this case, each item
        will be treated as separate attribute of the edges.

        Parameters
        ----------
        h_uv : tensor or dict of tensor
          Edge representation.
        u : node, container or tensor
          The source node(s).
        v : node, container or tensor
          The destination node(s).
        """
        # sanity check
        u_is_all = is_all(u)
        v_is_all = is_all(v)
        assert u_is_all == v_is_all
        if u_is_all:
            num_edges = self.cached_graph.num_edges()
        else:
            u = utils.toindex(u)
            v = utils.toindex(v)
            num_edges = max(len(u), len(v))
        if isinstance(h_uv, dict):
            for key, val in h_uv.items():
                assert F.shape(val)[0] == num_edges
        else:
            assert F.shape(h_uv)[0] == num_edges
        # set
        if u_is_all:
            if isinstance(h_uv, dict):
                for key, val in h_uv.items():
                    self._edge_frame[key] = val
            else:
                self._edge_frame[__REPR__] = h_uv
        else:
            eid = self.cached_graph.get_edge_id(u, v)
            if isinstance(h_uv, dict):
                self._edge_frame[eid] = h_uv
            else:
                self._edge_frame[eid] = {__REPR__ : h_uv}

    def set_e_repr_by_id(self, h_uv, eid=ALL):
        """Set edge(s) representation by edge id.

        Parameters
        ----------
        h_uv : tensor or dict of tensor
          Edge representation.
        eid : int, container or tensor
          The edge id(s).
        """
        # sanity check
        if is_all(eid):
            num_edges = self.cached_graph.num_edges()
        else:
            eid = utils.toindex(eid)
            num_edges = len(eid)
        if isinstance(h_uv, dict):
            for key, val in h_uv.items():
                assert F.shape(val)[0] == num_edges
        else:
            assert F.shape(h_uv)[0] == num_edges
        # set
        if is_all(eid):
            if isinstance(h_uv, dict):
                for key, val in h_uv.items():
                    self._edge_frame[key] = val
            else:
                self._edge_frame[__REPR__] = h_uv
        else:
            if isinstance(h_uv, dict):
                self._edge_frame[eid] = h_uv
            else:
                self._edge_frame[eid] = {__REPR__ : h_uv}

    def get_e_repr(self, u=ALL, v=ALL):
        """Get node(s) representation.

        Parameters
        ----------
        u : node, container or tensor
          The source node(s).
        v : node, container or tensor
          The destination node(s).
        """
        u_is_all = is_all(u)
        v_is_all = is_all(v)
        assert u_is_all == v_is_all
        if u_is_all:
            if len(self._edge_frame) == 1 and __REPR__ in self._edge_frame:
                return self._edge_frame[__REPR__]
            else:
                return dict(self._edge_frame)
        else:
            u = utils.toindex(u)
            v = utils.toindex(v)
            eid = self.cached_graph.get_edge_id(u, v)
            if len(self._edge_frame) == 1 and __REPR__ in self._edge_frame:
                return self._edge_frame.select_rows(eid)[__REPR__]
            else:
                return self._edge_frame.select_rows(eid)

    def pop_e_repr(self, key=__REPR__):
        """Get and remove the specified edge repr.

        Parameters
        ----------
        key : str
          The attribute name.
        """
        return self._edge_frame.pop(key)

    def get_e_repr_by_id(self, eid=ALL):
        """Get edge(s) representation by edge id.

        Parameters
        ----------
        eid : int, container or tensor
          The edge id(s).
        """
        if is_all(eid):
            if len(self._edge_frame) == 1 and __REPR__ in self._edge_frame:
                return self._edge_frame[__REPR__]
            else:
                return dict(self._edge_frame)
        else:
            eid = utils.toindex(eid)
            if len(self._edge_frame) == 1 and __REPR__ in self._edge_frame:
                return self._edge_frame.select_rows(eid)[__REPR__]
            else:
                return self._edge_frame.select_rows(eid)

    def register_message_func(self,
                              message_func,
                              batchable=False):
        """Register global message function.

        Parameters
        ----------
        message_func : callable
          Message function on the edge.
        batchable : bool
          Whether the provided message function allows batch computing.
        """
        self._message_func = (message_func, batchable)

    def register_edge_func(self,
                           edge_func,
                           batchable=False):
        """Register global edge update function.

        Parameters
        ----------
        edge_func : callable
          Message function on the edge.
        batchable : bool
          Whether the provided message function allows batch computing.
        """
        self._edge_func = (edge_func, batchable)

    def register_reduce_func(self,
                             reduce_func,
                             batchable=False):
        """Register global message reduce function.

        Parameters
        ----------
        reduce_func : str or callable
          Reduce function on incoming edges.
        batchable : bool
          Whether the provided reduce function allows batch computing.
        """
        self._reduce_func = (reduce_func, batchable)

    def register_update_func(self,
                             update_func,
                             batchable=False):
        """Register global node update function.

        Parameters
        ----------
        update_func : callable
          Update function on the node.
        batchable : bool
          Whether the provided update function allows batch computing.
        """
        self._update_func = (update_func, batchable)

    def sendto(self, u, v, message_func=None, batchable=False):
        """Trigger the message function on edge u->v

        The message function should be compatible with following signature:

        (node_reprs, edge_reprs) -> message

        It computes the representation of a message using the
        representations of the source node, and the edge u->v.
        All node_reprs and edge_reprs are dictionaries.
        The message function can be any of the pre-defined functions
        ('from_src').

        Parameters
        ----------
        u : node, container or tensor
          The source node(s).
        v : node, container or tensor
          The destination node(s).
        message_func : str or callable
          The message function.
        batchable : bool
          Whether the function allows batched computation.
        """
        if message_func is None:
            message_func, batchable = self._message_func
        assert message_func is not None
        if batchable:
            self._batch_sendto(u, v, message_func)
        else:
            self._nonbatch_sendto(u, v, message_func)

    def _nonbatch_sendto(self, u, v, message_func):
        f_msg = _get_message_func(message_func)
        if is_all(u) and is_all(v):
            u, v = self.cached_graph.edges()
        else:
            u = utils.toindex(u)
            v = utils.toindex(v)
        for uu, vv in utils.edge_iter(u, v):
            ret = f_msg(_get_repr(self.nodes[uu]),
                        _get_repr(self.edges[uu, vv]))
            self.edges[uu, vv][__MSG__] = ret

    def _batch_sendto(self, u, v, message_func):
        f_msg = _get_message_func(message_func)
        if is_all(u) and is_all(v):
            u, v = self.cached_graph.edges()
            self.msg_graph.add_edges(u, v)
            # call UDF
            src_reprs = self.get_n_repr(u)
            edge_reprs = self.get_e_repr()
            msgs = message_func(src_reprs, edge_reprs)
        else:
            u = utils.toindex(u)
            v = utils.toindex(v)
            u, v = utils.edge_broadcasting(u, v)
            eid = self.cached_graph.get_edge_id(u, v)
            self.msg_graph.add_edges(u, v)
            # call UDF
            src_reprs = self.get_n_repr(u)
            edge_reprs = self.get_e_repr_by_id(eid)
            msgs = message_func(src_reprs, edge_reprs)
        if isinstance(msgs, dict):
            self._msg_frame.append(msgs)
        else:
            self._msg_frame.append({__MSG__ : msgs})

    def update_edge(self, u, v, edge_func=None, batchable=False):
        """Update representation on edge u->v

        The edge function should be compatible with following signature:

        (node_reprs, node_reprs, edge_reprs) -> edge_reprs

        It computes the new edge representations using the representations
        of the source node, target node and the edge itself.
        All node_reprs and edge_reprs are dictionaries.

        Parameters
        ----------
        u : node, container or tensor
          The source node(s).
        v : node, container or tensor
          The destination node(s).
        edge_func : str or callable
          The update function.
        batchable : bool
          Whether the function allows batched computation.
        """
        if edge_func is None:
            edge_func, batchable = self._edge_func
        assert edge_func is not None
        if batchable:
            self._batch_update_edge(u, v, edge_func)
        else:
            self._nonbatch_update_edge(u, v, edge_func)

    def _nonbatch_update_edge(self, u, v, edge_func):
        if is_all(u) and is_all(v):
            u, v = self.cached_graph.edges()
        else:
            u = utils.toindex(u)
            v = utils.toindex(v)
        for uu, vv in utils.edge_iter(u, v):
            ret = edge_func(_get_repr(self.nodes[uu]),
                            _get_repr(self.nodes[vv]),
                            _get_repr(self.edges[uu, vv]))
            _set_repr(self.edges[uu, vv], ret)

    def _batch_update_edge(self, u, v, edge_func):
        if is_all(u) and is_all(v):
            u, v = self.cached_graph.edges()
            # call the UDF
            src_reprs = self.get_n_repr(u)
            dst_reprs = self.get_n_repr(v)
            edge_reprs = self.get_e_repr()
            new_edge_reprs = edge_func(src_reprs, dst_reprs, edge_reprs)
            self.set_e_repr(new_edge_reprs)
        else:
            u = utils.toindex(u)
            v = utils.toindex(v)
            u, v = utils.edge_broadcasting(u, v)
            eid = self.cached_graph.get_edge_id(u, v)
            # call the UDF
            src_reprs = self.get_n_repr(u)
            dst_reprs = self.get_n_repr(v)
            edge_reprs = self.get_e_repr_by_id(eid)
            new_edge_reprs = edge_func(src_reprs, dst_reprs, edge_reprs)
            self.set_e_repr_by_id(new_edge_reprs, eid)

    def recv(self,
             u,
             reduce_func=None,
             update_func=None,
             batchable=False):
        """Receive in-coming messages and update representation on node u.

        It computes the new node state using the messages sent from the predecessors
        of node u. If no message is found from the predecessors, reduce function
        will be skipped and a None type will be provided as the reduced messages for
        the update function.

        The reduce function should be compatible with following signature:

            (node_reprs, batched_messages) -> reduced_messages

        It computes the reduced edge representations using the representations
        of the in-coming edges (the same concept as messages).
        The reduce function can be any of the pre-defined functions ('sum',
        'max'). If built-in function is used, computation will be performed
        efficiently (using generic-SPMV kernels).

        The update function should be compatible with following signature:

            (node_reprs, reduced_messages) -> node_reprs

        It computes the new node representations using the representations
        of the in-coming edges (the same concept as messages) and the node
        itself. All node_reprs and edge_reprs are dictionaries.

        Parameters
        ----------
        u : node, container or tensor
          The node to be updated.
        reduce_func : str or callable
          The reduce function.
        update_func : str or callable
          The update function.
        batchable : bool
          Whether the reduce and update function allows batched computation.
        """
        if reduce_func is None:
            reduce_func, batchable = self._reduce_func
        if update_func is None:
            update_func, batchable = self._update_func
        assert reduce_func is not None
        assert update_func is not None
        if batchable:
            self._batch_recv(u, reduce_func, update_func)
        else:
            self._nonbatch_recv(u, reduce_func, update_func)

    def _nonbatch_recv(self, u, reduce_func, update_func):
        f_reduce = _get_reduce_func(reduce_func)
        f_update = update_func
        if is_all(u):
            u = list(range(0, self.number_of_nodes()))
        else:
            u = utils.toindex(u)
        for i, uu in enumerate(utils.node_iter(u)):
            # reduce phase
            msgs_batch = [self.edges[vv, uu].pop(__MSG__)
                          for vv in self.pred[uu] if __MSG__ in self.edges[vv, uu]]
            if len(msgs_batch) == 0:
                msgs_reduced = None
            else:
                msgs_reduced = f_reduce(_get_repr(self.nodes[uu]), msgs_batch)
            # update phase
            ret = f_update(_get_repr(self.nodes[uu]), msgs_reduced)
            _set_repr(self.nodes[uu], ret)

    def _batch_recv(self, v, reduce_func, update_func):
        f_update = update_func
        null_v, reordered_v, all_reduced_msgs = self._batch_reduce(v, reduce_func)
        if all_reduced_msgs is None:
            # no message; only do recv.
            if is_all(v):
                self.set_n_repr(f_update(self.get_n_repr(), None))
            else:
                self.set_n_repr(f_update(self.get_n_repr(v), None), v)
        else:
            # Read the node states in the degree-bucketing order.
            if len(null_v) == 0:
                null_ns = new_null_ns = None
            else:
                null_ns = self.get_n_repr(null_v)
                new_null_ns = f_update(null_ns, None)
            if len(reordered_v) == 0:
                reordered_ns = new_reordered_ns = None
            else:
                reordered_ns = self.get_n_repr(reordered_v)
                new_reordered_ns = f_update(reordered_ns, all_reduced_msgs)

            v_tensor = utils.pack2(null_v.totensor(), reordered_v.totensor())
            new_ns = utils.pack2(new_null_ns, new_reordered_ns)

            if is_all(v):
                # First do reorder and then replace the whole column.
                _, indices = F.sort(v_tensor)
                indices = utils.toindex(indices)
                # TODO(minjie): following code should be included in Frame somehow.
                if isinstance(new_ns, dict):
                    for key, val in new_ns.items():
                        idx = indices.totensor(F.get_context(val))
                        self._node_frame[key] = F.gather_row(val, idx)
                else:
                    idx = indices.totensor(F.get_context(new_ns))
                    self._node_frame[__REPR__] = F.gather_row(new_ns, idx)
            else:
                # Use setter to do reorder.
                self.set_n_repr(new_ns, v_tensor)

    def _batch_reduce(self, v, reduce_func):
        if is_all(v) and len(self._msg_frame) == 0:
            # no message has been sent
            return None, None, None

        if is_all(v):
            v = list(range(self.number_of_nodes()))

        # freeze message graph
        self.msg_graph.freeze()

        # sanity checks
        v = utils.toindex(v)
        f_reduce = _get_reduce_func(reduce_func)

        # degree bucketing
        degrees, v_buckets = scheduler.degree_bucketing(self.msg_graph, v)
        null_v_bucket = None
        non_null_v_buckets = []
        reduced_msgs = []
        for deg, v_bkt in zip(degrees, v_buckets):
            bkt_len = len(v_bkt)
            dst_reprs = self.get_n_repr(v_bkt)

            if deg == 0:
                assert null_v_bucket is None
                null_v_bucket = v_bkt
                continue
                
            uu, vv = self.msg_graph.in_edges(v_bkt)
            in_msg_ids = self.msg_graph.get_edge_id(uu, vv)
            in_msgs = self._msg_frame.select_rows(in_msg_ids)
            # Reshape the column tensor to (B, Deg, ...).
            def _reshape_fn(msg):
                msg_shape = F.shape(msg)
                new_shape = (bkt_len, deg) + msg_shape[1:]
                return F.reshape(msg, new_shape)
            if len(in_msgs) == 1 and __MSG__ in in_msgs:
                reshaped_in_msgs = _reshape_fn(in_msgs[__MSG__])
            else:
                reshaped_in_msgs = utils.LazyDict(
                        lambda key: _reshape_fn(in_msgs[key]), self._msg_frame.schemes)
            non_null_v_buckets.append(v_bkt)
            reduced_msgs.append(f_reduce(dst_reprs, reshaped_in_msgs))

        # TODO: clear partial messages
        self.clear_messages()

        # FIXME: this will only trigger if reduced_msgs is empty.  Remove?
        if len(reduced_msgs) == 0:
            # no message has been sent to the specified node
            return None, None, None
        
        # Read the node states in the degree-bucketing order.
        null_v = utils.toindex(null_v_bucket or [])
        reordered_v = utils.toindex(
                F.pack([v_bkt.totensor() for v_bkt in non_null_v_buckets])
                if len(non_null_v_buckets) > 0 else []
                )

        # Pack all reduced msgs together
        if isinstance(reduced_msgs[0], dict):
            keys = reduced_msgs[0].keys()
            all_reduced_msgs = {
                    key : F.pack([msg[key] for msg in reduced_msgs])
                    for key in keys}
        else:
            all_reduced_msgs = F.pack(reduced_msgs)

        return null_v, reordered_v, all_reduced_msgs

    def update_by_edge(self,
                       u, v,
                       message_func=None,
                       reduce_func=None,
                       update_func=None,
                       batchable=False):
        """Trigger the message function on u->v and update v.

        Parameters
        ----------
        u : node, container or tensor
          The source node(s).
        v : node, container or tensor
          The destination node(s).
        message_func : str or callable
          The message function.
        reduce_func : str or callable
          The reduce function.
        update_func : str or callable
          The update function.
        batchable : bool
          Whether the reduce and update function allows batched computation.
        """
        if message_func is None:
            message_func, batchable = self._message_func
        if reduce_func is None:
            reduce_func, batchable = self._reduce_func
        if update_func is None:
            update_func, batchable = self._update_func
        assert message_func is not None
        assert reduce_func is not None
        assert update_func is not None
        if batchable:
            self._batch_update_by_edge(
                    u, v, message_func, reduce_func, update_func)
        else:
            self._nonbatch_update_by_edge(
                    u, v, message_func, reduce_func, update_func)

    def _nonbatch_update_by_edge(
            self,
            u, v,
            message_func,
            reduce_func,
            update_func):
        if is_all(u) and is_all(v):
            u, v = self.cached_graph.edges()
        else:
            u = utils.toindex(u)
            v = utils.toindex(v)
        self._nonbatch_sendto(u, v, message_func)
        dst = set()
        for uu, vv in utils.edge_iter(u, v):
            dst.add(vv)
        self._nonbatch_recv(list(dst), reduce_func, update_func)

    def _batch_update_by_edge(
            self,
            u, v,
            message_func,
            reduce_func,
            update_func):
        if is_all(u) and is_all(v):
            self.update_all(message_func, reduce_func, update_func, True)
        elif message_func == 'from_src' and reduce_func == 'sum':
            # TODO(minjie): check the validity of edges u->v
            u = utils.toindex(u)
            v = utils.toindex(v)
            # TODO(minjie): broadcasting is optional for many-one input.
            u, v = utils.edge_broadcasting(u, v)
            # relabel destination nodes.
            new2old, old2new = utils.build_relabel_map(v)
            u = u.totensor()
            v = v.totensor()
            # TODO(minjie): should not directly use []
            new_v = old2new[v]
            # create adj mat
            idx = F.pack([F.unsqueeze(new_v, 0), F.unsqueeze(u, 0)])
            dat = F.ones((len(u),))
            n = self.number_of_nodes()
            m = len(new2old)
            # TODO(minjie): context
            adjmat = F.sparse_tensor(idx, dat, [m, n])
            ctx_adjmat = utils.CtxCachedObject(lambda ctx: F.to_context(adjmat, ctx))
            # TODO(minjie): use lazy dict for reduced_msgs
            reduced_msgs = {}
            for key in self._node_frame.schemes:
                col = self._node_frame[key]
                reduced_msgs[key] = F.spmm(ctx_adjmat.get(F.get_context(col)), col)
            if len(reduced_msgs) == 1 and __REPR__ in reduced_msgs:
                reduced_msgs = reduced_msgs[__REPR__]
            node_repr = self.get_n_repr(new2old)
            new_node_repr = update_func(node_repr, reduced_msgs)
            self.set_n_repr(new_node_repr, new2old)
        else:
            u = utils.toindex(u)
            v = utils.toindex(v)
            self._batch_sendto(u, v, message_func)
            unique_v = F.unique(v.totensor())
            self._batch_recv(unique_v, reduce_func, update_func)

    def update_to(self,
                  v,
                  message_func=None,
                  reduce_func=None,
                  update_func=None,
                  batchable=False):
        """Pull messages from the node's predecessors and then update it.

        Parameters
        ----------
        v : node, container or tensor
          The node to be updated.
        message_func : str or callable
          The message function.
        reduce_func : str or callable
          The reduce function.
        update_func : str or callable
          The update function.
        batchable : bool
          Whether the reduce and update function allows batched computation.
        """
        if message_func is None:
            message_func, batchable = self._message_func
        if reduce_func is None:
            reduce_func, batchable = self._reduce_func
        if update_func is None:
            update_func, batchable = self._update_func
        assert message_func is not None
        assert reduce_func is not None
        assert update_func is not None
        if batchable:
            v = utils.toindex(v)
            uu, vv = self.cached_graph.in_edges(v)
            
            if len(uu) == 0:
                # no send, just do a recv
                self._batch_recv(v, reduce_func, update_func)
            else:
                self._batch_update_by_edge(uu, vv, message_func,
                        reduce_func, update_func)
        else:
            v = utils.toindex(v)
            for vv in utils.node_iter(v):
                assert vv in self.nodes
                uu = list(self.pred[vv])
                if len(uu) > 0:
                    self._nonbatch_sendto(uu, vv, message_func)
                self._nonbatch_recv(vv, reduce_func, update_func)

    def update_from(self,
                    u,
                    message_func=None,
                    reduce_func=None,
                    update_func=None,
                    batchable=False):
        """Send message from the node to its successors and update them.

        Parameters
        ----------
        u : node, container or tensor
          The node that sends out messages.
        message_func : str or callable
          The message function.
        reduce_func : str or callable
          The reduce function.
        update_func : str or callable
          The update function.
        batchable : bool
          Whether the reduce and update function allows batched computation.
        """
        if message_func is None:
            message_func, batchable = self._message_func
        if reduce_func is None:
            reduce_func, batchable = self._reduce_func
        if update_func is None:
            update_func, batchable = self._update_func
        assert message_func is not None
        assert reduce_func is not None
        assert update_func is not None
        if batchable:
            u = utils.toindex(u)
            uu, vv = self.cached_graph.out_edges(u)
            self._batch_update_by_edge(uu, vv, message_func,
                    reduce_func, update_func)
        else:
            u = utils.toindex(u)
            for uu in utils.node_iter(u):
                assert uu in self.nodes
                for v in self.succ[uu]:
                    self._nonbatch_update_by_edge(uu, v,
                            message_func, reduce_func, update_func)

    def update_all(self,
                   message_func=None,
                   reduce_func=None,
                   update_func=None,
                   batchable=False):
        """Send messages through all the edges and update all nodes.

        Parameters
        ----------
        message_func : str or callable
          The message function.
        reduce_func : str or callable
          The reduce function.
        update_func : str or callable
          The update function.
        batchable : bool
          Whether the reduce and update function allows batched computation.
        """
        if message_func is None:
            message_func, batchable = self._message_func
        if reduce_func is None:
            reduce_func, batchable = self._reduce_func
        if update_func is None:
            update_func, batchable = self._update_func
        assert message_func is not None
        assert reduce_func is not None
        assert update_func is not None
        if batchable:
            if message_func == 'from_src' and reduce_func == 'sum':
                # TODO(minjie): use lazy dict for reduced_msgs
                reduced_msgs = {}
                for key in self._node_frame.schemes:
                    col = self._node_frame[key]
                    adjmat = self.cached_graph.adjmat(F.get_context(col))
                    reduced_msgs[key] = F.spmm(adjmat, col)
                if len(reduced_msgs) == 1 and __REPR__ in reduced_msgs:
                    reduced_msgs = reduced_msgs[__REPR__]
                node_repr = self.get_n_repr()
                self.set_n_repr(update_func(node_repr, reduced_msgs))
            else:
                self._batch_sendto(ALL, ALL, message_func)
                self._batch_recv(ALL, reduce_func, update_func)
        else:
            u, v = zip(*self.edges)
            u = list(u)
            v = list(v)
            self._nonbatch_sendto(u, v, message_func)
            self._nonbatch_recv(list(self.nodes()), reduce_func, update_func)

    def propagate(self,
                  message_func=None,
                  reduce_func=None,
                  update_func=None,
                  batchable=False,
                  iterator='bfs',
                  **kwargs):
        """Propagate messages and update nodes using iterator.

        A convenient function for passing messages and updating
        nodes according to the iterator. The iterator can be
        any of the pre-defined iterators ('bfs', 'dfs', 'pre-order',
        'mid-order', 'post-order'). The computation will be unrolled
        in the backend efficiently. User can also provide custom
        iterator that generates the edges and nodes.

        Parameters
        ----------
        message_func : str or callable
          The message function.
        reduce_func : str or callable
          The reduce function.
        update_func : str or callable
          The update function.
        batchable : bool
          Whether the reduce and update function allows batched computation.
        iterator : str or generator of steps.
          The iterator of the graph.
        kwargs : keyword arguments, optional
            Arguments for pre-defined iterators.
        """
        if isinstance(iterator, str):
            # TODO Call pre-defined routine to unroll the computation.
            raise RuntimeError('Not implemented.')
        else:
            # NOTE: the iteration can return multiple edges at each step.
            for u, v in iterator:
                self.update_by_edge(u, v,
                        message_func, reduce_func, update_func, batchable)

    def subgraph(self, nodes):
        """Generate the subgraph among the given nodes.

        The generated graph contains only the graph structure. The node/edge
        features are not shared implicitly. Use `copy_from` to get node/edge
        features from parent graph.

        Parameters
        ----------
        nodes : list, or iterable
            A container of the nodes to construct subgraph.

        Returns
        -------
        G : DGLSubGraph
            The subgraph.
        """
        return dgl.DGLSubGraph(self, nodes)

    def merge(self, subgraphs, reduce_func='sum'):
        """Merge subgraph features back to this parent graph.

        Parameters
        ----------
        subgraphs : iterator of DGLSubGraph
            The subgraphs to be merged.
        reduce_func : str
            The reduce function (only 'sum' is supported currently)
        """
        # sanity check: all the subgraphs and the parent graph
        # should have the same node/edge feature schemes.
        # merge node features
        to_merge = []
        for sg in subgraphs:
            if len(sg.node_attr_schemes()) == 0:
                continue
            if sg.node_attr_schemes() != self.node_attr_schemes():
                raise RuntimeError('Subgraph and parent graph do not '
                                   'have the same node attribute schemes.')
            to_merge.append(sg)
        self._node_frame = merge_frames(
                [sg._node_frame for sg in to_merge],
                [sg._parent_nid for sg in to_merge],
                self._node_frame.num_rows,
                reduce_func)

        # merge edge features
        to_merge.clear()
        for sg in subgraphs:
            if len(sg.edge_attr_schemes()) == 0:
                continue
            if sg.edge_attr_schemes() != self.edge_attr_schemes():
                raise RuntimeError('Subgraph and parent graph do not '
                                   'have the same edge attribute schemes.')
            to_merge.append(sg)
        self._edge_frame = merge_frames(
                [sg._edge_frame for sg in to_merge],
                [sg._parent_eid for sg in to_merge],
                self._edge_frame.num_rows,
                reduce_func)

    def draw(self):
        """Plot the graph using dot."""
        from networkx.drawing.nx_agraph import graphviz_layout

        pos = graphviz_layout(self, prog='dot')
        nx.draw(self, pos, with_labels=True)

    @property
    def cached_graph(self):
        # TODO: dirty flag when mutated
        if self._cached_graph is None:
            self._cached_graph = create_cached_graph(self)
        return self._cached_graph

    @property
    def msg_graph(self):
        # TODO: dirty flag when mutated
        if self._msg_graph is None:
            self._msg_graph = CachedGraph()
            self._msg_graph.add_nodes(self.number_of_nodes())
        return self._msg_graph

    def clear_messages(self):
        if self._msg_graph is not None:
            self._msg_graph = CachedGraph()
            self._msg_graph.add_nodes(self.number_of_nodes())
            self._msg_frame.clear()

    @property
    def edge_list(self):
        """Return edges in the addition order."""
        return self._edge_list

    def get_edge_id(self, u, v):
        """Return the continuous edge id(s) assigned.

        Parameters
        ----------
        u : node, container or tensor
          The source node(s).
        v : node, container or tensor
          The destination node(s).

        Returns
        -------
        eid : tensor
          The tensor contains edge id(s).
        """
        u = utils.toindex(u)
        v = utils.toindex(v)
        return self.cached_graph.get_edge_id(u, v)

    def _add_node_callback(self, node):
        #print('New node:', node)
        self._cached_graph = None

    def _del_node_callback(self, node):
        #print('Del node:', node)
        raise RuntimeError('Node removal is not supported currently.')
        node = utils.convert_to_id_tensor(node)
        self._node_frame.delete_rows(node)
        self._cached_graph = None

    def _add_edge_callback(self, u, v):
        #print('New edge:', u, v)
        self._edge_list.append((u, v))
        self._cached_graph = None

    def _del_edge_callback(self, u, v):
        #print('Del edge:', u, v)
        raise RuntimeError('Edge removal is not supported currently.')
        u = utils.convert_to_id_tensor(u)
        v = utils.convert_to_id_tensor(v)
        eid = self.get_edge_id(u, v)
        self._edge_frame.delete_rows(eid)
        self._cached_graph = None

def _get_repr(attr_dict):
    if len(attr_dict) == 1 and __REPR__ in attr_dict:
        return attr_dict[__REPR__]
    else:
        return attr_dict

def _set_repr(attr_dict, attr):
    if isinstance(attr, dict):
        attr_dict.update(attr)
    else:
        attr_dict[__REPR__] = attr

def _get_reduce_func(reduce_func):
    if isinstance(reduce_func, str):
        # built-in reduce func
        if reduce_func == 'sum':
            return builtin.reduce_sum
        elif reduce_func == 'max':
            return builtin.reduce_max
        else:
            raise ValueError(
                    "Unknown built-in reduce function: %s" % reduce_func)
    return reduce_func

def _get_message_func(message_func):
    if isinstance(message_func, str):
        # built-in message func
        if message_func == 'from_src':
            return builtin.message_from_src
        else:
            raise ValueError(
                    "Unknown built-in message function: %s" % message_func)
    return message_func
