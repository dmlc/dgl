"""Base graph class specialized for neural networks on graphs.
"""
from __future__ import absolute_import

from collections import MutableMapping
import networkx as nx
from networkx.classes.digraph import DiGraph

from dgl.base import ALL, is_all
import dgl.backend as F
from dgl.backend import Tensor
import dgl.builtin as builtin
from dgl.cached_graph import CachedGraph, create_cached_graph
import dgl.context as context
from dgl.frame import Frame
import dgl.scheduler as scheduler
import dgl.utils as utils

__MSG__ = "__MSG__"
__REPR__ = "__REPR__"

class _NodeDict(MutableMapping):
    def __init__(self, cb):
        self._dict = {}
        self._cb = cb
    def __setitem__(self, key, val):
        if isinstance(val, _AdjInnerDict):
            # This node dict is used as adj_outer_list
            val.src = key
        elif key not in self._dict:
            self._cb(key)
        self._dict[key] = val
    def __getitem__(self, key):
        return self._dict[key]
    def __delitem__(self, key):
        # FIXME: add callback
        del self._dict[key]
    def __len__(self):
        return len(self._dict)
    def __iter__(self):
        return iter(self._dict)

class _AdjInnerDict(MutableMapping):
    def __init__(self, cb):
        self._dict = {}
        self.src = None
        self._cb = cb
    def __setitem__(self, key, val):
        if key not in self._dict:
            self._cb(self.src, key)
        self._dict[key] = val
    def __getitem__(self, key):
        return self._dict[key]
    def __delitem__(self, key):
        # FIXME: add callback
        del self._dict[key]
    def __len__(self):
        return len(self._dict)
    def __iter__(self):
        return iter(self._dict)

class DGLGraph(DiGraph):
    """Base graph class specialized for neural networks on graphs.

    TODO(minjie): document of batching semantics
    TODO(minjie): document of __REPR__ semantics

    Parameters
    ----------
    data : graph data
        Data to initialize graph. Same as networkx's semantics.
    attr : keyword arguments, optional
        Attributes to add to graph as key=value pairs.
    """
    def __init__(self, graph_data=None, **attr):
        # setup dict overlay
        self.node_dict_factory = lambda : _NodeDict(self._add_node_callback)
        # In networkx 2.1, DiGraph is not using this factory. Instead, the outer
        # dict uses the same data structure as the node dict.
        self.adjlist_outer_dict_factory = None
        self.adjlist_inner_dict_factory = lambda : _AdjInnerDict(self._add_edge_callback)
        self.edge_attr_dict_factory = dict
        self._context = context.cpu()
        # call base class init
        super(DGLGraph, self).__init__(graph_data, **attr)
        self._init_state()

    def _init_state(self):
        # cached graph and storage
        self._cached_graph = None
        self._node_frame = Frame()
        self._edge_frame = Frame()
        # other class members
        self._msg_graph = None
        self._msg_frame = Frame()
        self._message_func = None
        self._reduce_func = None
        self._update_func = None
        self._edge_func = None
        self._edge_cb_state = True
        self._edge_list = []

    def clear(self):
        super(DGLGraph, self).clear()
        self._init_state()

    def get_n_attr_list(self):
        return self._node_frame.schemes

    def get_e_attr_list(self):
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
        if isinstance(u, str) and u == ALL:
            num_nodes = self.number_of_nodes()
        else:
            u = utils.convert_to_id_tensor(u, self.context)
            num_nodes = len(u)
        if isinstance(hu, dict):
            for key, val in hu.items():
                assert F.shape(val)[0] == num_nodes
        else:
            assert F.shape(hu)[0] == num_nodes
        # set
        if isinstance(u, str) and u == ALL:
            if isinstance(hu, dict):
                for key, val in hu.items():
                    self._node_frame[key] = val
            else:
                self._node_frame[__REPR__] = hu
        else:
            if isinstance(hu, dict):
                for key, val in hu.items():
                    self._node_frame[key] = F.scatter_row(self._node_frame[key], u, val)
            else:
                self._node_frame[__REPR__] = F.scatter_row(self._node_frame[__REPR__], u, hu)

    def get_n_repr(self, u=ALL):
        """Get node(s) representation.

        Parameters
        ----------
        u : node, container or tensor
          The node(s).
        """
        if isinstance(u, str) and u == ALL:
            if len(self._node_frame) == 1 and __REPR__ in self._node_frame:
                return self._node_frame[__REPR__]
            else:
                return dict(self._node_frame)
        else:
            u = utils.convert_to_id_tensor(u, self.context)
            if len(self._node_frame) == 1 and __REPR__ in self._node_frame:
                return self._node_frame[__REPR__][u]
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
        u_is_all = isinstance(u, str) and u == ALL
        v_is_all = isinstance(v, str) and v == ALL
        assert u_is_all == v_is_all
        if u_is_all:
            num_edges = self.number_of_edges()
        else:
            u = utils.convert_to_id_tensor(u, self.context)
            v = utils.convert_to_id_tensor(v, self.context)
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
                for key, val in h_uv.items():
                    self._edge_frame[key] = F.scatter_row(self._edge_frame[key], eid, val)
            else:
                self._edge_frame[__REPR__] = F.scatter_row(self._edge_frame[__REPR__], eid, h_uv)

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
        if isinstance(eid, str) and eid == ALL:
            num_edges = self.number_of_edges()
        else:
            eid = utils.convert_to_id_tensor(eid, self.context)
            num_edges = len(eid)
        if isinstance(h_uv, dict):
            for key, val in h_uv.items():
                assert F.shape(val)[0] == num_edges
        else:
            assert F.shape(h_uv)[0] == num_edges
        # set
        if isinstance(eid, str) and eid == ALL:
            if isinstance(h_uv, dict):
                for key, val in h_uv.items():
                    self._edge_frame[key] = val
            else:
                self._edge_frame[__REPR__] = h_uv
        else:
            if isinstance(h_uv, dict):
                for key, val in h_uv.items():
                    self._edge_frame[key] = F.scatter_row(self._edge_frame[key], eid, val)
            else:
                self._edge_frame[__REPR__] = F.scatter_row(self._edge_frame[__REPR__], eid, h_uv)

    def get_e_repr(self, u=ALL, v=ALL):
        """Get node(s) representation.

        Parameters
        ----------
        u : node, container or tensor
          The source node(s).
        v : node, container or tensor
          The destination node(s).
        """
        u_is_all = isinstance(u, str) and u == ALL
        v_is_all = isinstance(v, str) and v == ALL
        assert u_is_all == v_is_all
        if u_is_all:
            if len(self._edge_frame) == 1 and __REPR__ in self._edge_frame:
                return self._edge_frame[__REPR__]
            else:
                return dict(self._edge_frame)
        else:
            u = utils.convert_to_id_tensor(u, self.context)
            v = utils.convert_to_id_tensor(v, self.context)
            eid = self.cached_graph.get_edge_id(u, v)
            if len(self._edge_frame) == 1 and __REPR__ in self._edge_frame:
                return self._edge_frame[__REPR__][eid]
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
        if isinstance(eid, str) and eid == ALL:
            if len(self._edge_frame) == 1 and __REPR__ in self._edge_frame:
                return self._edge_frame[__REPR__]
            else:
                return dict(self._edge_frame)
        else:
            eid = utils.convert_to_id_tensor(eid, self.context)
            if len(self._edge_frame) == 1 and __REPR__ in self._edge_frame:
                return self._edge_frame[__REPR__][eid]
            else:
                return self._edge_frame.select_rows(eid)

    def set_device(self, ctx):
        """Set device context for this graph.

        Parameters
        ----------
        ctx : dgl.context.Context
          The device context.
        """
        self._context = ctx

    @property
    def context(self):
        """Get the device context of this graph."""
        return self._context

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

    def readout(self,
                readout_func,
                nodes=ALL,
                edges=ALL):
        """Trigger the readout function on the specified nodes/edges.

        Parameters
        ----------
        readout_func : callable
          Readout function.
        nodes : str, node, container or tensor
          The nodes to get reprs from.
        edges : str, pair of nodes, pair of containers or pair of tensors
          The edges to get reprs from.
        """
        nodes = self._nodes_or_all(nodes)
        edges = self._edges_or_all(edges)
        nstates = [self.nodes[n] for n in nodes]
        estates = [self.edges[e] for e in edges]
        return readout_func(nstates, estates)

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
            u = utils.convert_to_id_tensor(u)
            v = utils.convert_to_id_tensor(v)
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
            u = utils.convert_to_id_tensor(u)
            v = utils.convert_to_id_tensor(v)
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
        reordered_v, all_reduced_msgs = self._batch_reduce(v, reduce_func)
        if all_reduced_msgs is None:
            # no message; only do recv.
            if is_all(v):
                self.set_n_repr(f_update(self.get_n_repr(), None))
            else:
                self.set_n_repr(f_update(self.get_n_repr(v), None), v)
        else:
            # Read the node states in the degree-bucketing order.
            reordered_ns = self.get_n_repr(reordered_v)
            new_ns = f_update(reordered_ns, all_reduced_msgs)
            if is_all(v):
                # First do reorder and then replace the whole column.
                _, indices = F.sort(reordered_v)
                # TODO(minjie): manually convert ids to context.
                indices = F.to_context(indices, self.context)
                if isinstance(new_ns, dict):
                    for key, val in new_ns.items():
                        self._node_frame[key] = F.gather_row(val, indices)
                else:
                    self._node_frame[__REPR__] = F.gather_row(new_ns, indices)
            else:
                # Use setter to do reorder.
                self.set_n_repr(new_ns, reordered_v)

    def _batch_reduce(self, v, reduce_func):
        if is_all(v) and len(self._msg_frame) == 0:
            # no message has been sent
            return None, None

        if is_all(v):
            v = list(range(self.number_of_nodes()))
        # sanity checks
        v = utils.convert_to_id_tensor(v)
        f_reduce = _get_reduce_func(reduce_func)
        # degree bucketing
        degrees, v_buckets = scheduler.degree_bucketing(self.msg_graph, v)
        reduced_msgs = []
        for deg, v_bkt in zip(degrees, v_buckets):
            if deg == 0:
                continue
            bkt_len = len(v_bkt)
            uu, vv = self.msg_graph.in_edges(v_bkt)
            in_msg_ids = self.msg_graph.get_edge_id(uu, vv)
            # TODO(minjie): manually convert ids to context.
            in_msg_ids = F.to_context(in_msg_ids, self.context)
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
            dst_reprs = self.get_n_repr(v_bkt)
            reduced_msgs.append(f_reduce(dst_reprs, reshaped_in_msgs))

        if len(reduced_msgs) == 0:
            # no message has been sent to the specified node
            return None, None

        # TODO: clear partial messages
        self.clear_messages()

        # Read the node states in the degree-bucketing order.
        reordered_v = F.pack(v_buckets)
        # Pack all reduced msgs together
        if isinstance(reduced_msgs[0], dict):
            all_reduced_msgs = {key : F.pack(val) for key, val in reduced_msgs.items()}
        else:
            all_reduced_msgs = F.pack(reduced_msgs)

        return reordered_v, all_reduced_msgs

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
            u = utils.convert_to_id_tensor(u)
            v = utils.convert_to_id_tensor(v)
            # TODO(minjie): broadcasting is optional for many-one input.
            u, v = utils.edge_broadcasting(u, v)
            # relabel destination nodes.
            new2old, old2new = utils.build_relabel_map(v)
            # TODO(minjie): should not directly use []
            new_v = old2new[v]
            # create adj mat
            idx = F.pack([F.unsqueeze(new_v, 0), F.unsqueeze(u, 0)])
            dat = F.ones((len(u),))
            n = self.number_of_nodes()
            m = len(new2old)
            adjmat = F.sparse_tensor(idx, dat, [m, n])
            adjmat = F.to_context(adjmat, self.context)
            # TODO(minjie): use lazy dict for reduced_msgs
            reduced_msgs = {}
            for key in self._node_frame.schemes:
                col = self._node_frame[key]
                reduced_msgs[key] = F.spmm(adjmat, col)
            if len(reduced_msgs) == 1 and __REPR__ in reduced_msgs:
                reduced_msgs = reduced_msgs[__REPR__]
            node_repr = self.get_n_repr(new2old)
            new_node_repr = update_func(node_repr, reduced_msgs)
            self.set_n_repr(new_node_repr, new2old)
        else:
            u = utils.convert_to_id_tensor(u, self.context)
            v = utils.convert_to_id_tensor(v, self.context)
            self._batch_sendto(u, v, message_func)
            unique_v = F.unique(v)
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
            uu, vv = self.cached_graph.in_edges(v)
            self.update_by_edge(uu, vv, message_func,
                    reduce_func, update_func, batchable)
        else:
            for vv in utils.node_iter(v):
                assert vv in self.nodes
                uu = list(self.pred[vv])
                self.sendto(uu, vv, message_func, batchable)
                self.recv(vv, reduce_func, update_func, batchable)

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
            uu, vv = self.cached_graph.out_edges(u)
            self.update_by_edge(uu, vv, message_func,
                    reduce_func, update_func, batchable)
        else:
            for uu in utils.node_iter(u):
                assert uu in self.nodes
                for v in self.succ[uu]:
                    self.update_by_edge(uu, v,
                            message_func, reduce_func, update_func, batchable)

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
                adjmat = self.cached_graph.adjmat(self.context)
                reduced_msgs = {}
                for key in self._node_frame.schemes:
                    col = self._node_frame[key]
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

    def _nodes_or_all(self, nodes):
        return self.nodes() if nodes == ALL else nodes

    def _edges_or_all(self, edges):
        return self.edges() if edges == ALL else edges

    def _add_node_callback(self, node):
        self._cached_graph = None

    def _add_edge_callback(self, u, v):
        # In networkx 2.1, two adjlists are maintained. One for succ, one for pred.
        # We only record once for the succ addition.
        if self._edge_cb_state:
            #print('New edge:', u, v)
            self._edge_list.append((u, v))
        self._edge_cb_state = not self._edge_cb_state
        self._cached_graph = None

    @property
    def edge_list(self):
        """Return edges in the addition order."""
        return self._edge_list


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
