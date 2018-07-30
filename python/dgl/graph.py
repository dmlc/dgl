"""Base graph class specialized for neural networks on graphs.
"""
from __future__ import absolute_import

from collections import MutableMapping
import networkx as nx
from networkx.classes.digraph import DiGraph

import dgl.backend as F
from dgl.backend import Tensor
import dgl.builtin as builtin
#import dgl.state as state
from dgl.frame import Frame
from dgl.cached_graph import CachedGraph, create_cached_graph
import dgl.scheduler as scheduler
import dgl.utils as utils

__MSG__ = "__msg__"
__REPR__ = "__repr__"
ALL = "__all__"

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
    #node_dict_factory = state.NodeDict
    #adjlist_outer_dict_factory = state.AdjOuterDict
    #adjlist_inner_dict_factory = state.AdjInnerDict
    #edge_attr_dict_factory = state.EdgeAttrDict

    def __init__(self, graph_data=None, **attr):
        # call base class init
        super(DGLGraph, self).__init__(graph_data, **attr)
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
        hu : any
          Node representation.
        u : node, container or tensor
          The node(s).
        """
        # sanity check
        if isinstance(u, str) and u == ALL:
            num_nodes = self.number_of_nodes()
        else:
            u = utils.convert_to_id_tensor(u)
            num_nodes = len(u)
        if isinstance(hu, dict):
            for key, val in hu.items():
                assert F.shape(val)[0] == num_nodes
        else:
            F.shape(hu)[0] == num_nodes
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
                    self._node_frame[key][u] = val
            else:
                self._node_frame[__REPR__][u] = hu

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
            u = utils.convert_to_id_tensor(u)
            if len(self._node_frame) == 1 and __REPR__ in self._node_frame:
                return self._node_frame[__REPR__][u]
            else:
                return self._node_frame.select_rows(u)

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
        h_uv : any
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
            u = utils.convert_to_id_tensor(u)
            v = utils.convert_to_id_tensor(v)
            num_edges = max(len(u), len(v))
        if isinstance(h_uv, dict):
            for key, val in h_uv.items():
                assert F.shape(val)[0] == num_edges
        else:
            F.shape(h_uv)[0] == num_edges
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
                    self._edge_frame[key][eid] = val
            else:
                self._edge_frame[__REPR__][eid] = h_uv

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
            u = utils.convert_to_id_tensor(u)
            v = utils.convert_to_id_tensor(v)
            eid = self.cached_graph.get_edge_id(u, v)
            if len(self._edge_frame) == 1 and __REPR__ in self._edge_frame:
                return self._edge_frame[__REPR__][eid]
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
        for uu, vv in utils.edge_iter(u, v):
            ret = f_msg(_get_repr(self.nodes[uu]),
                        _get_repr(self.edges[uu, vv]))
            self.edges[uu, vv][__MSG__] = ret

    def _batch_sendto(self, u, v, message_func):
        u = utils.convert_to_id_tensor(u)
        v = utils.convert_to_id_tensor(v)
        edge_id = self.cached_graph.get_edge_id(u, v)
        self.msg_graph.add_edges(u, v)
        if len(u) != len(v) and len(u) == 1:
            u = F.broadcast_to(u, v)
        src_reprs = _get_repr(self._node_frame.select_rows(u))
        edge_reprs = _get_repr(self._edge_frame.select_rows(edge_id))
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
        for uu, vv in utils.edge_iter(u, v):
            ret = edge_func(_get_repr(self.nodes[uu]),
                            _get_repr(self.nodes[vv]),
                            _get_repr(self.edges[uu, vv]))
            _set_repr(self.edges[uu, vv], ret)

    def _batch_update_edge(self, u, v, edge_func):
        u = utils.convert_to_id_tensor(u)
        v = utils.convert_to_id_tensor(v)
        edge_id = self.cached_graph.get_edge_id(u, v)
        if len(u) != len(v) and len(u) == 1:
            u = F.broadcast_to(u, v)
        elif len(u) != len(v) and len(v) == 1:
            v = F.broadcast_to(v, u)
        src_reprs = _get_repr(self._node_frame.select_rows(u))
        dst_reprs = _get_repr(self._node_frame.select_rows(v))
        edge_reprs = _get_repr(self._edge_frame.select_rows(edge_id))
        new_edge_reprs = edge_func(src_reprs, dst_reprs, edge_reprs)
        _batch_set_repr(self._edge_frame, edge_id, new_edge_reprs)

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
        for i, uu in enumerate(utils.node_iter(u)):
            # reduce phase
            msgs_batch = [self.edges[vv, uu].pop(__MSG__)
                          for vv in self.pred[uu] if __MSG__ in self.edges[vv, uu]]
            if len(msgs_batch) == 0:
                msgs_reduced = None
            elif len(msgs_batch) == 1:
                msgs_reduced = msgs_batch[0]
            else:
                msgs_reduced = f_reduce(_get_repr(self.nodes[uu]), msgs_batch)
            # update phase
            ret = f_update(_get_repr(self.nodes[uu]), msgs_reduced)
            _set_repr(self.nodes[uu], ret)

    def _batch_recv(self, v, reduce_func, update_func):
        # sanity checks
        v = utils.convert_to_id_tensor(v)
        f_reduce = _get_reduce_func(reduce_func)
        f_update = update_func
        # degree bucketing
        degrees, v_buckets = scheduler.degree_bucketing(self.msg_graph, v)
        reduced_msgs = []
        for deg, v_bkt in zip(degrees, v_buckets):
            bkt_len = len(v_bkt)
            uu, vv = self.msg_graph.in_edges(v_bkt)
            in_msg_ids = self.msg_graph.get_edge_id(uu, vv)
            # The in_msgs represents the rows selected. Since our storage
            # is column-based, it will only be materialized when user
            # tries to get the column (e.g. when user called `msgs['h']`)
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
            dst_reprs = _get_repr(self._node_frame.select_rows(v_bkt))
            reduced_msgs.append(f_reduce(dst_reprs, reshaped_in_msgs))

        # TODO: clear partial messages
        self.clear_messages()

        # Read the node states in the degree-bucketing order.
        reordered_v = F.pack(v_buckets)
        reordered_ns = _get_repr(self._node_frame.select_rows(reordered_v))
        # Pack all reduced msgs together
        if isinstance(reduced_msgs, dict):
            all_reduced_msgs = {key : F.pack(val) for key, val in reduced_msgs.items()}
        else:
            all_reduced_msgs = F.pack(reduced_msgs)
        new_ns = f_update(reordered_ns, all_reduced_msgs)
        _batch_set_repr(self._node_frame, reordered_v, new_ns)

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
            message_func=None,
            reduce_func=None,
            update_func=None):
        self._nonbatch_sendto(u, v, message_func)
        dst = set()
        for uu, vv in utils.edge_iter(u, v):
            dst.add(vv)
        self._nonbatch_recv(list(dst), reduce_func, update_func)

    def _batch_update_by_edge(
            self,
            u, v,
            message_func=None,
            reduce_func=None,
            update_func=None):
        if message_func == 'from_src' and reduce_func == 'sum':
            # Specialized to generic-SPMV
            raise NotImplementedError('SPVM specialization')
        else:
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
            u, v = self.cached_graph.edges()
            self._batch_update_by_edge(u, v,
                    message_func, reduce_func, update_func)
        else:
            u = [uu for uu, _ in self.edges]
            v = [vv for _, vv in self.edges]
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

def _batch_set_repr(frame, rows, attr):
    if isinstance(attr, dict):
        frame.update_rows(rows, attr)
    else:
        frame.update_rows(rows, {__REPR__ : attr})

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
