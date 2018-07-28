"""Base graph class specialized for neural networks on graphs.
"""
from __future__ import absolute_import

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
        super(DGLGraph, self).__init__(graph_data, **attr)
        self._cached_graph = None
        self._node_frame = Frame()
        self._edge_frame = Frame()
        self._msg_frame = Frame()
        self._msg_graph = None
        self._message_func = None
        self._reduce_func = None
        self._update_func = None
        self._edge_func = None

    def init_reprs(self, h_init=None):
        # TODO(gaiyu): multiple nodes
        print("[DEPRECATED]: please directly set node attrs "
              "(e.g. g.nodes[node]['x'] = val).")
        for n in self.nodes:
            self.set_repr(n, h_init)

    def set_n_repr(self, u, h_u):
        assert u in self.nodes
        kwarg = {__REPR__: h_u}
        self.add_node(u, **kwarg)

    def get_n_repr(self, u):
        assert u in self.nodes
        return self.nodes[u][__REPR__]

    def set_e_repr(self, u, v, h_uv):
        assert (u, v) in self.edges
        self.edges[u, v][__REPR__] = h_uv

    def get_e_repr(self, u, v):
        assert (u, v) in self.edges
        return self.edges[u, v][__REPR__]

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
        # TODO(minjie): tensorize following loop.
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
            self._batched_sendto(u, v, message_func)
        else:
            self._non_batched_sendto(u, v, message_func)

    def _non_batched_sendto(self, u, v, message_func):
        f_msg = _get_message_func(message_func)
        for uu, vv in utils.edge_iter(u, v):
            ret = f_msg(self._get_repr(self.nodes[uu]),
                        self._get_repr(self.edges[uu, vv]))
            self.edges[uu, vv][__MSG__] = ret

    def _batched_sendto(self, u, v, message_func):
        # TODO: __REPR__
        u = utils.convert_to_id_tensor(u)
        v = utils.convert_to_id_tensor(v)
        edge_id = self.cached_graph.get_edge_id(u, v)
        self.msg_graph.add_edges(u, v)
        if len(u) != len(v) and len(u) == 1:
            u = F.broadcast_to(u, v)
        src_reprs = self._node_frame.select_rows(u)
        edge_reprs = self._edge_frame.select_rows(edge_id)
        msgs = message_func(src_reprs, edge_reprs)
        self._msg_frame.append(msgs)

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
            self._batched_update_edge(u, v, edge_func)
        else:
            self._non_batched_update_edge(u, v, edge_func)

    def _non_batched_update_edge(self, u, v, edge_func):
        for uu, vv in utils.edge_iter(u, v):
            ret = edge_func(self._get_repr(self.nodes[uu]),
                            self._get_repr(self.nodes[vv]),
                            self._get_repr(self.edges[uu, vv]))
            self._set_repr(self.edges[uu, vv], ret)

    def _batched_update_edge(self, u, v, edge_func):
        # TODO: __REPR__
        u = utils.convert_to_id_tensor(u)
        v = utils.convert_to_id_tensor(v)
        edge_id = self.cached_graph.get_edge_id(u, v)
        if len(u) != len(v) and len(u) == 1:
            u = F.broadcast_to(u, v)
        elif len(u) != len(v) and len(v) == 1:
            v = F.broadcast_to(v, u)
        src_reprs = self._node_frame.select_rows(u)
        dst_reprs = self._node_frame.select_rows(v)
        edge_reprs = self._edge_frame.select_rows(edge_id)
        new_edge_reprs = edge_func(src_reprs, dst_reprs, edge_reprs)
        self._edge_frame.update_rows(edge_id, new_edge_reprs)

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
            self._batched_recv(u, reduce_func, update_func)
        else:
            self._non_batched_recv(u, reduce_func, update_func)

    def _non_batched_recv(self, u, reduce_func, update_func):
        u_is_container = isinstance(u, list)
        u_is_tensor = isinstance(u, Tensor)
        f_reduce = _get_reduce_func(reduce_func)
        f_update = update_func
        # TODO(minjie): tensorize the loop.
        for i, uu in enumerate(utils.node_iter(u)):
            # TODO(minjie): tensorize the message batching
            # reduce phase
            msgs_batch = [self.edges[vv, uu].pop(__MSG__)
                          for vv in self.pred[uu] if __MSG__ in self.edges[vv, uu]]
            if len(msgs_batch) == 0:
                msgs_reduced = None
            elif len(msgs_batch) == 1:
                msgs_reduced = msgs_batch[0]
            else:
                msgs_reduced = f_reduce(self._get_repr(self.nodes[uu]), msgs_batch)
            # update phase
            ret = f_update(self._get_repr(self.nodes[uu]), msgs_reduced)
            self._set_repr(self.nodes[uu], ret)

    def _batched_recv(self, v, reduce_func, update_func):
        # sanity checks
        v = utils.convert_to_id_tensor(v)
        f_reduce = _get_reduce_func(reduce_func)
        f_update = update_func
        # TODO: __REPR__
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
            def _reshape_fn(key):
                msg = in_msgs[key]
                msg_shape = F.shape(msg)
                new_shape = (bkt_len, deg) + msg_shape[1:]
                return F.reshape(msg, new_shape)
            reshaped_in_msgs = utils.LazyDict(_reshape_fn, self._msg_frame.schemes)
            dst_reprs = self._node_frame.select_rows(v_bkt)
            reduced_msgs.append(f_reduce(dst_reprs, reshaped_in_msgs))

        # TODO: clear partial messages
        self.clear_messages()

        # Read the node states in the degree-bucketing order.
        reordered_v = F.pack(v_buckets)
        reordered_ns = self._node_frame.select_rows(reordered_v)
        # TODO(minjie): handle rich-typed reduced msgs
        all_reduced_msgs = F.pack(reduced_msgs)
        new_ns = f_update(reordered_ns, all_reduced_msgs)
        self._node_frame.update_rows(reordered_v, new_ns)

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
            self._batched_update_by_edge(
                    u, v, message_func, reduce_func, update_func)
        else:
            self._non_batched_update_by_edge(
                    u, v, message_func, reduce_func, update_func)

    def _non_batched_update_by_edge(
            self,
            u, v,
            message_func=None,
            reduce_func=None,
            update_func=None):
        self._non_batched_sendto(u, v, message_func)
        dst = set()
        for uu, vv in utils.edge_iter(u, v):
            dst.add(vv)
        self._non_batched_recv(list(dst), reduce_func, update_func)

    def _batched_update_by_edge(
            self,
            u, v,
            message_func=None,
            reduce_func=None,
            update_func=None):
        if message_func == 'from_src' and reduce_func == 'sum':
            # Specialized to generic-SPMV
            raise NotImplementedError('SPVM specialization')
        else:
            self._batched_sendto(u, v, message_func)
            unique_v = F.unique(v)
            self._batched_recv(unique_v, reduce_func, update_func)

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
            self._batched_update_by_edge(u, v,
                    message_func, reduce_func, update_func)
        else:
            u = [uu for uu, _ in self.edges]
            v = [vv for _, vv in self.edges]
            self._non_batched_sendto(u, v, message_func)
            self._non_batched_recv(list(self.nodes()), reduce_func, update_func)

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

    @staticmethod
    def _get_repr(attr_dict):
        if len(attr_dict) == 1 and __REPR__ in attr_dict:
            return attr_dict[__REPR__]
        else:
            return attr_dict

    @staticmethod
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
