"""Base graph class specialized for neural networks on graphs.
"""

from collections import defaultdict
import networkx as nx
from networkx.classes.digraph import DiGraph

import dgl.backend as F
from dgl.backend import Tensor
import dgl.utils as utils

__MSG__ = "__msg__"
__E_REPR__ = "__e_repr__"
__N_REPR__ = "__n_repr__"
__MFUNC__ = "__mfunc__"
__EFUNC__ = "__efunc__"
__UFUNC__ = "__ufunc__"
__RFUNC__ = "__rfunc__"

class DGLGraph(DiGraph):
    """Base graph class specialized for neural networks on graphs.

    TODO(minjie): document of multi-node and multi-edge syntax.

    Parameters
    ----------
    data : graph data
        Data to initialize graph. Same as networkx's semantics.
    attr : keyword arguments, optional
        Attributes to add to graph as key=value pairs.
    """
    def __init__(self, graph_data=None, **attr):
        super(DGLGraph, self).__init__(graph_data, **attr)
        self.m_func = None
        self.u_func = None
        self.e_func = None
        self.readout_func = None

    def init_reprs(self, h_init=None):
        # TODO(gaiyu): multiple nodes
        print("[DEPRECATED]: please directly set node attrs "
              "(e.g. g.nodes[node]['x'] = val).")
        for n in self.nodes:
            self.set_repr(n, h_init)

    def set_repr(self, u, h_u, name=__N_REPR__):
        # TODO(gaiyu): multiple nodes
        print("[DEPRECATED]: please directly set node attrs "
              "(e.g. g.nodes[node]['x'] = val).")
        assert u in self.nodes
        kwarg = {name: h_u}
        self.add_node(u, **kwarg)

    def get_repr(self, u, name=__N_REPR__):
        # TODO(gaiyu): multiple nodes
        print("[DEPRECATED]: please directly get node attrs "
              "(e.g. g.nodes[node]['x']).")
        assert u in self.nodes
        return self.nodes[u][name]

    def register_message_func(self, message_func, edges='all', batchable=False):
        """Register computation on edges.

        The message function should be compatible with following signature:

        (node_reprs, node_reprs, edge_reprs) -> edge_reprs

        It computes the representation of a message
        using the representations of the source node, target node and the edge
        itself. All node_reprs and edge_reprs are dictionaries.

        Parameters
        ----------
        message_func : callable
          Message function on the edge.
        edges : str, pair of nodes, pair of containers, pair of tensors
          The edges for which the message function is registered. Default is
          registering for all the edges. Registering for multiple edges is
          supported.
        batchable : bool
          Whether the provided message function allows batch computing.

        Examples
        --------

        Register for all edges.
        >>> g.register_message_func(mfunc)

        Register for a specific edge.
        >>> g.register_message_func(mfunc, (u, v))

        Register for multiple edges.
        >>> u = [u1, u2, u3, ...]
        >>> v = [v1, v2, v3, ...]
        >>> g.register_message_func(mfunc, (u, v))
        """
        if edges == 'all':
            self.m_func = message_func
        else:
            for e in edges:
                self.edges[e][__MFUNC__] = message_func

    def register_edge_func(self, edge_func, edges='all', batchable=False):
        """Register computation on edges.

        The edge function should be compatible with following signature:

        (node_reprs, node_reprs, edge_reprs) -> edge_reprs

        It computes the new edge representations (the same concept as messages)
        using the representations of the source node, target node and the edge
        itself. All node_reprs and edge_reprs are dictionaries.

        Parameters
        ----------
        edge_func : callable
          Message function on the edge.
        edges : str, pair of nodes, pair of containers, pair of tensors
          The edges for which the message function is registered. Default is
          registering for all the edges. Registering for multiple edges is
          supported.
        batchable : bool
          Whether the provided message function allows batch computing.

        Examples
        --------

        Register for all edges.
        >>> g.register_edge_func(efunc)

        Register for a specific edge.
        >>> g.register_edge_func(efunc, (u, v))

        Register for multiple edges.
        >>> u = [u1, u2, u3, ...]
        >>> v = [v1, v2, v3, ...]
        >>> g.register_edge_func(mfunc, (u, v))
        """
        if edges == 'all':
            self.e_func = edge_func
        else:
            for e in edges:
                self.edges[e][__EFUNC__] = edge_func

    def register_reduce_func(self, reduce_func, nodes='all', batchable=False):
        """Register message reduce function on incoming edges.

        The reduce function should be compatible with following signature:

        edge_reprs -> reduced_edge_repr

        It computes the reduced edge representations using the representations
        of the in-coming edges (the same concept as messages).

        The reduce function can be any of the pre-defined functions ('sum',
        'max'). If built-in function is used, computation will be performed
        efficiently (using generic-SPMV kernels).

        Parameters
        ----------
        reduce_func : str or callable
          Reduce function on incoming edges.
        nodes : str, node, container or tensor
          The nodes for which the reduce function is registered. Default is
          registering for all the nodes. Registering for multiple nodes is
          supported.
        batchable : bool
          Whether the provided reduce function allows batch computing.

        Examples
        --------

        Register for all nodes.
        >>> g.register_reduce_func(rfunc)

        Register for a specific node.
        >>> g.register_reduce_func(rfunc, u) # TODO Not implemented

        Register for multiple nodes.
        >>> u = [u1, u2, u3, ...]
        >>> g.register_reduce_func(rfunc, u)
        """
        if isinstance(reduce_func, str):
            # built-in reduce func
            if reduce_func == 'sum':
                reduce_func = F.reduce_sum
            elif reduce_func == 'max':
                reduce_func = F.reduce_max
            else:
                raise NotImplementedError(
                        "Built-in function %s not implemented" % reduce_func)
        if nodes == 'all':
            self.r_func = reduce_func
        else:
            for n in nodes:
                self.nodes[n][__RFUNC__] = reduce_func

    def register_update_func(self, update_func, nodes='all', batchable=False):
        """Register computation on nodes.

        The update function should be compatible with following signature:

        (node_reprs, reduced_edge_repr) -> node_reprs

        It computes the new node representations using the representations
        of the in-coming edges (the same concept as messages) and the node
        itself. All node_reprs and edge_reprs are dictionaries.

        Parameters
        ----------
        update_func : callable
          Update function on the node.
        nodes : str, node, container or tensor
          The nodes for which the update function is registered. Default is
          registering for all the nodes. Registering for multiple nodes is
          supported.
        batchable : bool
          Whether the provided update function allows batch computing.

        Examples
        --------

        Register for all nodes.
        >>> g.register_update_func(ufunc)

        Register for a specific node.
        >>> g.register_update_func(ufunc, u) # TODO Not implemented

        Register for multiple nodes.
        >>> u = [u1, u2, u3, ...]
        >>> g.register_update_func(ufunc, u)
        """
        if nodes == 'all':
            self.u_func = update_func
        else:
            for n in nodes:
                self.nodes[n][__UFUNC__] = update_func

    def register_readout_func(self, readout_func):
        """Register computation on the whole graph.

        The readout_func should be compatible with following signature:

        (node_reprs, edge_reprs) -> any

        It takes the representations of selected nodes and edges and
        returns readout values.

        NOTE: readout function can be implemented outside of DGLGraph.
        One can simple get the node/edge reprs of the graph and perform
        arbitrary computation.

        Parameters
        ----------
        readout_func : callable
          The readout function.

        See Also
        --------
        readout
        """
        self.readout_func = readout_func

    def readout(self, nodes='all', edges='all', **kwargs):
        """Trigger the readout function on the specified nodes/edges.

        Parameters
        ----------
        nodes : str, node, container or tensor
          The nodes to get reprs from.
        edges : str, pair of nodes, pair of containers or pair of tensors
          The edges to get reprs from.
        kwargs : keyword arguments, optional
            Arguments for the readout function.
        """
        nodes = self._nodes_or_all(nodes)
        edges = self._edges_or_all(edges)
        assert self.readout_func is not None, \
            "Readout function is not registered."
        # TODO(minjie): tensorize following loop.
        nstates = [self.nodes[n] for n in nodes]
        estates = [self.edges[e] for e in edges]
        return self.readout_func(nstates, estates, **kwargs)

    def sendto(self, u, v):
        """Trigger the message function on edge u->v

        Parameters
        ----------
        u : node, container or tensor
          The source node(s).
        v : node, container or tensor
          The destination node(s).
        """
        # TODO(minjie): tensorize the loop.
        for uu, vv in utils.edge_iter(u, v):
            f_msg = self.edges[uu, vv].get(__MFUNC__, self.m_func)
            assert f_msg is not None, \
                "message function not registered for edge (%s->%s)" % (uu, vv)
            m = f_msg(self.nodes[uu], self.nodes[vv], self.edges[uu, vv])
            self.edges[uu, vv][__MSG__] = m

    def update_edge(self, u, v):
        """Update representation on edge u->v

        Parameters
        ----------
        u : node, container or tensor
          The source node(s).
        v : node, container or tensor
          The destination node(s).
        """
        # TODO(minjie): tensorize the loop.
        for uu, vv in utils.edge_iter(u, v):
            f_edge = self.edges[uu, vv].get(__EFUNC__, self.m_func)
            assert f_edge is not None, \
                "edge function not registered for edge (%s->%s)" % (uu, vv)
            m = f_edge(self.nodes[uu], self.nodes[vv], self.edges[uu, vv])
            self.edges[uu, vv][__E_REPR__] = m

    def recvfrom(self, u, preds=None):
        """Trigger the update function on node u.

        It computes the new node state using the messages and edge
        states from preds->u. If `u` is one node, `preds` is a list
        of predecessors. If `u` is a container or tensor of nodes,
        then `preds[i]` should be the predecessors of `u[i]`.

        Parameters
        ----------
        u : node, container or tensor
          The node to be updated.
        preds : container
          Nodes with pre-computed messages to u. Default is all
          the predecessors.
        """
        u_is_container = isinstance(u, list)
        u_is_tensor = isinstance(u, Tensor)
        # TODO(minjie): tensorize the loop.
        for i, uu in enumerate(utils.node_iter(u)):
            if preds is None:
                v = list(self.pred[uu])
            elif u_is_container or u_is_tensor:
                v = preds[i]
            else:
                v = preds
            # TODO(minjie): tensorize the message batching
            m = [self.edges[vv, uu][__MSG__] for vv in v]
            f_reduce = self.nodes[uu].get(__RFUNC__, self.r_func)
            assert f_reduce is not None, \
                "Reduce function not registered for node %s" % uu
            msgs_reduced_repr = f_reduce(m)
            f_update = self.nodes[uu].get(__UFUNC__, self.u_func)
            assert f_update is not None, \
                "Update function not registered for node %s" % uu
            self.node[uu].update(f_update(self.nodes[uu], msgs_reduced_repr))

    def update_by_edge(self, u, v):
        """Trigger the message function on u->v and update v.

        Parameters
        ----------
        u : node, container or tensor
          The source node(s).
        v : node, container or tensor
          The destination node(s).
        """
        self.sendto(u, v)
        # TODO(minjie): tensorize the following loops.
        preds = defaultdict(list)
        for uu, vv in utils.edge_iter(u, v):
            preds[vv].append(uu)
        if len(preds) == 1:
            dst = list(preds.keys())[0]
            src = preds[dst]
            self.recvfrom(dst, src)
        elif len(preds) > 1:
            dst = list(preds.keys())
            src = [preds[d] for d in dst]
            self.recvfrom(dst, src)

    def update_to(self, u):
        """Pull messages from the node's predecessors and then update it.

        Parameters
        ----------
        u : node, container or tensor
          The node to be updated.
        """
        # TODO(minjie): tensorize the following code.
        for uu in utils.node_iter(u):
            assert uu in self.nodes
            preds = list(self.pred[uu])
            self.sendto(preds, uu)
            self.recvfrom(uu, preds)

    def update_from(self, u):
        """Send message from the node to its successors and update them.

        Parameters
        ----------
        u : node, container or tensor
          The node that sends out messages.
        """
        # TODO(minjie): tensorize the following code.
        for uu in utils.node_iter(u):
            assert uu in self.nodes
            for v in self.succ[uu]:
                self.update_by_edge(uu, v)

    def update_all(self):
        """Send messages through all the edges and update all nodes.
        """
        # TODO(minjie): tensorize the following code.
        u = [uu for uu, _ in self.edges]
        v = [vv for _, vv in self.edges]
        self.sendto(u, v)
        self.recvfrom(list(self.nodes()))
        # TODO(zz): this is a hack
        if self.e_func:
            self.update_edge(u, v)

    def propagate(self, iterator='bfs', **kwargs):
        """Propagate messages and update nodes using iterator.

        A convenient function for passing messages and updating
        nodes according to the iterator. The iterator can be
        any of the pre-defined iterators ('bfs', 'dfs', 'pre-order',
        'mid-order', 'post-order'). The computation will be unrolled
        in the backend efficiently. User can also provide custom
        iterator that generates the edges and nodes.

        Parameters
        ----------
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
                self.update_by_edge(u, v)

    def draw(self):
        """Plot the graph using dot."""
        from networkx.drawing.nx_agraph import graphviz_layout

        pos = graphviz_layout(self, prog='dot')
        nx.draw(self, pos, with_labels=True)

    def _nodes_or_all(self, nodes='all'):
        return self.nodes() if nodes == 'all' else nodes

    def _edges_or_all(self, edges='all'):
        return self.edges() if edges == 'all' else edges
