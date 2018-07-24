"""Base graph class specialized for neural networks on graphs.
"""

from collections import defaultdict
from functools import reduce
import networkx as nx
from networkx.classes.digraph import DiGraph
import numpy as np

import dgl.backend as F
from dgl.backend import Tensor
import dgl.state as state
import dgl.utils as utils

__MSG__ = "__msg__"
__REPR__ = "__repr__"
__MFUNC__ = "__mfunc__"
__EFUNC__ = "__efunc__"
__UFUNC__ = "__ufunc__"
__RFUNC__ = "__rfunc__"
__READOUT__ = "__readout__"

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
    node_dict_factory = state.NodeDict
    adjlist_outer_dict_factory = state.AdjOuterDict
    adjlist_inner_dict_factory = state.AdjInnerDict
    edge_attr_dict_factory = state.EdgeAttrDict

    def __init__(self, graph_data=None, **attr):
        super(DGLGraph, self).__init__(graph_data, **attr)
        self._glb_func = {}

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
                              edges='all',
                              batchable=False):
        """Register computation on edges.

        The message function should be compatible with following signature:

        (node_reprs, node_reprs, edge_reprs) -> msg

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
        def _msg_edge_func(u, v, uv):
            ret = message_func(u, v, uv) # TODO(gaiyu): u and uv only
            if isinstance(ret, dict):
                assert all(isinstance(key, str) for key in ret) # TODO(gaiyu): remove
                return {__MSG__ + key : value for key, value in ret}
            else:
                return {__MSG__ : ret}
        self._internal_register_edge(__MFUNC__, _msg_edge_func, edges, batchable)

    def register_edge_func(self,
                           edge_func,
                           edges='all',
                           batchable=False):
        """Register computation on edges.

        The edge function should be compatible with following signature:

        (node_reprs, node_reprs, edge_reprs) -> edge_reprs

        It computes the new edge representations (the same concept as messages)
        using the representations of the source node, target node and the edge
        itself. All node_reprs and edge_reprs are dictionaries. If `update_func`
        returns a dict, edge attribute dict(s) will be updated with it. Otherwise, 
        the returned object will become new default edge representation(s).

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
        self._internal_register_edge(__EFUNC__, edge_func, edges, batchable)

    def register_reduce_func(self,
                             reduce_func,
                             nodes='all',
                             batchable=False):
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
        self._internal_register_node(__RFUNC__, reduce_func, nodes, batchable)

    def register_update_func(self,
                             update_func,
                             nodes='all',
                             batchable=False):
        """Register computation on nodes.

        The update function should be compatible with following signature:

        (node_reprs, reduced_edge_repr) -> node_reprs

        It computes the new node representations using the representations
        of the in-coming edges (the same concept as messages) and the node
        itself. All node_reprs and edge_reprs are dictionaries. If `update_func` 
        returns a dict, node attribute dict(s) will be updated with it. Otherwise, 
        the returned object will become new default node representation(s).

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
        name : str
          The name of the function.

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
        self._internal_register_node(__UFUNC__, update_func, nodes, batchable)

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
        self._glb_func[__READOUT__] = readout_func

    def readout(self,
                nodes='all',
                edges='all',
                **kwargs):
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
        assert __READOUT__ in self._glb_func, \
            "Readout function has not been registered."
        # TODO(minjie): tensorize following loop.
        nstates = [self.nodes[n] for n in nodes]
        estates = [self.edges[e] for e in edges]
        return self._glb_func[__READOUT__](nstates, estates, **kwargs)

    def sendto(self, u, v):
        """Trigger the message function on edge u->v

        Parameters
        ----------
        u : node, container or tensor
          The source node(s).
        v : node, container or tensor
          The destination node(s).
        """
        # Cannot store messages separately (graph mutation)
        self._internal_trigger_edges(u, v, __MFUNC__)

    def update_edge(self, u, v):
        """Update representation on edge u->v

        Parameters
        ----------
        u : node, container or tensor
          The source node(s).
        v : node, container or tensor
          The destination node(s).
        """
        self._internal_trigger_edges(u, v, __EFUNC__)

    def recv(self, v, reduce_func, update_func, batchable=False):
        # TODO(gaiyu): __REPR__
        if batchable:
            if isinstance(v, list):
                assert utils.homogeneous(v, int)
                v = F.tensor(v)
            elif isinstance(v, Tensor):
                assert F.isinteger(u) and len(F.shape(u)) == 1
            elif isinstance(v, int):
                v = F.tensor([v])
            else:
                raise KeyError()

            # TODO(gaiyu): purge message

            keys = [x for x in self._edge_frame.column_names() if x.startswith(__MSG__)]
            select_columns = self._edge_frame.select_columns(keys)
            uv_frame = select_columns.filter_by('dst', v)
            assert uv_frame.dropna().num_rows() == uv_frame.num_rows()
            groupby = uv_frame.groupby('dst', {'deg' : aggregate.COUNT()})
            unique = groupby['deg'].unique().to_numpy()
            uv_frame = uv_frame.join('dst', groupby)

            dst_list = []
            r_list = []
            for x in unique:
                frame = uv_frame[uv_frame['deg'] == x]
                src, dst = frame['src'], frame['dst']

                v_dict = self.nodes[dst]

                uv_dict = self.edges[src, dst]
                shape = [int(frame.num_rows() / x), x, -1]
                uv_dict = {k[:len(__MSG__)] : F.reshape(_.pop(k), shape) for k in keys}

                v_dict = reduce_func(v_dict, uv_dict)
                assert all(isinstance(x, Tensor) and \
                           F.shape(x)[0] == n for x in r.values())

                dst_list.append(dst)
                r_list.append(r_dict)

            def valid(x_list):
                key_set = set(x_list[0])
                return all(set(x) == key_set for x in x_list) and \
                    all(F.packable([x[key] for x in x_list]) for key in key_set)

            assert valid(w_list)
            assert valid(r_list)

            wx = {key : F.pack([wx[key] for wx in w_list]) for key in w_list[0]}
            rx = {key : F.pack([rx[key] for rx in r_list]) for key in r_list[0]}
            ux = update_func(wx, rx)

            assert all(isinstance(x, Tensor) and \
                       F.shape(x)[0] == F.shape(v)[0] for x in ux.values())

            for key, value in u.items():
                self.nodes[v][key] = value

            for key in keys:
                if not self._edge_frame[key].shape:
                    del self._edge_frame[key]
        else:
            pass

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
        dst = set()
        for uu, vv in utils.edge_iter(u, v):
            dst.add(vv)
        self.recv(list(dst))

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
            self.recv(uu)

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
        self.recv(list(self.nodes()))

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

    def _internal_register_node(self, name, func, nodes, batchable):
        # TODO(minjie): handle batchable
        # TODO(minjie): group nodes based on their registered func
        if nodes == 'all':
            self._glb_func[name] = func
        else:
            for n in nodes:
                self.nodes[n][name] = func

    def _internal_register_edge(self, name, func, edges, batchable):
        # TODO(minjie): handle batchable
        # TODO(minjie): group edges based on their registered func
        if edges == 'all':
            self._glb_func[name] = func
        else:
            for e in edges:
                self.edges[e][name] = func

    @staticmethod
    def _get_repr(attr_dict):
        return attr_dict[__REPR__] if __REPR__ in attr_dict else attr_dict

    @staticmethod
    def _set_repr(self, attr_dict, attr):
        if isinstance(attr, dict):
            attr_dict.update(attr)
        else:
            attr_dict[__REPR__] = attr

    def _internal_trigger_edges(self, u, v, func):
        uu = self._get_repr(self.nodes[u])
        vv = self._get_repr(self.nodes[v])
        uv = self._get_repr(self.edges[u, v])

        uu_shape = F.shape(uu)
        vv_shape = F.shape(vv)
        if uu_shape != vv_shape:
            assert uu_shape[0] == 1 or vv_shape[0] == 1 # TODO(gaiyu): remove
            if uu_shape[0] < vv_shape[0]:
                uu = F.concatenate([uu] * vv_shape[0])
            else:
                vv = F.concatenate([vv] * uu_shape[0])

        self._set_repr(self.edges[u][v], func(uu, vv, uv))
