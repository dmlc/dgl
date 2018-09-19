"""Base graph class specialized for neural networks on graphs.
"""
from __future__ import absolute_import

import networkx as nx

import dgl
from .base import ALL, is_all, __MSG__, __REPR__
from . import backend as F
from .backend import Tensor
from .frame import FrameRef, merge_frames
from .function.message import BundledMessageFunction
from .function.reducer import BundledReduceFunction
from .graph_index import GraphIndex, create_graph_index
from . import scheduler
from . import utils

__all__ = ['DLGraph']

class DGLGraph(object):
    """Base graph class specialized for neural networks on graphs.

    TODO(minjie): document of batching semantics
    TODO(minjie): document of __REPR__ semantics

    Parameters
    ----------
    graph_data : graph data
        Data to initialize graph. Same as networkx's semantics.
    node_frame : FrameRef
        Node feature storage.
    edge_frame : FrameRef
        Edge feature storage.
    attr : keyword arguments, optional
        Attributes to add to graph as key=value pairs.
    """
    def __init__(self,
                 graph_data=None,
                 node_frame=None,
                 edge_frame=None,
                 **attr):
        # TODO: keyword attr
        # graph
        self._graph = create_graph_index(graph_data)
        # frame
        self._node_frame = node_frame if node_frame is not None else FrameRef()
        self._edge_frame = edge_frame if edge_frame is not None else FrameRef()
        # msg graph & frame
        self._msg_graph = create_graph_index()
        self._msg_frame = FrameRef()
        self.reset_messages()
        # registered functions
        self._message_func = (None, None)
        self._reduce_func = (None, None)
        self._edge_func = (None, None)
        self._apply_node_func = (None, None)
        self._apply_edge_func = (None, None)

    def add_nodes(self, num, reprs=None):
        """Add nodes.
        
        Parameters
        ----------
        num : int
            Number of nodes to be added.
        reprs : dict
            Optional node representations.
        """
        self._graph.add_nodes(num)
        self._msg_graph.add_nodes(num)
        #TODO(minjie): change frames
        assert reprs is None

    def add_edge(self, u, v, reprs=None):
        """Add one edge.
        
        Parameters
        ----------
        u : int
            The src node.
        v : int
            The dst node.
        reprs : dict
            Optional edge representation.
        """
        self._graph.add_edge(u, v)
        #TODO(minjie): change frames
        assert reprs is None

    def add_edges(self, u, v, reprs=None):
        """Add many edges.
        
        Parameters
        ----------
        u : list, tensor
            The src nodes.
        v : list, tensor
            The dst nodes.
        reprs : dict
            Optional node representations.
        """
        u = utils.toindex(u)
        v = utils.toindex(v)
        self._graph.add_edges(u, v)
        #TODO(minjie): change frames
        assert reprs is None

    def clear(self):
        """Clear the graph and its storage."""
        self._graph.clear()
        self._node_frame.clear()
        self._edge_frame.clear()
        self._msg_graph.clear()
        self._msg_frame.clear()

    def reset_messages(self):
        """Clear all messages."""
        self._msg_graph.clear()
        self._msg_frame.clear()
        self._msg_graph.add_nodes(self.number_of_nodes())

    def number_of_nodes(self):
        """Return the number of nodes.

        Returns
        -------
        int
            The number of nodes
        """
        return self._graph.number_of_nodes()

    def __len__(self):
        """Return the number of nodes."""
        return self.number_of_nodes()

    def number_of_edges(self):
        """Return the number of edges.

        Returns
        -------
        int
            The number of edges
        """
        return self._graph.number_of_edges()

    def has_node(self, vid):
        """Return true if the node exists.

        Parameters
        ----------
        vid : int
            The nodes

        Returns
        -------
        bool
            True if the node exists
        """
        return self.has_node(vid)
    
    def __contains__(self, vid):
        """Same as has_node."""
        return self.has_node(vid)

    def has_nodes(self, vids):
        """Return true if the nodes exist.

        Parameters
        ----------
        vid : list, tensor
            The nodes

        Returns
        -------
        tensor
            0-1 array indicating existence
        """
        vids = utils.toindex(vids)
        rst = self._graph.has_nodes(vids)
        return rst.tousertensor()

    def has_edge(self, u, v):
        """Return true if the edge exists.

        Parameters
        ----------
        u : int
            The src node.
        v : int
            The dst node.

        Returns
        -------
        bool
            True if the edge exists
        """
        return self._graph.has_edge(u, v)

    def has_edges(self, u, v):
        """Return true if the edge exists.

        Parameters
        ----------
        u : list, tensor
            The src nodes.
        v : list, tensor
            The dst nodes.

        Returns
        -------
        tensor
            0-1 array indicating existence
        """
        u = utils.toindex(u)
        v = utils.toindex(v)
        rst = self._graph.has_edges(u, v)
        return rst.tousertensor()

    def predecessors(self, v, radius=1):
        """Return the predecessors of the node.

        Parameters
        ----------
        v : int
            The node.
        radius : int, optional
            The radius of the neighborhood.

        Returns
        -------
        tensor
            Array of predecessors
        """
        return self._graph.predecessors(v).tousertensor()

    def successors(self, v, radius=1):
        """Return the successors of the node.

        Parameters
        ----------
        v : int
            The node.
        radius : int, optional
            The radius of the neighborhood.

        Returns
        -------
        tensor
            Array of successors
        """
        return self._graph.successors(v).tousertensor()

    def edge_id(self, u, v):
        """Return the id of the edge.

        Parameters
        ----------
        u : int
            The src node.
        v : int
            The dst node.

        Returns
        -------
        int
            The edge id.
        """
        return self._graph.edge_id(u, v)

    def edge_ids(self, u, v):
        """Return the edge ids.

        Parameters
        ----------
        u : list, tensor
            The src nodes.
        v : list, tensor
            The dst nodes.

        Returns
        -------
        tensor
            The edge id array.
        """
        u = utils.toindex(u)
        v = utils.toindex(v)
        rst = self._graph.edge_ids(u, v)
        return rst.tousertensor()

    def in_edges(self, v):
        """Return the in edges of the node(s).

        Parameters
        ----------
        v : int, list, tensor
            The node(s).
        
        Returns
        -------
        tensor
            The src nodes.
        tensor
            The dst nodes.
        tensor
            The edge ids.
        """
        v = utils.toindex(v)
        src, dst, eid = self._graph.in_edges(v)
        return src.tousertensor(), dst.tousertensor(), eid.tousertensor()

    def out_edges(self, v):
        """Return the out edges of the node(s).

        Parameters
        ----------
        v : int, list, tensor
            The node(s).
        
        Returns
        -------
        tensor
            The src nodes.
        tensor
            The dst nodes.
        tensor
            The edge ids.
        """
        v = utils.toindex(v)
        src, dst, eid = self._graph.out_edges(v)
        return src.tousertensor(), dst.tousertensor(), eid.tousertensor()

    def edges(self, sorted=False):
        """Return all the edges.

        Parameters
        ----------
        sorted : bool
            True if the returned edges are sorted by their src and dst ids.
        
        Returns
        -------
        tensor
            The src nodes.
        tensor
            The dst nodes.
        tensor
            The edge ids.
        """
        src, dst, eid = self._graph.edges(sorted)
        return src.tousertensor(), dst.tousertensor(), eid.tousertensor()

    def in_degree(self, v):
        """Return the in degree of the node.

        Parameters
        ----------
        v : int
            The node.

        Returns
        -------
        int
            The in degree.
        """
        return self._graph.in_degree(v)

    def in_degrees(self, v):
        """Return the in degrees of the nodes.

        Parameters
        ----------
        v : list, tensor
            The nodes.

        Returns
        -------
        tensor
            The in degree array.
        """
        return self._graph.in_degrees(v).tousertensor()

    def out_degree(self, v):
        """Return the out degree of the node.

        Parameters
        ----------
        v : int
            The node.

        Returns
        -------
        int
            The out degree.
        """
        return self._graph.out_degree(v)

    def out_degrees(self, v):
        """Return the out degrees of the nodes.

        Parameters
        ----------
        v : list, tensor
            The nodes.

        Returns
        -------
        tensor
            The out degree array.
        """
        return self._graph.out_degrees(v).tousertensor()

    def to_networkx(self, node_attrs=None, edge_attrs=None):
        """Convert to networkx graph.

        The edge id will be saved as the 'id' edge attribute.

        Parameters
        ----------
        node_attrs : iterable of str, optional
            The node attributes to be copied.
        edge_attrs : iterable of str, optional
            The edge attributes to be copied.

        Returns
        -------
        networkx.DiGraph
            The nx graph
        """
        nx_graph = self._graph.to_networkx()
        #TODO: attributes
        return nx_graph

    def from_networkx(self, nx_graph, node_attrs=None, edge_attrs=None):
        """Convert from networkx graph.

        If 'id' edge attribute exists, the edge will be added follows
        the edge id order. Otherwise, order is undefined.
        
        Parameters
        ----------
        nx_graph : networkx.DiGraph
            The nx graph
        node_attrs : iterable of str, optional
            The node attributes needs to be copied.
        edge_attrs : iterable of str, optional
            The edge attributes needs to be copied.
        """
        self.clear()
        self._graph.from_networkx(nx_graph)
        self._msg_graph.add_nodes(self._graph.number_of_nodes())
        # copy attributes
        def _batcher(lst):
            if isinstance(lst[0], Tensor):
                return F.pack([F.unsqueeze(x, 0) for x in lst])
            else:
                return F.tensor(lst)
        if node_attrs is not None:
            attr_dict = {attr : [] for attr in node_attrs}
            for nid in range(self.number_of_nodes()):
                for attr in node_attrs:
                    attr_dict[attr].append(nx_graph.nodes[nid][attr])
            for attr in node_attrs:
                self._node_frame[attr] = _batcher(attr_dict[attr])
        if edge_attrs is not None:
            attr_dict = {attr : [] for attr in edge_attrs}
            src, dst, _ = self._graph.edges()
            for u, v in zip(src.tolist(), dst.tolist()):
                for attr in edge_attrs:
                    attr_dict[attr].append(nx_graph.edges[u, v][attr])
            for attr in edge_attrs:
                self._edge_frame[attr] = _batcher(attr_dict[attr])

    def node_attr_schemes(self):
        """Return the node attribute schemes.

        Returns
        -------
        iterable
            The set of attribute names
        """
        return self._node_frame.schemes

    def edge_attr_schemes(self):
        """Return the edge attribute schemes.

        Returns
        -------
        iterable
            The set of attribute names
        """
        return self._edge_frame.schemes

    def set_n_repr(self, hu, u=ALL, inplace=False):
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
        if utils.is_dict_like(hu):
            for key, val in hu.items():
                assert F.shape(val)[0] == num_nodes
        else:
            assert F.shape(hu)[0] == num_nodes
        # set
        if is_all(u):
            if utils.is_dict_like(hu):
                for key, val in hu.items():
                    self._node_frame[key] = val
            else:
                self._node_frame[__REPR__] = hu
        else:
            if utils.is_dict_like(hu):
                self._node_frame.update_rows(u, hu, inplace=inplace)
            else:
                self._node_frame.update_rows(u, {__REPR__ : hu}, inplace=inplace)

    def get_n_repr(self, u=ALL):
        """Get node(s) representation.

        Parameters
        ----------
        u : node, container or tensor
            The node(s).

        Returns
        -------
        dict
            Representation dict
        """
        if len(self.node_attr_schemes()) == 0:
            return dict()
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

        Returns
        -------
        Tensor
            The popped representation
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
            self.set_e_repr_by_id(h_uv, eid=ALL)
        else:
            u = utils.toindex(u)
            v = utils.toindex(v)
            eid = self._graph.edge_ids(u, v)
            self.set_e_repr_by_id(h_uv, eid=eid)

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
            num_edges = self.number_of_edges()
        else:
            eid = utils.toindex(eid)
            num_edges = len(eid)
        if utils.is_dict_like(h_uv):
            for key, val in h_uv.items():
                assert F.shape(val)[0] == num_edges
        else:
            assert F.shape(h_uv)[0] == num_edges
        # set
        if is_all(eid):
            if utils.is_dict_like(h_uv):
                for key, val in h_uv.items():
                    self._edge_frame[key] = val
            else:
                self._edge_frame[__REPR__] = h_uv
        else:
            if utils.is_dict_like(h_uv):
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

        Returns
        -------
        dict
            Representation dict
        """
        u_is_all = is_all(u)
        v_is_all = is_all(v)
        assert u_is_all == v_is_all
        if len(self.edge_attr_schemes()) == 0:
            return dict()
        if u_is_all:
            return self.get_e_repr_by_id(eid=ALL)
        else:
            u = utils.toindex(u)
            v = utils.toindex(v)
            eid = self._graph.edge_ids(u, v)
            return self.get_e_repr_by_id(eid=eid)

    def pop_e_repr(self, key=__REPR__):
        """Get and remove the specified edge repr.

        Parameters
        ----------
        key : str
          The attribute name.

        Returns
        -------
        Tensor
            The popped representation
        """
        return self._edge_frame.pop(key)

    def get_e_repr_by_id(self, eid=ALL):
        """Get edge(s) representation by edge id.

        Parameters
        ----------
        eid : int, container or tensor
          The edge id(s).

        Returns
        -------
        dict
            Representation dict
        """
        if len(self.edge_attr_schemes()) == 0:
            return dict()
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

    def register_apply_node_func(self,
                                 apply_node_func,
                                 batchable=False):
        """Register global node apply function.

        Parameters
        ----------
        apply_node_func : callable
          Apply function on the node.
        batchable : bool
          Whether the provided function allows batch computing.
        """
        self._apply_node_func = (apply_node_func, batchable)

    def register_apply_edge_func(self,
                                 apply_edge_func,
                                 batchable=False):
        """Register global edge apply function.

        Parameters
        ----------
        apply_edge_func : callable
          Apply function on the edge.
        batchable : bool
          Whether the provided function allows batch computing.
        """
        self._apply_edge_func = (apply_edge_func, batchable)

    def apply_nodes(self, v, apply_node_func="default", batchable=False):
        """Apply the function on node representations.

        Parameters
        ----------
        v : int, iterable of int, tensor
          The node id(s).
        apply_node_func : callable
          The apply node function.
        batchable : bool
          Whether the provided function allows batch computing.
        """
        if apply_node_func == "default":
            apply_node_func, batchable = self._apply_node_func
        if not apply_node_func:
            # Skip none function call.
            return
        if batchable:
            new_repr = apply_node_func(self.get_n_repr(v))
            self.set_n_repr(new_repr, v)
        else:
            raise RuntimeError('Disabled')
            if is_all(v):
                v = self.nodes()
            v = utils.toindex(v)
            for vv in utils.node_iter(v):
                ret = apply_node_func(_get_repr(self.nodes[vv]))
                _set_repr(self.nodes[vv], ret)

    def apply_edges(self, u, v, apply_edge_func="default", batchable=False):
        """Apply the function on edge representations.

        Parameters
        ----------
        u : int, iterable of int, tensor
          The src node id(s).
        v : int, iterable of int, tensor
          The dst node id(s).
        apply_edge_func : callable
          The apply edge function.
        batchable : bool
          Whether the provided function allows batch computing.
        """
        if apply_edge_func == "default":
            apply_edge_func, batchable = self._apply_edge_func
        if not apply_edge_func:
            # Skip none function call.
            return
        if batchable:
            new_repr = apply_edge_func(self.get_e_repr(u, v))
            self.set_e_repr(new_repr, u, v)
        else:
            if is_all(u) == is_all(v):
                u, v = zip(*self.edges)
            u = utils.toindex(u)
            v = utils.toindex(v)
            for uu, vv in utils.edge_iter(u, v):
                ret = apply_edge_func(_get_repr(self.edges[uu, vv]))
                _set_repr(self.edges[uu, vv], ret)

    def send(self, u, v, message_func="default", batchable=False):
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
        message_func : callable
          The message function.
        batchable : bool
          Whether the function allows batched computation.
        """
        if message_func == "default":
            message_func, batchable = self._message_func
        assert message_func is not None
        if isinstance(message_func, (tuple, list)):
            message_func = BundledMessageFunction(message_func)
        if batchable:
            self._batch_send(u, v, message_func)
        else:
            self._nonbatch_send(u, v, message_func)

    def _nonbatch_send(self, u, v, message_func):
        raise RuntimeError('Disabled')
        if is_all(u) and is_all(v):
            u, v = self.cached_graph.edges()
        else:
            u = utils.toindex(u)
            v = utils.toindex(v)
        for uu, vv in utils.edge_iter(u, v):
            ret = message_func(_get_repr(self.nodes[uu]),
                               _get_repr(self.edges[uu, vv]))
            self.edges[uu, vv][__MSG__] = ret

    def _batch_send(self, u, v, message_func):
        if is_all(u) and is_all(v):
            u, v, _ = self._graph.edges()
            self._msg_graph.add_edges(u, v)
            # call UDF
            src_reprs = self.get_n_repr(u)
            edge_reprs = self.get_e_repr()
            msgs = message_func(src_reprs, edge_reprs)
        else:
            u = utils.toindex(u)
            v = utils.toindex(v)
            u, v = utils.edge_broadcasting(u, v)
            self._msg_graph.add_edges(u, v)
            # call UDF
            src_reprs = self.get_n_repr(u)
            edge_reprs = self.get_e_repr(u, v)
            msgs = message_func(src_reprs, edge_reprs)
        if utils.is_dict_like(msgs):
            self._msg_frame.append(msgs)
        else:
            self._msg_frame.append({__MSG__ : msgs})

    def update_edge(self, u=ALL, v=ALL, edge_func="default", batchable=False):
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
        edge_func : callable
          The update function.
        batchable : bool
          Whether the function allows batched computation.
        """
        if edge_func == "default":
            edge_func, batchable = self._edge_func
        assert edge_func is not None
        if batchable:
            self._batch_update_edge(u, v, edge_func)
        else:
            self._nonbatch_update_edge(u, v, edge_func)

    def _nonbatch_update_edge(self, u, v, edge_func):
        raise RuntimeError('Disabled')
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
            u, v = self._graph.edges()
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
            eid = self._graph.edge_ids(u, v)
            # call the UDF
            src_reprs = self.get_n_repr(u)
            dst_reprs = self.get_n_repr(v)
            edge_reprs = self.get_e_repr_by_id(eid)
            new_edge_reprs = edge_func(src_reprs, dst_reprs, edge_reprs)
            self.set_e_repr_by_id(new_edge_reprs, eid)

    def recv(self,
             u,
             reduce_func="default",
             apply_node_func="default",
             batchable=False):
        """Receive and reduce in-coming messages and update representation on node u.

        It computes the new node state using the messages sent from the predecessors
        of node u. If no message is found from the predecessors, reduce function
        will be skipped.

        The reduce function should be compatible with following signature:

            (node_reprs, batched_messages) -> node_reprs

        It computes the new node representations using the representations
        of the in-coming edges (the same concept as messages).
        The reduce function can also be pre-defined functions.

        An optinoal apply_node function could be specified and should follow following
        signature:

            node_reprs -> node_reprs

        All node_reprs and edge_reprs support tensor and dictionary types.

        Parameters
        ----------
        u : node, container or tensor
          The node to be updated.
        reduce_func : callable
          The reduce function.
        apply_node_func : callable, optional
          The update function.
        batchable : bool, optional
          Whether the reduce and update function allows batched computation.
        """
        if reduce_func == "default":
            reduce_func, batchable = self._reduce_func
        assert reduce_func is not None
        if isinstance(reduce_func, (list, tuple)):
            reduce_func = BundledReduceFunction(reduce_func)
        if batchable:
            self._batch_recv(u, reduce_func)
        else:
            self._nonbatch_recv(u, reduce_func)
        # optional apply nodes
        self.apply_nodes(u, apply_node_func, batchable)

    def _nonbatch_recv(self, u, reduce_func):
        raise RuntimeError('Disabled')
        if is_all(u):
            u = list(range(0, self.number_of_nodes()))
        else:
            u = utils.toindex(u)
        for i, uu in enumerate(utils.node_iter(u)):
            # reduce phase
            msgs_batch = [self.edges[vv, uu].pop(__MSG__)
                          for vv in self.pred[uu] if __MSG__ in self.edges[vv, uu]]
            if len(msgs_batch) != 0:
                new_repr = reduce_func(_get_repr(self.nodes[uu]), msgs_batch)
                _set_repr(self.nodes[uu], new_repr)

    def _batch_recv(self, v, reduce_func):
        if self._msg_frame.num_rows == 0:
            # no message has ever been sent
            return

        v_is_all = is_all(v)
        if v_is_all:
            v = list(range(self.number_of_nodes()))
        if len(v) == 0:
            # no vertex to be triggered.
            return
        v = utils.toindex(v)

        # degree bucketing
        degrees, v_buckets = scheduler.degree_bucketing(self._msg_graph, v)
        if degrees == [0]:
            # no message has been sent to the specified node
            return

        reordered_v = []
        new_reprs = []
        has_zero_degree = False
        for deg, v_bkt in zip(degrees, v_buckets):
            if deg == 0:
                # no need to trigger reduce func for zero-degree nodes
                has_zero_degree = True
                continue
            bkt_len = len(v_bkt)
            dst_reprs = self.get_n_repr(v_bkt)
            uu, vv, in_msg_ids = self._msg_graph.in_edges(v_bkt)
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
            reordered_v.append(v_bkt.tousertensor())
            new_reprs.append(reduce_func(dst_reprs, reshaped_in_msgs))

        # TODO: clear partial messages
        self.reset_messages()

        # Pack all reducer results together
        reordered_v = F.pack(reordered_v)
        if utils.is_dict_like(new_reprs[0]):
            keys = new_reprs[0].keys()
            new_reprs = {key : F.pack([repr[key] for repr in new_reprs])
                         for key in keys}
        else:
            new_reprs = {__REPR__ : F.pack(new_reprs)}

        if v_is_all and not has_zero_degree:
            # First do reorder and then replace the whole column.
            _, indices = F.sort(reordered_v)
            indices = utils.toindex(indices)
            new_reprs = utils.reorder(new_reprs, indices)
            self.set_n_repr(new_reprs)
        else:
            # Use setter to do reorder.
            self.set_n_repr(new_reprs, reordered_v)

    def send_and_recv(self,
                      u, v,
                      message_func="default",
                      reduce_func="default",
                      apply_node_func="default",
                      batchable=False):
        """Trigger the message function on u->v and update v.

        Parameters
        ----------
        u : node, container or tensor
          The source node(s).
        v : node, container or tensor
          The destination node(s).
        message_func : callable
          The message function.
        reduce_func : callable
          The reduce function.
        apply_node_func : callable, optional
          The update function.
        batchable : bool
          Whether the reduce and update function allows batched computation.
        """
        u = utils.toindex(u)
        v = utils.toindex(v)
        if len(u) == 0:
            # no edges to be triggered
            assert len(v) == 0
            return
        unique_v = utils.toindex(F.unique(v.tousertensor()))

        # TODO(minjie): better way to figure out `batchable` flag
        if message_func == "default":
            message_func, batchable = self._message_func
        if reduce_func == "default":
            reduce_func, _ = self._reduce_func
        assert message_func is not None
        assert reduce_func is not None

        if batchable:
            executor = scheduler.get_executor(
                    'send_and_recv', self, src=u, dst=v,
                    message_func=message_func, reduce_func=reduce_func)
        else:
            executor = None

        if executor:
            executor.run()
        else:
            self.send(u, v, message_func, batchable=batchable)
            self.recv(unique_v, reduce_func, None, batchable=batchable)
        self.apply_nodes(unique_v, apply_node_func, batchable=batchable)

    def pull(self,
             v,
             message_func="default",
             reduce_func="default",
             apply_node_func="default",
             batchable=False):
        """Pull messages from the node's predecessors and then update it.

        Parameters
        ----------
        v : node, container or tensor
          The node to be updated.
        message_func : callable
          The message function.
        reduce_func : callable
          The reduce function.
        apply_node_func : callable, optional
          The update function.
        batchable : bool
          Whether the reduce and update function allows batched computation.
        """
        v = utils.toindex(v)
        if len(v) == 0:
            return
        uu, vv, _ = self._graph.in_edges(v)
        self.send_and_recv(uu, vv, message_func, reduce_func,
                apply_node_func=None, batchable=batchable)
        unique_v = F.unique(v.tousertensor())
        self.apply_nodes(unique_v, apply_node_func, batchable=batchable)

    def push(self,
             u,
             message_func="default",
             reduce_func="default",
             apply_node_func="default",
             batchable=False):
        """Send message from the node to its successors and update them.

        Parameters
        ----------
        u : node, container or tensor
          The node that sends out messages.
        message_func : callable
          The message function.
        reduce_func : callable
          The reduce function.
        apply_node_func : callable
          The update function.
        batchable : bool
          Whether the reduce and update function allows batched computation.
        """
        u = utils.toindex(u)
        if len(u) == 0:
            return
        uu, vv, _ = self._graph.out_edges(u)
        self.send_and_recv(uu, vv, message_func,
                reduce_func, apply_node_func, batchable=batchable)

    def update_all(self,
                   message_func="default",
                   reduce_func="default",
                   apply_node_func="default",
                   batchable=False):
        """Send messages through all the edges and update all nodes.

        Parameters
        ----------
        message_func : callable
          The message function.
        reduce_func : callable
          The reduce function.
        apply_node_func : callable, optional
          The update function.
        batchable : bool
          Whether the reduce and update function allows batched computation.
        """
        if message_func == "default":
            message_func, batchable = self._message_func
        if reduce_func == "default":
            reduce_func, _ = self._reduce_func
        assert message_func is not None
        assert reduce_func is not None

        if batchable:
            executor = scheduler.get_executor(
                    "update_all", self, message_func=message_func, reduce_func=reduce_func)
        else:
            executor = None

        if executor:
            executor.run()
        else:
            self.send(ALL, ALL, message_func, batchable=batchable)
            self.recv(ALL, reduce_func, None, batchable=batchable)
        self.apply_nodes(ALL, apply_node_func, batchable=batchable)

    def propagate(self,
                  iterator='bfs',
                  message_func="default",
                  reduce_func="default",
                  apply_node_func="default",
                  batchable=False,
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
        apply_node_func : str or callable
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
                self.send_and_recv(u, v,
                        message_func, reduce_func, apply_node_func, batchable)

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

def _get_repr(attr_dict):
    if len(attr_dict) == 1 and __REPR__ in attr_dict:
        return attr_dict[__REPR__]
    else:
        return attr_dict

def _set_repr(attr_dict, attr):
    if utils.is_dict_like(attr):
        attr_dict.update(attr)
    else:
        attr_dict[__REPR__] = attr
