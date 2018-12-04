"""Base graph class specialized for neural networks on graphs."""
from __future__ import absolute_import

import networkx as nx
import numpy as np

import dgl
from .base import ALL, is_all, DGLError, dgl_warning
from . import backend as F
from .frame import FrameRef, Frame, merge_frames
from .graph_index import GraphIndex, create_graph_index
from .runtime import ir, scheduler, Runtime
from . import utils
from .view import NodeView, EdgeView
from .udf import NodeBatch, EdgeBatch

__all__ = ['DGLGraph']

class DGLGraph(object):
    """Base graph class.

    The graph stores nodes, edges and also their features.

    DGL graph is always directional. Undirected graph can be represented using
    two bi-directional edges.

    Nodes are identified by consecutive integers starting from zero.

    Edges can be specified by two end points (u, v) or the integer id assigned
    when the edges are added.  Edge IDs are automatically assigned by the order
    of addition, i.e. the first edge being added has an ID of 0, the second
    being 1, so on so forth.

    Node and edge features are stored as a dictionary from the feature name
    to the feature data (in tensor).

    Parameters
    ----------
    graph_data : graph data, optional
        Data to initialize graph. Same as networkx's semantics.
    node_frame : FrameRef, optional
        Node feature storage.
    edge_frame : FrameRef, optional
        Edge feature storage.
    multigraph : bool, optional
        Whether the graph would be a multigraph (default: False)
    readonly : bool, optional
        Whether the graph structure is read-only (default: False).

    Examples
    --------
    Create an empty graph with no nodes and edges.

    >>> G = dgl.DGLGraph()

    G can be grown in several ways.

    **Nodes:**

    Add N nodes:

    >>> G.add_nodes(10)  # 10 isolated nodes are added

    **Edges:**

    Add one edge at a time,

    >>> G.add_edge(0, 1)

    or multiple edges,

    >>> G.add_edges([1, 2, 3], [3, 4, 5])  # three edges: 1->3, 2->4, 3->5

    or multiple edges starting from the same node,

    >>> G.add_edges(4, [7, 8, 9])  # three edges: 4->7, 4->8, 4->9

    or multiple edges pointing to the same node,

    >>> G.add_edges([2, 6, 8], 5)  # three edges: 2->5, 6->5, 8->5

    or multiple edges using tensor type (demo in pytorch syntax).

    >>> import torch as th
    >>> G.add_edges(th.tensor([3, 4, 5]), 1)  # three edges: 3->1, 4->1, 5->1

    NOTE: Removing nodes and edges is not supported by DGLGraph.

    **Features:**

    Both nodes and edges can have feature data. Features are stored as
    key/value pair. The key must be hashable while the value must be tensor
    type. Features are batched on the first dimension.

    Use G.ndata to get/set features for all nodes.

    >>> G = dgl.DGLGraph()
    >>> G.add_nodes(3)
    >>> G.ndata['x'] = th.zeros((3, 5))  # init 3 nodes with zero vector(len=5)
    >>> G.ndata
    {'x' : tensor([[0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.]])}

    Use G.nodes to get/set features for some nodes.

    >>> G.nodes[[0, 2]].data['x'] = th.ones((2, 5))
    >>> G.ndata
    {'x' : tensor([[1., 1., 1., 1., 1.],
                   [0., 0., 0., 0., 0.],
                   [1., 1., 1., 1., 1.]])}

    Similarly, use G.edata and G.edges to get/set features for edges.

    >>> G.add_edges([0, 1], 2)  # 0->2, 1->2
    >>> G.edata['y'] = th.zeros((2, 4))  # init 2 edges with zero vector(len=4)
    >>> G.edata
    {'y' : tensor([[0., 0., 0., 0.],
                   [0., 0., 0., 0.]])}
    >>> G.edges[1, 2].data['y'] = th.ones((1, 4))
    >>> G.edata
    {'y' : tensor([[0., 0., 0., 0.],
                   [1., 1., 1., 1.]])}

    Note that each edge is assigned a unique id equal to its adding
    order. So edge 1->2 has id=1. DGL supports directly use edge id
    to access edge features.

    >>> G.edges[0].data['y'] += 2.
    >>> G.edata
    {'y' : tensor([[2., 2., 2., 2.],
                   [1., 1., 1., 1.]])}
    """
    def __init__(self,
                 graph_data=None,
                 node_frame=None,
                 edge_frame=None,
                 multigraph=False,
                 readonly=False):
        # graph
        self._readonly=readonly
        self._graph = create_graph_index(graph_data, multigraph, readonly)
        # frame
        if node_frame is None:
            self._node_frame = FrameRef(Frame(num_rows=self.number_of_nodes()))
        else:
            self._node_frame = node_frame
        if edge_frame is None:
            self._edge_frame = FrameRef(Frame(num_rows=self.number_of_edges()))
        else:
            self._edge_frame = edge_frame
        # msg graph & frame
        self._msg_graph = create_graph_index(multigraph=multigraph)
        self._msg_frame = FrameRef()
        self.reset_messages()
        # registered functions
        self._message_func = None
        self._reduce_func = None
        self._apply_node_func = None
        self._apply_edge_func = None

    def add_nodes(self, num, data=None):
        """Add multiple new nodes.

        Parameters
        ----------
        num : int
            Number of nodes to be added.
        data : dict, optional
            Feature data of the added nodes.

        Notes
        -----
        If new nodes are added with features, and any of the old nodes
        do not have some of the feature fields, those fields are filled
        by initializers defined with ``set_n_initializer`` (default filling
        with zeros).

        Examples
        --------
        >>> G = dgl.DGLGraph()
        >>> g.add_nodes(2)
        >>> g.number_of_nodes()
        2
        >>> g.add_nodes(3)
        >>> g.number_of_nodes()
        5

        Adding new nodes with features (using PyTorch as example):
        >>> import torch as th
        >>> g.add_nodes(2, {'x': th.ones(2, 4)})    # default zero initializer
        >>> g.ndata['x']
        tensor([[0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.]])
        """
        self._graph.add_nodes(num)
        self._msg_graph.add_nodes(num)
        if data is None:
            # Initialize feature placeholders if there are features existing
            self._node_frame.add_rows(num)
        else:
            self._node_frame.append(data)

    def add_edge(self, u, v, data=None):
        """Add one new edge between u and v.

        Parameters
        ----------
        u : int
            The source node ID.  Must exist in the graph.
        v : int
            The destination node ID.  Must exist in the graph.
        data : dict, optional
            Feature data of the added edges.

        Notes
        -----
        If new edges are added with features, and any of the old edges
        do not have some of the feature fields, those fields are filled
        by initializers defined with ``set_e_initializer`` (default filling
        with zeros).

        Examples
        --------
        The following example uses PyTorch backend.
        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(3)
        >>> G.add_edge(0, 1)

        Adding new edge with features
        >>> import torch as th
        >>> G.add_edge(0, 2, {'x': th.ones(1, 4)})
        >>> G.edges()
        (tensor([0, 0]), tensor([1, 2]))
        >>> G.edata['x']
        tensor([[0., 0., 0., 0.],
                [1., 1., 1., 1.]])
        >>> G.edges[0, 2].data['x']
        tensor([[1., 1., 1., 1.]])

        See Also
        --------
        add_edges
        """
        self._graph.add_edge(u, v)
        if data is None:
            # Initialize feature placeholders if there are features existing
            self._edge_frame.add_rows(1)
        else:
            self._edge_frame.append(data)

    def add_edges(self, u, v, data=None):
        """Add multiple edges for list of source nodes u and destination nodes
        v.  A single edge is added between every pair of ``u[i]`` and ``v[i]``.

        Parameters
        ----------
        u : list, tensor
            The source node IDs.  All nodes must exist in the graph.
        v : list, tensor
            The destination node IDs.  All nodes must exist in the graph.
        data : dict, optional
            Feature data of the added edges.

        Notes
        -----
        If new edges are added with features, and any of the old edges
        do not have some of the feature fields, those fields are filled
        by initializers defined with ``set_e_initializer`` (default filling
        with zeros).

        Examples
        --------
        The following example uses PyTorch backend.
        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(4)
        >>> G.add_edges([0, 2], [1, 3]) # add edges (0, 1) and (2, 3)

        Adding new edges with features
        >>> import torch as th
        >>> G.add_edges([1, 3], [2, 0], {'x': th.ones(2, 4)}) # (1, 2), (3, 0)
        >>> G.edata['x']
        tensor([[0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.]])

        See Also
        --------
        add_edge
        """
        u = utils.toindex(u)
        v = utils.toindex(v)
        self._graph.add_edges(u, v)
        if data is None:
            # Initialize feature placeholders if there are features existing
            # NOTE: use max due to edge broadcasting syntax
            self._edge_frame.add_rows(max(len(u), len(v)))
        else:
            self._edge_frame.append(data)

    def clear(self):
        """Remove all nodes and edges, as well as their features, from the
        graph.

        Examples
        --------
        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(4)
        >>> G.add_edges([0, 1, 2, 3], [1, 2, 3, 0])
        >>> G.number_of_nodes()
        4
        >>> G.number_of_edges()
        4
        >>> G.clear()
        >>> G.number_of_nodes()
        0
        >>> G.number_of_edges()
        0
        """
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
        """Return the number of nodes in the graph.

        Returns
        -------
        int
            The number of nodes
        """
        return self._graph.number_of_nodes()

    def __len__(self):
        """Return the number of nodes in the graph."""
        return self.number_of_nodes()

    @property
    def is_multigraph(self):
        """True if the graph is a multigraph, False otherwise.
        """
        return self._graph.is_multigraph()

    def number_of_edges(self):
        """Return the number of edges in the graph.

        Returns
        -------
        int
            The number of edges
        """
        return self._graph.number_of_edges()

    def has_node(self, vid):
        """Return True if the graph contains node `vid`.

        Identical to `vid in G`.

        Parameters
        ----------
        vid : int
            The node ID.

        Returns
        -------
        bool
            True if the node exists

        Examples
        --------
        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(3)
        >>> G.has_node(0)
        True
        >>> G.has_node(4)
        False

        Equivalently,
        >>> 0 in G
        True

        See Also
        --------
        has_nodes
        """
        return self._graph.has_node(vid)

    def __contains__(self, vid):
        """Return True if the graph contains node `vid`.

        Examples
        --------
        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(3)
        >>> 0 in G
        True
        """
        return self._graph.has_node(vid)

    def has_nodes(self, vids):
        """Return a 0-1 array `a` given the node ID array `vids`.

        `a[i]` is 1 if the graph contains node `vids[i]`, 0 otherwise.

        Parameters
        ----------
        vid : list or tensor
            The array of node IDs.

        Returns
        -------
        a : tensor
            0-1 array indicating existence

        Examples
        --------
        The following example uses PyTorch backend.
        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(3)
        >>> G.has_nodes([0, 1, 2, 3, 4])
        tensor([1, 1, 1, 0, 0])

        See Also
        --------
        has_node
        """
        vids = utils.toindex(vids)
        rst = self._graph.has_nodes(vids)
        return rst.tousertensor()

    def has_edge_between(self, u, v):
        """Return True if the edge (u, v) is in the graph.

        Parameters
        ----------
        u : int
            The source node ID.
        v : int
            The destination node ID.

        Returns
        -------
        bool
            True if the edge is in the graph, False otherwise.

        Examples
        --------
        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(3)
        >>> G.add_edge(0, 1)
        >>> G.has_edge_between(0, 1)
        True
        >>> G.has_edge_between(1, 0)
        False

        See Also
        --------
        has_edges_between
        """
        return self._graph.has_edge_between(u, v)

    def has_edges_between(self, u, v):
        """Return a 0-1 array `a` given the source node ID array `u` and
        destination node ID array `v`.

        `a[i]` is 1 if the graph contains edge `(u[i], v[i])`, 0 otherwise.

        Parameters
        ----------
        u : list, tensor
            The source node ID array.
        v : list, tensor
            The destination node ID array.

        Returns
        -------
        a : tensor
            0-1 array indicating existence.

        Examples
        --------
        The following example uses PyTorch backend.
        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(3)
        >>> G.add_edges([0, 0], [1, 2]) # (0, 1), (0, 2)

        Check if (0, 1), (0, 2), (1, 0), (2, 0) exist in the graph above:
        >>> G.has_edges_between([0, 0, 1, 2], [1, 2, 0, 0])
        tensor([1, 1, 0, 0])

        See Also
        --------
        has_edge_between
        """
        u = utils.toindex(u)
        v = utils.toindex(v)
        rst = self._graph.has_edges_between(u, v)
        return rst.tousertensor()

    def predecessors(self, v):
        """Return the predecessors of node `v` in the graph.

        Node `u` is a predecessor of `v` if an edge `(u, v)` exist in the
        graph.

        Parameters
        ----------
        v : int
            The node.

        Returns
        -------
        tensor
            Array of predecessor node IDs.

        Examples
        --------
        The following example uses PyTorch backend.
        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(3)
        >>> G.add_edges([1, 2], [0, 0]) # (1, 0), (2, 0)
        >>> G.predecessors(0)
        tensor([1, 2])

        See Also
        --------
        successors
        """
        return self._graph.predecessors(v).tousertensor()

    def successors(self, v):
        """Return the successors of node `v` in the graph.

        Node `u` is a successor of `v` if an edge `(v, u)` exist in the
        graph.

        Parameters
        ----------
        v : int
            The node.

        Returns
        -------
        tensor
            Array of successor node IDs.

        Examples
        --------
        The following example uses PyTorch backend.
        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(3)
        >>> G.add_edges([0, 0], [1, 2]) # (0, 1), (0, 2)
        >>> G.successors(0)
        tensor([1, 2])

        See Also
        --------
        predecessors
        """
        return self._graph.successors(v).tousertensor()

    def edge_id(self, u, v, force_multi=False):
        """Return the edge ID, or an array of edge IDs, between source node
        `u` and destination node `v`.

        Parameters
        ----------
        u : int
            The source node ID.
        v : int
            The destination node ID.
        force_multi : bool
            If False, will return a single edge ID if the graph is a simple graph.
            If True, will always return an array.

        Returns
        -------
        int or tensor
            The edge ID if force_multi == True and the graph is a simple graph.
            The edge ID array otherwise.

        Examples
        --------
        The following example uses PyTorch backend.

        For simple graphs:
        >>> G = dgl.DGLGraph()
        >>> G.add_node(3)
        >>> G.add_edges([0, 0], [1, 2]) # (0, 1), (0, 2)
        >>> G.edge_id(0, 2)
        1
        >>> G.edge_id(0, 1)
        0

        For multigraphs:
        >>> G = dgl.DGLGraph(multigraph=True)
        >>> G.add_nodes(3)

        Adding edges (0, 1), (0, 2), (0, 1), (0, 2), so edge ID 0 and 2 both
        connect from 0 and 1, while edge ID 1 and 3 both connect from 0 and 2.
        >>> G.add_edges([0, 0, 0, 0], [1, 2, 1, 2])
        >>> G.edge_id(0, 1)
        tensor([0, 2])

        See Also
        --------
        edge_ids
        """
        idx = self._graph.edge_id(u, v)
        return idx.tousertensor() if force_multi or self.is_multigraph else idx[0]

    def edge_ids(self, u, v, force_multi=False):
        """Return all edge IDs between source node array `u` and destination
        node array `v`.

        Parameters
        ----------
        u : list, tensor
            The source node ID array.
        v : list, tensor
            The destination node ID array.
        force_multi : bool
            Whether to always treat the graph as a multigraph.

        Returns
        -------
        tensor, or (tensor, tensor, tensor)
            If the graph is a simple graph and `force_multi` is False, return
            a single edge ID array `e`.  `e[i]` is the edge ID between `u[i]`
            and `v[i]`.
            Otherwise, return three arrays `(eu, ev, e)`.  `e[i]` is the ID
            of an edge between `eu[i]` and `ev[i]`.  All edges between `u[i]`
            and `v[i]` are returned.

        Notes
        -----
        If the graph is a simple graph, `force_multi` is False, and no edge
        exist between some pairs of `u[i]` and `v[i]`, the result is undefined.

        Examples
        --------
        The following example uses PyTorch backend.

        For simple graphs:
        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(3)
        >>> G.add_edges([0, 0], [1, 2]) # (0, 1), (0, 2)
        >>> G.edge_ids([0, 0], [2, 1])  # get edge ID of (0, 2) and (0, 1)
        >>> G.edge_ids([0, 0], [2, 1])
        tensor([1, 0])

        For multigraphs
        >>> G = dgl.DGLGraph(multigraph=True)
        >>> G.add_nodes(4)
        >>> G.add_edges([0, 0, 0], [1, 1, 2])   # (0, 1), (0, 1), (0, 2)

        Get all edges between (0, 1), (0, 2), (0, 3).  Note that there is no
        edge between 0 and 3:
        >>> G.edge_ids([0, 0, 0], [1, 2, 3])
        (tensor([0, 0, 0]), tensor([1, 1, 2]), tensor([0, 1, 2]))

        See Also
        --------
        edge_id
        """
        u = utils.toindex(u)
        v = utils.toindex(v)
        src, dst, eid = self._graph.edge_ids(u, v)
        if force_multi or self.is_multigraph:
            return src.tousertensor(), dst.tousertensor(), eid.tousertensor()
        else:
            return eid.tousertensor()

    def find_edges(self, eid):
        """Given an edge ID array, return the source and destination node ID
        array `s` and `d`.  `s[i]` and `d[i]` are source and destination node
        ID for edge `eid[i]`.

        Parameters
        ----------
        eid : list, tensor
            The edge ID array.

        Returns
        -------
        tensor
            The source node ID array.
        tensor
            The destination node ID array.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(3)
        >>> G.add_edges([0, 0, 1], [1, 2, 2])   # (0, 1), (0, 2), (1, 2)
        >>> G.find_edges([0, 2])
        (tensor([0, 1]), tensor([1, 2]))
        """
        eid = utils.toindex(eid)
        src, dst, _ = self._graph.find_edges(eid)
        return src.tousertensor(), dst.tousertensor()

    def in_edges(self, v, form='uv'):
        """Return the inbound edges of the node(s).

        Parameters
        ----------
        v : int, list, tensor
            The node(s).
        form : str, optional
            The return form. Currently support:
            - 'all' : a tuple (u, v, eid)
            - 'uv'  : a pair (u, v), default
            - 'eid' : one eid tensor

        Returns
        -------
        A tuple of Tensors `(eu, ev, eid)` if `form == 'all'`.
            `eid[i]` is the ID of an inbound edge to `ev[i]` from `eu[i]`.
            All inbound edges to `v` are returned.
        A pair of Tensors (eu, ev) if form == 'uv'
            `eu[i]` is the source node of an inbound edge to `ev[i]`.
            All inbound edges to `v` are returned.
        One Tensor if form == 'eid'
            `eid[i]` is ID of an inbound edge to any of the nodes in `v`.

        Examples
        --------
        The following example uses PyTorch backend.
        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(3)
        >>> G.add_edges([0, 0, 1], [1, 2, 2])   # (0, 1), (0, 2), (1, 2)

        For a single node:
        >>> G.in_edges(2)
        (tensor([0, 1]), tensor([2, 2]))
        >>> G.in_edges(2, 'all')
        (tensor([0, 1]), tensor([2, 2]), tensor([1, 2]))
        >>> G.in_edges(2, 'eid')
        tensor([1, 2])

        For multiple nodes:
        >>> G.in_edges([1, 2])
        (tensor([0, 0, 1]), tensor([1, 2, 2]))
        >>> G.in_edges([1, 2], 'all')
        (tensor([0, 0, 1]), tensor([1, 2, 2]), tensor([0, 1, 2]))
        """
        v = utils.toindex(v)
        src, dst, eid = self._graph.in_edges(v)
        if form == 'all':
            return (src.tousertensor(), dst.tousertensor(), eid.tousertensor())
        elif form == 'uv':
            return (src.tousertensor(), dst.tousertensor())
        elif form == 'eid':
            return eid.tousertensor()
        else:
            raise DGLError('Invalid form:', form)

    def out_edges(self, v, form='uv'):
        """Return the outbound edges of the node(s).

        Parameters
        ----------
        v : int, list, tensor
            The node(s).
        form : str, optional
            The return form. Currently support:
            - 'all' : a tuple (u, v, eid)
            - 'uv'  : a pair (u, v), default
            - 'eid' : one eid tensor

        Returns
        -------
        A tuple of Tensors `(eu, ev, eid)` if `form == 'all'`.
            `eid[i]` is the ID of an outbound edge from `eu[i]` from `ev[i]`.
            All outbound edges from `v` are returned.
        A pair of Tensors (eu, ev) if form == 'uv'
            `ev[i]` is the destination node of an outbound edge from `eu[i]`.
            All outbound edges from `v` are returned.
        One Tensor if form == 'eid'
            `eid[i]` is ID of an outbound edge from any of the nodes in `v`.

        Examples
        --------
        The following example uses PyTorch backend.
        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(3)
        >>> G.add_edges([0, 0, 1], [1, 2, 2])   # (0, 1), (0, 2), (1, 2)

        For a single node:
        >>> G.out_edges(0)
        (tensor([0, 0]), tensor([1, 2]))
        >>> G.out_edges(0, 'all')
        (tensor([0, 0]), tensor([1, 2]), tensor([0, 1]))
        >>> G.out_edges(0, 'eid')
        tensor([0, 1])

        For multiple nodes:
        >>> G.out_edges([0, 1])
        (tensor([0, 0, 1]), tensor([1, 2, 2]))
        >>> G.out_edges([0, 1], 'all')
        (tensor([0, 0, 1]), tensor([1, 2, 2]), tensor([0, 1, 2]))
        """
        v = utils.toindex(v)
        src, dst, eid = self._graph.out_edges(v)
        if form == 'all':
            return (src.tousertensor(), dst.tousertensor(), eid.tousertensor())
        elif form == 'uv':
            return (src.tousertensor(), dst.tousertensor())
        elif form == 'eid':
            return eid.tousertensor()
        else:
            raise DGLError('Invalid form:', form)

    def all_edges(self, form='uv', sorted=False):
        """Return all the edges.

        Parameters
        ----------
        form : str, optional
            The return form. Currently support:
            - 'all' : a tuple (u, v, eid)
            - 'uv'  : a pair (u, v), default
            - 'eid' : one eid tensor
        sorted : bool
            True if the returned edges are sorted by their src and dst ids.

        Returns
        -------
        A tuple of Tensors (u, v, eid) if form == 'all'
            `eid[i]` is the ID of an edge between `u[i]` and `v[i]`.
            All edges are returned.
        A pair of Tensors (u, v) if form == 'uv'
            An edge exists between `u[i]` and `v[i]`.
            If `n` edges exist between `u` and `v`, then `u` and `v` as a pair
            will appear `n` times.
        One Tensor if form == 'eid'
            `eid[i]` is the ID of an edge in the graph.

        Examples
        --------
        The following example uses PyTorch backend.
        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(3)
        >>> G.add_edges([0, 0, 1], [1, 2, 2])   # (0, 1), (0, 2), (1, 2)
        >>> G.all_edges()
        (tensor([0, 0, 1]), tensor([1, 2, 2]))
        >>> G.all_edges('all')
        (tensor([0, 0, 1]), tensor([1, 2, 2]), tensor([0, 1, 2]))
        """
        src, dst, eid = self._graph.edges(sorted)
        if form == 'all':
            return (src.tousertensor(), dst.tousertensor(), eid.tousertensor())
        elif form == 'uv':
            return (src.tousertensor(), dst.tousertensor())
        elif form == 'eid':
            return eid.tousertensor()
        else:
            raise DGLError('Invalid form:', form)

    def in_degree(self, v):
        """Return the in-degree of node `v`.

        Parameters
        ----------
        v : int
            The node ID.

        Returns
        -------
        int
            The in-degree.

        Examples
        --------
        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(3)
        >>> G.add_edges([0, 0, 1], [1, 2, 2])   # (0, 1), (0, 2), (1, 2)
        >>> G.in_degree(2)
        2

        See Also
        --------
        in_degrees
        """
        return self._graph.in_degree(v)

    def in_degrees(self, v=ALL):
        """Return the array `d` of in-degrees of the node array `v`.

        `d[i]` is the in-degree of node `v[i]`.

        Parameters
        ----------
        v : list, tensor, optional.
            The node ID array. Default is to return the degrees of all the nodes.

        Returns
        -------
        d : tensor
            The in-degree array.

        Examples
        --------
        The following example uses PyTorch backend.
        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(3)
        >>> G.add_edges([0, 0, 1], [1, 2, 2])   # (0, 1), (0, 2), (1, 2)
        >>> G.in_degrees([1, 2])
        tensor([1, 2])

        See Also
        --------
        in_degree
        """
        if is_all(v):
            v  = utils.toindex(slice(0, self.number_of_nodes()))
        else:
            v = utils.toindex(v)
        return self._graph.in_degrees(v).tousertensor()

    def out_degree(self, v):
        """Return the out-degree of node `v`.

        Parameters
        ----------
        v : int
            The node ID.

        Returns
        -------
        int
            The out-degree.

        Examples
        --------
        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(3)
        >>> G.add_edges([0, 0, 1], [1, 2, 2])   # (0, 1), (0, 2), (1, 2)
        >>> G.out_degree(0)
        2

        See Also
        --------
        out_degrees
        """
        return self._graph.out_degree(v)

    def out_degrees(self, v=ALL):
        """Return the array `d` of out-degrees of the node array `v`.

        `d[i]` is the out-degree of node `v[i]`.

        Parameters
        ----------
        v : list, tensor
            The node ID array. Default is to return the degrees of all the nodes.

        Returns
        -------
        d : tensor
            The out-degree array.

        Examples
        --------
        The following example uses PyTorch backend.
        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(3)
        >>> G.add_edges([0, 0, 1], [1, 2, 2])   # (0, 1), (0, 2), (1, 2)
        >>> G.out_degrees([0, 1])
        tensor([2, 1])

        See Also
        --------
        out_degree
        """
        if is_all(v):
            v  = utils.toindex(slice(0, self.number_of_nodes()))
        else:
            v = utils.toindex(v)
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
        if node_attrs is not None:
            for n in nx_graph.nodes:
                nf = self.get_n_repr(n)
                nx_graph.nodes[n].update({key: nf[key] for key in node_attrs})
        if edge_attrs is not None:
            for u, v, id in nx_graph.data('id'):
                ef = self.get_e_repr(id)
                nx_graph.edges[u, v].update({key: ef[key] for key in edge_attrs})
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
        self._node_frame.add_rows(self.number_of_nodes())
        self._edge_frame.add_rows(self.number_of_edges())
        self._msg_graph.add_nodes(self._graph.number_of_nodes())
        # copy attributes
        def _batcher(lst):
            if F.is_tensor(lst[0]):
                return F.cat([F.unsqueeze(x, 0) for x in lst], dim=0)
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

    def from_scipy_sparse_matrix(self, a):
        """ Convert from scipy sparse matrix.

        Parameters
        ----------
        a : scipy sparse matrix
            The graph's adjacency matrix
        """
        self.clear()
        self._graph.from_scipy_sparse_matrix(a)
        self._node_frame.add_rows(self.number_of_nodes())
        self._edge_frame.add_rows(self.number_of_edges())
        self._msg_graph.add_nodes(self._graph.number_of_nodes())

    def node_attr_schemes(self):
        """Return the node feature schemes.

        Returns
        -------
        dict of str to schemes
            The schemes of node feature columns.
        """
        return self._node_frame.schemes

    def edge_attr_schemes(self):
        """Return the edge feature schemes.

        Returns
        -------
        dict of str to schemes
            The schemes of edge feature columns.
        """
        return self._edge_frame.schemes

    def set_n_initializer(self, initializer, field=None):
        """Set the initializer for empty node features.

        Initializer is a callable that returns a tensor given the shape, data type
        and device context.

        Parameters
        ----------
        initializer : callable
            The initializer.
        field : str, optional
            The feature field name. Default is set an initializer for all the
            feature fields.

        See Also
        --------
        dgl.init.base_initializer
        """
        self._node_frame.set_initializer(initializer, field)

    def set_e_initializer(self, initializer, field=None):
        """Set the initializer for empty edge features.

        Initializer is a callable that returns a tensor given the shape, data type
        and device context.

        Parameters
        ----------
        initializer : callable
            The initializer.
        field : str, optional
            The feature field name. Default is set an initializer for all the
            feature fields.

        See Also
        --------
        dgl.init.base_initializer
        """
        self._edge_frame.set_initializer(initializer, field)

    @property
    def nodes(self):
        """Return a node view that can used to set/get feature data."""
        return NodeView(self)

    @property
    def ndata(self):
        """Return the data view of all the nodes."""
        return self.nodes[:].data

    @property
    def edges(self):
        """Return a edges view that can used to set/get feature data."""
        return EdgeView(self)

    @property
    def edata(self):
        """Return the data view of all the edges."""
        return self.edges[:].data

    def set_n_repr(self, hu, u=ALL, inplace=False):
        """Set node(s) representation.

        `hu` is a dictionary from the feature name to feature tensor. Each tensor
        is of shape (B, D1, D2, ...), where B is the number of nodes to be updated,
        and (D1, D2, ...) be the shape of the node representation tensor. The
        length of the given node ids must match B (i.e, len(u) == B).

        All update will be done out of place to work with autograd unless the
        inplace flag is true.

        Parameters
        ----------
        hu : dict of tensor
            Node representation.
        u : node, container or tensor
            The node(s).
        inplace : bool
            If True, update will be done in place, but autograd will break.
        """
        # sanity check
        if not utils.is_dict_like(hu):
            raise DGLError('Expect dictionary type for feature data.'
                           ' Got "%s" instead.' % type(hu))
        if is_all(u):
            num_nodes = self.number_of_nodes()
        else:
            u = utils.toindex(u)
            num_nodes = len(u)
        for key, val in hu.items():
            nfeats = F.shape(val)[0]
            if nfeats != num_nodes:
                raise DGLError('Expect number of features to match number of nodes (len(u)).'
                               ' Got %d and %d instead.' % (nfeats, num_nodes))
        # set
        if is_all(u):
            for key, val in hu.items():
                self._node_frame[key] = val
        else:
            self._node_frame.update_rows(u, hu, inplace=inplace)

    def get_n_repr(self, u=ALL):
        """Get node(s) representation.

        The returned feature tensor batches multiple node features on the first dimension.

        Parameters
        ----------
        u : node, container or tensor
            The node(s).

        Returns
        -------
        dict
            Representation dict from feature name to feature tensor.
        """
        if len(self.node_attr_schemes()) == 0:
            return dict()
        if is_all(u):
            return dict(self._node_frame)
        else:
            u = utils.toindex(u)
            return self._node_frame.select_rows(u)

    def pop_n_repr(self, key):
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

    def set_e_repr(self, he, edges=ALL, inplace=False):
        """Set edge(s) representation.

        `he` is a dictionary from the feature name to feature tensor. Each tensor
        is of shape (B, D1, D2, ...), where B is the number of edges to be updated,
        and (D1, D2, ...) be the shape of the edge representation tensor.

        All update will be done out of place to work with autograd unless the
        inplace flag is true.

        Parameters
        ----------
        he : tensor or dict of tensor
            Edge representation.
        edges : edges
            Edges can be a pair of endpoint nodes (u, v), or a
            tensor of edge ids. The default value is all the edges.
        inplace : bool
            If True, update will be done in place, but autograd will break.
        """
        # parse argument
        if is_all(edges):
            eid = ALL
        elif isinstance(edges, tuple):
            u, v = edges
            u = utils.toindex(u)
            v = utils.toindex(v)
            # Rewrite u, v to handle edge broadcasting and multigraph.
            _, _, eid = self._graph.edge_ids(u, v)
        else:
            eid = utils.toindex(edges)

        # sanity check
        if not utils.is_dict_like(he):
            raise DGLError('Expect dictionary type for feature data.'
                           ' Got "%s" instead.' % type(he))

        if is_all(eid):
            num_edges = self.number_of_edges()
        else:
            eid = utils.toindex(eid)
            num_edges = len(eid)
        for key, val in he.items():
            nfeats = F.shape(val)[0]
            if nfeats != num_edges:
                raise DGLError('Expect number of features to match number of edges.'
                               ' Got %d and %d instead.' % (nfeats, num_edges))
        # set
        if is_all(eid):
            # update column
            for key, val in he.items():
                self._edge_frame[key] = val
        else:
            # update row
            self._edge_frame.update_rows(eid, he, inplace=inplace)

    def get_e_repr(self, edges=ALL):
        """Get node(s) representation.

        Parameters
        ----------
        edges : edges
            Edges can be a pair of endpoint nodes (u, v), or a
            tensor of edge ids. The default value is all the edges.

        Returns
        -------
        dict
            Representation dict
        """
        if len(self.edge_attr_schemes()) == 0:
            return dict()
        # parse argument
        if is_all(edges):
            eid = ALL
        elif isinstance(edges, tuple):
            u, v = edges
            u = utils.toindex(u)
            v = utils.toindex(v)
            # Rewrite u, v to handle edge broadcasting and multigraph.
            _, _, eid = self._graph.edge_ids(u, v)
        else:
            eid = utils.toindex(edges)

        if is_all(eid):
            return dict(self._edge_frame)
        else:
            eid = utils.toindex(eid)
            return self._edge_frame.select_rows(eid)

    def pop_e_repr(self, key):
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

    def register_message_func(self, func):
        """Register global message function.

        Parameters
        ----------
        func : callable
          Message function on the edge.
        """
        self._message_func = func

    def register_reduce_func(self, func):
        """Register global message reduce function.

        Parameters
        ----------
        func : str or callable
          Reduce function on incoming edges.
        """
        self._reduce_func = func

    def register_apply_node_func(self, func):
        """Register global node apply function.

        Parameters
        ----------
        func : callable
            Apply function on the node.
        """
        self._apply_node_func = func

    def register_apply_edge_func(self, func):
        """Register global edge apply function.

        Parameters
        ----------
        edge_func : callable
            Apply function on the edge.
        """
        self._apply_edge_func = func

    def apply_nodes(self, func="default", v=ALL, inplace=False):
        """Apply the function on the node features.

        Applying a None function will be ignored.

        Parameters
        ----------
        func : callable, optional
            The UDF applied on the node features.
        v : int, iterable of int, tensor, optional
            The node id(s).
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.
        """
        if func == "default":
            func = self._apply_node_func
        if is_all(v):
            v = utils.toindex(slice(0, self.number_of_nodes()))
        else:
            v = utils.toindex(v)
        with ir.prog() as prog:
            scheduler.schedule_apply_nodes(graph=self,
                                           v=v,
                                           apply_func=func,
                                           inplace=inplace)
            Runtime.run(prog)

    def apply_edges(self, func="default", edges=ALL, inplace=False):
        """Apply the function on the edge features.

        Parameters
        ----------
        func : callable, optional
            The UDF applied on the edge features.
        edges : edges, optional
            Edges can be a pair of endpoint nodes (u, v), or a
            tensor of edge ids. The default value is all the edges.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.

        Notes
        -----
        On multigraphs, if u and v are specified, then all the edges
        between u and v will be updated.
        """
        if func == "default":
            func = self._apply_edge_func
        assert func is not None

        if is_all(edges):
            u, v, _ = self._graph.edges()
            eid = utils.toindex(slice(0, self.number_of_edges()))
        elif isinstance(edges, tuple):
            u, v = edges
            u = utils.toindex(u)
            v = utils.toindex(v)
            # Rewrite u, v to handle edge broadcasting and multigraph.
            u, v, eid = self._graph.edge_ids(u, v)
        else:
            eid = utils.toindex(edges)
            u, v, _ = self._graph.find_edges(eid)

        with ir.prog() as prog:
            scheduler.schedule_apply_edges(graph=self,
                                           u=u,
                                           v=v,
                                           eid=eid,
                                           apply_func=func,
                                           inplace=inplace)
            Runtime.run(prog)

    def send(self, edges, message_func="default"):
        """Send messages along the given edges.

        Parameters
        ----------
        edges : edges, optional
            Edges can be a pair of endpoint nodes (u, v), or a
            tensor of edge ids.
        message_func : callable
            The message function.

        Notes
        -----
        On multigraphs, if u and v are specified, then the messages will be sent
        along all edges between u and v.
        """
        if message_func == "default":
            message_func = self._message_func

        if is_all(edges):
            eid = ALL
            u, v, _ = self._graph.edges()
        elif isinstance(edges, tuple):
            u, v = edges
            u = utils.toindex(u)
            v = utils.toindex(v)
            # Rewrite u, v to handle edge broadcasting and multigraph.
            u, v, eid = self._graph.edge_ids(u, v)
        else:
            eid = utils.toindex(edges)
            u, v, _ = self._graph.find_edges(eid)

        with ir.prog() as prog:
            scheduler.schedule_send(graph=self, u=u, v=v, eid=eid,
                                    message_func=message_func)
            Runtime.run(prog)

        # update message graph and frame
        self._msg_graph.add_edges(u, v)

    def recv(self,
             v,
             reduce_func="default",
             apply_node_func="default",
             inplace=False):
        """Receive and reduce in-coming messages and update representation on node v.

        TODO(minjie): document on zero-in-degree case
        TODO(minjie): document on how returned new features are merged with the old features
        TODO(minjie): document on how many times UDFs will be called

        Parameters
        ----------
        v : node, container or tensor
          The node to be updated.
        reduce_func : callable
          The reduce function.
        apply_node_func : callable, optional
          The update function.
        inplace: bool, optional
          If True, update will be done in place, but autograd will break.
        """
        if reduce_func == "default":
            reduce_func = self._reduce_func
        if apply_node_func == "default":
            apply_node_func = self._apply_node_func
        assert reduce_func is not None

        if self._msg_frame.num_rows == 0:
            # no message has ever been sent
            return

        v_is_all = is_all(v)
        if v_is_all:
            v = F.arange(0, self.number_of_nodes())
        elif isinstance(v, int):
            v = [v]
        v = utils.toindex(v)
        if len(v) == 0:
            # no vertex to be triggered.
            return

        with ir.prog() as prog:
            scheduler.schedule_recv(graph=self,
                                    recv_nodes=v,
                                    reduce_func=reduce_func,
                                    apply_func=apply_node_func,
                                    inplace=inplace)
            Runtime.run(prog)

        # FIXME(minjie): multi send bug
        self.reset_messages()

    def send_and_recv(self,
                      edges,
                      message_func="default",
                      reduce_func="default",
                      apply_node_func="default",
                      inplace=False):
        """Send messages along edges and receive them on the targets.

        Parameters
        ----------
        edges : edges
            Edges can be a pair of endpoint nodes (u, v), or a
            tensor of edge ids. The default value is all the edges.
        message_func : callable, optional
            The message function. Registered function will be used if not
            specified.
        reduce_func : callable, optional
            The reduce function. Registered function will be used if not
            specified.
        apply_node_func : callable, optional
            The update function. Registered function will be used if not
            specified.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.

        Notes
        -----
        On multigraphs, if u and v are specified, then the messages will be sent
        and received along all edges between u and v.
        """
        if message_func == "default":
            message_func = self._message_func
        if reduce_func == "default":
            reduce_func = self._reduce_func
        if apply_node_func == "default":
            apply_node_func = self._apply_node_func

        assert message_func is not None
        assert reduce_func is not None

        if isinstance(edges, tuple):
            u, v = edges
            u = utils.toindex(u)
            v = utils.toindex(v)
            # Rewrite u, v to handle edge broadcasting and multigraph.
            u, v, eid = self._graph.edge_ids(u, v)
        else:
            eid = utils.toindex(edges)
            u, v, _ = self._graph.find_edges(eid)

        if len(u) == 0:
            # no edges to be triggered
            return

        with ir.prog() as prog:
            scheduler.schedule_snr(graph=self,
                                   edge_tuples=(u, v, eid),
                                   message_func=message_func,
                                   reduce_func=reduce_func,
                                   apply_func=apply_node_func,
                                   inplace=inplace)
            Runtime.run(prog)

    def pull(self,
             v,
             message_func="default",
             reduce_func="default",
             apply_node_func="default",
             inplace=False):
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
        inplace: bool, optional
          If True, update will be done in place, but autograd will break.
        """
        if message_func == "default":
            message_func = self._message_func
        if reduce_func == "default":
            reduce_func = self._reduce_func
        if apply_node_func == "default":
            apply_node_func = self._apply_node_func

        assert message_func is not None
        assert reduce_func is not None

        v = utils.toindex(v)
        if len(v) == 0:
            return
        with ir.prog() as prog:
            scheduler.schedule_pull(graph=self,
                                    pull_nodes=v,
                                    message_func=message_func,
                                    reduce_func=reduce_func,
                                    apply_func=apply_node_func,
                                    inplace=inplace)
            Runtime.run(prog)

    def push(self,
             u,
             message_func="default",
             reduce_func="default",
             apply_node_func="default",
             inplace=False):
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
        inplace: bool, optional
          If True, update will be done in place, but autograd will break.
        """
        if message_func == "default":
            message_func = self._message_func
        if reduce_func == "default":
            reduce_func = self._reduce_func
        if apply_node_func == "default":
            apply_node_func = self._apply_node_func

        assert message_func is not None
        assert reduce_func is not None

        u = utils.toindex(u)
        if len(u) == 0:
            return
        with ir.prog() as prog:
            scheduler.schedule_push(graph=self,
                                    u=u,
                                    message_func=message_func,
                                    reduce_func=reduce_func,
                                    apply_func=apply_node_func,
                                    inplace=inplace)
            Runtime.run(prog)

    def update_all(self,
                   message_func="default",
                   reduce_func="default",
                   apply_node_func="default"):
        """Send messages through all the edges and update all nodes.

        Parameters
        ----------
        message_func : callable
          The message function.
        reduce_func : callable
          The reduce function.
        apply_node_func : callable, optional
          The update function.
        """
        if message_func == "default":
            message_func = self._message_func
        if reduce_func == "default":
            reduce_func = self._reduce_func
        if apply_node_func == "default":
            apply_node_func = self._apply_node_func
        assert message_func is not None
        assert reduce_func is not None

        with ir.prog() as prog:
            scheduler.schedule_update_all(graph=self,
                                          message_func=message_func,
                                          reduce_func=reduce_func,
                                          apply_func=apply_node_func)
            Runtime.run(prog)

    def prop_nodes(self,
                   nodes_generator,
                   message_func="default",
                   reduce_func="default",
                   apply_node_func="default"):
        """Propagate messages using graph traversal by triggering pull() on nodes.

        The traversal order is specified by the ``nodes_generator``. It generates
        node frontiers, which is a list or a tensor of nodes. The nodes in the
        same frontier will be triggered together, while nodes in different frontiers
        will be triggered according to the generating order.

        Parameters
        ----------
        node_generators : generator
            The generator of node frontiers.
        message_func : callable, optional
            The message function.
        reduce_func : callable, optional
            The reduce function.
        apply_node_func : callable, optional
            The update function.
        """
        for node_frontier in nodes_generator:
            self.pull(node_frontier,
                    message_func, reduce_func, apply_node_func)

    def prop_edges(self,
                   edges_generator,
                   message_func="default",
                   reduce_func="default",
                   apply_node_func="default"):
        """Propagate messages using graph traversal by triggering send_and_recv() on edges.

        The traversal order is specified by the ``edges_generator``. It
        generates edge frontiers, which is a list or a tensor of edge ids or
        end points.  The edges in the same frontier will be triggered together,
        while edges in different frontiers will be triggered according to the
        generating order.

        Parameters
        ----------
        edges_generator : generator
            The generator of edge frontiers.
        message_func : callable, optional
            The message function.
        reduce_func : callable, optional
            The reduce function.
        apply_node_func : callable, optional
            The update function.
        """
        for edge_frontier in edges_generator:
            self.send_and_recv(edge_frontier,
                    message_func, reduce_func, apply_node_func)

    def subgraph(self, nodes):
        """Return the subgraph induced on given nodes.

        Parameters
        ----------
        nodes : list, or iterable
            A node ID array to construct subgraph.
            All nodes must exist in the graph.

        Returns
        -------
        G : DGLSubGraph
            The subgraph.
            The nodes are relabeled so that node `i` in the subgraph is mapped
            to node `nodes[i]` in the original graph.
            The edges are also relabeled.
            One can retrieve the mapping from subgraph node/edge ID to parent
            node/edge ID via `parent_nid` and `parent_eid` properties of the
            subgraph.

        Examples
        --------
        The following example uses PyTorch backend.
        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(5)
        >>> G.add_edges([0, 1, 2, 3, 4], [1, 2, 3, 4, 0])   # 5-node cycle
        >>> SG = G.subgraph([0, 1, 4])
        >>> SG.nodes()
        tensor([0, 1, 2])
        >>> SG.edges()
        (tensor([0, 2]), tensor([1, 0]))
        >>> SG.parent_nid
        tensor([0, 1, 4])
        >>> SG.parent_eid
        tensor([0, 4])

        See Also
        --------
        DGLSubGraph
        subgraphs
        edge_subgraph
        """
        induced_nodes = utils.toindex(nodes)
        sgi = self._graph.node_subgraph(induced_nodes)
        return dgl.DGLSubGraph(self, sgi.induced_nodes, sgi.induced_edges, sgi)

    def subgraphs(self, nodes):
        """Return a list of subgraphs, each induced in the corresponding given
        nodes in the list.

        Equivalent to
        [self.subgraph(nodes_list) for nodes_list in nodes]

        Parameters
        ----------
        nodes : a list of lists or iterable
            A list of node ID arrays to construct corresponding subgraphs.
            All nodes in all the list items must exist in the graph.

        Returns
        -------
        G : A list of DGLSubGraph
            The subgraphs.

        See Also
        --------
        DGLSubGraph
        subgraph
        """
        induced_nodes = [utils.toindex(n) for n in nodes]
        sgis = self._graph.node_subgraphs(induced_nodes)
        return [dgl.DGLSubGraph(self, sgi.induced_nodes, sgi.induced_edges,
            sgi) for sgi in sgis]

    def edge_subgraph(self, edges):
        """Return the subgraph induced on given edges.

        Parameters
        ----------
        edges : list, or iterable
            An edge ID array to construct subgraph.
            All edges must exist in the subgraph.

        Returns
        -------
        G : DGLSubGraph
            The subgraph.
            The edges are relabeled so that edge `i` in the subgraph is mapped
            to edge `edges[i]` in the original graph.
            The nodes are also relabeled.
            One can retrieve the mapping from subgraph node/edge ID to parent
            node/edge ID via `parent_nid` and `parent_eid` properties of the
            subgraph.

        Examples
        --------
        The following example uses PyTorch backend.
        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(5)
        >>> G.add_edges([0, 1, 2, 3, 4], [1, 2, 3, 4, 0])   # 5-node cycle
        >>> SG = G.edge_subgraph([0, 4])
        >>> SG.nodes()
        tensor([0, 1, 2])
        >>> SG.edges()
        (tensor([0, 2]), tensor([1, 0]))
        >>> SG.parent_nid
        tensor([0, 1, 4])
        >>> SG.parent_eid
        tensor([0, 4])

        See Also
        --------
        DGLSubGraph
        subgraph
        """
        induced_edges = utils.toindex(edges)
        sgi = self._graph.edge_subgraph(induced_edges)
        return dgl.DGLSubGraph(self, sgi.induced_nodes, sgi.induced_edges, sgi)

    def adjacency_matrix(self, transpose=False, ctx=F.cpu()):
        """Return the adjacency matrix representation of this graph.

        By default, a row of returned adjacency matrix represents the destination
        of an edge and the column represents the source.

        When transpose is True, a row represents the source and a column represents
        a destination.

        Parameters
        ----------
        transpose : bool, optional (default=False)
            A flag to tranpose the returned adjacency matrix.
        ctx : context, optional (default=cpu)
            The context of returned adjacency matrix.

        Returns
        -------
        SparseTensor
            The adjacency matrix.
        """
        return self._graph.adjacency_matrix(transpose, ctx)

    def incidence_matrix(self, type, ctx=F.cpu()):
        """Return the incidence matrix representation of this graph.

        An incidence matrix is an n x m sparse matrix, where n is
        the number of nodes and m is the number of edges. Each nnz
        value indicating whether the edge is incident to the node
        or not.

        There are three types of an incidence matrix :math:`I`:

        * "in":

          - :math:`I[v, e] = 1` if e is the in-edge of v (or v is the dst node of e);
          - :math:`I[v, e] = 0` otherwise.

        * "out":

          - :math:`I[v, e] = 1` if e is the out-edge of v (or v is the src node of e);
          - :math:`I[v, e] = 0` otherwise.

        * "both":

          - :math:`I[v, e] = 1` if e is the in-edge of v;
          - :math:`I[v, e] = -1` if e is the out-edge of v;
          - :math:`I[v, e] = 0` otherwise (including self-loop).

        Parameters
        ----------
        type : str
            Can be either "in", "out" or "both"
        ctx : context, optional (default=cpu)
            The context of returned incidence matrix.

        Returns
        -------
        SparseTensor
            The incidence matrix.
        """
        return self._graph.incidence_matrix(type, ctx)

    def line_graph(self, backtracking=True, shared=False):
        """Return the line graph of this graph.

        Parameters
        ----------
        backtracking : bool, optional
            Whether the returned line graph is backtracking.

        shared : bool, optional
            Whether the returned line graph shares representations with `self`.

        Returns
        -------
        DGLGraph
            The line graph of this graph.
        """
        graph_data = self._graph.line_graph(backtracking)
        node_frame = self._edge_frame if shared else None
        return DGLGraph(graph_data, node_frame)

    def filter_nodes(self, predicate, nodes=ALL):
        """Return a tensor of node IDs that satisfy the given predicate.

        Parameters
        ----------
        predicate : callable
            The predicate should take in a NodeBatch object, and return a
            boolean tensor with N elements indicating which node satisfy
            the predicate.
        nodes : container or tensor
            The nodes to filter on

        Returns
        -------
        tensor
            The filtered nodes
        """
        if is_all(nodes):
            v = utils.toindex(slice(0, self.number_of_nodes()))
        else:
            v = utils.toindex(nodes)

        n_repr = self.get_n_repr(v)
        nb = NodeBatch(self, v, n_repr)
        n_mask = predicate(nb)

        if is_all(nodes):
            return F.nonzero_1d(n_mask)
        else:
            nodes = F.tensor(nodes)
            return nodes[n_mask]

    def filter_edges(self, predicate, edges=ALL):
        """Return a tensor of edge IDs that satisfy the given predicate.

        Parameters
        ----------
        predicate : callable
            The predicate should take in an EdgeBatch object, and return a
            boolean tensor with E elements indicating which edge satisfy
            the predicate.
        edges : edges
            Edges can be a pair of endpoint nodes (u, v), or a
            tensor of edge ids. The default value is all the edges.

        Returns
        -------
        tensor
            The filtered edges
        """
        if is_all(edges):
            eid = ALL
            u, v, _ = self._graph.edges()
        elif isinstance(edges, tuple):
            u, v = edges
            u = utils.toindex(u)
            v = utils.toindex(v)
            # Rewrite u, v to handle edge broadcasting and multigraph.
            u, v, eid = self._graph.edge_ids(u, v)
        else:
            eid = utils.toindex(edges)
            u, v, _ = self._graph.find_edges(eid)

        src_data = self.get_n_repr(u)
        edge_data = self.get_e_repr(eid)
        dst_data = self.get_n_repr(v)
        eb = EdgeBatch(self, (u, v, eid),
                src_data, edge_data, dst_data)

        e_mask = predicate(eb)

        if is_all(edges):
            return F.nonzero_1d(e_mask)
        else:
            edges = F.tensor(edges)
            return edges[e_mask]
