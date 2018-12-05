"""Base graph class specialized for neural networks on graphs."""
from __future__ import absolute_import

import networkx as nx
import numpy as np
from collections import defaultdict

import dgl
from .base import ALL, is_all, DGLError, dgl_warning
from . import backend as F
from .frame import FrameRef, Frame
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

    or multiple edges using tensor type

    .. note:: Here we use pytorch syntax for demo. The general idea applies
        to other frameworks with minor syntax change (e.g. replace
        ``torch.tensor`` with ``mxnet.ndarray``).

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

    **Message Passing:**

    One common operation for updating node features is message passing,
    where the source nodes send messages through edges to the destinations.
    With :class:`DGLGraph`, we can do this with :func:`send` and :func:`recv`.

    In the example below, the source nodes add 1 to their node features as
    the messages and send the messages to the destinations.

    >>> # Define the function for sending messages.
    >>> def send_source(edges): return {'m': edges.src['x'] + 1}
    >>> # Set the function defined to be the default message function.
    >>> G.register_message_func(send_source)
    >>> # Send messages through all edges.
    >>> G.send(G.edges())

    Just like you need to go to your mailbox for retrieving mails, the destination
    nodes also need to receive the messages and potentially update their features.

    >>> # Define a function for summing messages received and replacing the original feature.
    >>> def simple_reduce(nodes): return {'x': nodes.mailbox['m'].sum(1)}
    >>> # Set the function defined to be the default message reduce function.
    >>> G.register_reduce_func(simple_reduce)
    >>> # All existing edges have node 2 as the destination.
    >>> # Receive the messages for node 2 and update its feature.
    >>> G.recv(v=2)
    >>> G.ndata
    {'x': tensor([[1., 1., 1., 1., 1.],
                  [0., 0., 0., 0., 0.],
                  [3., 3., 3., 3., 3.]])} # 3 = (1 + 1) + (0 + 1)

    For more examples about message passing, please read our tutorials.
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

        Adding new nodes with features:

        .. note:: Here we use pytorch syntax for demo. The general idea applies
            to other frameworks with minor syntax change (e.g. replace
            ``torch.tensor`` with ``mxnet.ndarray``).

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

        .. note:: Here we use pytorch syntax for demo. The general idea applies
            to other frameworks with minor syntax change (e.g. replace
            ``torch.tensor`` with ``mxnet.ndarray``).

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

        .. note:: Here we use pytorch syntax for demo. The general idea applies
            to other frameworks with minor syntax change (e.g. replace
            ``torch.tensor`` with ``mxnet.ndarray``).

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
        """Return a 0-1 array ``a`` given the node ID array ``vids``.

        ``a[i]`` is 1 if the graph contains node ``vids[i]``, 0 otherwise.

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
        A tuple of Tensors ``(eu, ev, eid)`` if ``form == 'all'``.
            ``eid[i]`` is the ID of an inbound edge to ``ev[i]`` from ``eu[i]``.
            All inbound edges to ``v`` are returned.
        A pair of Tensors (eu, ev) if form == 'uv'
            ``eu[i]`` is the source node of an inbound edge to ``ev[i]``.
            All inbound edges to ``v`` are returned.
        One Tensor if form == 'eid'
            ``eid[i]`` is ID of an inbound edge to any of the nodes in ``v``.

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
        A tuple of Tensors ``(eu, ev, eid)`` if ``form == 'all'``.
            ``eid[i]`` is the ID of an outbound edge from ``eu[i]`` from ``ev[i]``.
            All outbound edges from ``v`` are returned.
        A pair of Tensors (eu, ev) if form == 'uv'
            ``ev[i]`` is the destination node of an outbound edge from ``eu[i]``.
            All outbound edges from ``v`` are returned.
        One Tensor if form == 'eid'
            ``eid[i]`` is ID of an outbound edge from any of the nodes in ``v``.

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
            ``eid[i]`` is the ID of an edge between ``u[i]`` and ``v[i]``.
            All edges are returned.
        A pair of Tensors (u, v) if form == 'uv'
            An edge exists between ``u[i]`` and ``v[i]``.
            If ``n`` edges exist between ``u`` and ``v``, then ``u`` and ``v`` as a pair
            will appear ``n`` times.
        One Tensor if form == 'eid'
            ``eid[i]`` is the ID of an edge in the graph.

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
        """Return the in-degree of node ``v``.

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

        Examples
        --------

        .. note:: Here we use pytorch syntax for demo. The general idea applies
            to other frameworks with minor syntax change (e.g. replace
            ``torch.tensor`` with ``mxnet.ndarray``).

        >>> import torch as th
        >>> g = DGLGraph()
        >>> g.add_nodes(5, {'n1': th.randn(5, 10)})
        >>> g.add_edges([0,1,3,4], [2,4,0,3], {'e1': th.randn(4, 6)})
        >>> nxg = g.to_networkx(node_attrs=['n1'], edge_attrs=['e1'])
        """
        nx_graph = self._graph.to_networkx()
        if node_attrs is not None:
            for nid, attr in nx_graph.nodes(data=True):
                nf = self.get_n_repr(nid)
                attr.update({key: nf[key].squeeze(0) for key in node_attrs})
        if edge_attrs is not None:
            for u, v, attr in nx_graph.edges(data=True):
                eid = attr['id']
                ef = self.get_e_repr(eid)
                attr.update({key: ef[key].squeeze(0) for key in edge_attrs})
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

        Examples
        --------

        .. note:: Here we use pytorch syntax for demo. The general idea applies
            to other frameworks with minor syntax change (e.g. replace
            ``torch.tensor`` with ``mxnet.ndarray``).

        >>> import torch as th
        >>> import networkx as nx
        >>> nxg = nx.DiGraph()
        >>> nxg.add_edge(0, 1, id=0, e1=5, e2=th.zeros(4))
        >>> nxg.add_edge(2, 3, id=2, e1=6, e2=th.ones(4))
        >>> nxg.add_edge(1, 2, id=1, e1=2, e2=th.full((4,), 2))
        >>> g = dgl.DGLGraph()
        >>> g.from_networkx(nxg, edge_attrs=['e1', 'e2'])
        >>> g.edata['e1']
        tensor([5, 2, 6])
        >>> g.edata['e2']
        tensor([[0., 0., 0., 0.],
                [2., 2., 2., 2.],
                [1., 1., 1., 1.]])
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
            attr_dict = defaultdict(list)
            for nid in range(self.number_of_nodes()):
                for attr in node_attrs:
                    attr_dict[attr].append(nx_graph.nodes[nid][attr])
            for attr in node_attrs:
                self._node_frame[attr] = _batcher(attr_dict[attr])
        if edge_attrs is not None:
            has_edge_id = 'id' in next(iter(nx_graph.edges(data=True)))[-1]
            attr_dict = defaultdict(lambda: [None] * self.number_of_edges())
            if has_edge_id:
                for _, _, attrs in nx_graph.edges(data=True):
                    for key in edge_attrs:
                        attr_dict[key][attrs['id']] = attrs[key]
            else:
                # XXX: assuming networkx iteration order is deterministic
                for eid, (_, _, attr) in enumerate(nx_graph.edges(data=True)):
                    for key in edge_attrs:
                        attr_dict[key][eid] = attrs[key]
            for attr in edge_attrs:
                self._edge_frame[attr] = _batcher(attr_dict[attr])

    def from_scipy_sparse_matrix(self, a):
        """ Convert from scipy sparse matrix.

        Parameters
        ----------
        a : scipy sparse matrix
            The graph's adjacency matrix

        Examples
        --------
        >>> from scipy.sparse import coo_matrix
        >>> row = np.array([0, 3, 1, 0])
        >>> col = np.array([0, 3, 1, 2])
        >>> data = np.array([4, 5, 7, 9])
        >>> a = coo_matrix((data, (row, col)), shape=(4, 4))
        >>> g = dgl.DGLGraph()
        >>> g.from_scipy_sparse_matrix(a)
        """
        self.clear()
        self._graph.from_scipy_sparse_matrix(a)
        self._node_frame.add_rows(self.number_of_nodes())
        self._edge_frame.add_rows(self.number_of_edges())
        self._msg_graph.add_nodes(self._graph.number_of_nodes())

    def node_attr_schemes(self):
        """Return the node feature schemes.

        Each feature scheme is a named tuple that stores the shape and data type
        of the node feature

        Returns
        -------
        dict of str to schemes
            The schemes of node feature columns.

        Examples
        --------
        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(3)
        >>> G.ndata['x'] = torch.zeros((3,5))
        >>> G.node_attr_schemes()
        {'x': Scheme(shape=(5,), dtype=torch.float32)}
        """
        return self._node_frame.schemes

    def edge_attr_schemes(self):
        """Return the edge feature schemes.

        Each feature scheme is a named tuple that stores the shape and data type
        of the node feature

        Returns
        -------
        dict of str to schemes
            The schemes of edge feature columns.

        Examples
        --------

        .. note:: Here we use pytorch syntax for demo. The general idea applies
            to other frameworks with minor syntax change (e.g. replace
            ``torch.tensor`` with ``mxnet.ndarray``).

        >>> import torch as th
        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(3)
        >>> G.add_edges([0, 1], 2)  # 0->2, 1->2
        >>> G.edata['y'] = th.zeros((2, 4))
        >>> G.edge_attr_schemes()
        {'y': Scheme(shape=(4,), dtype=torch.float32)}
        """
        return self._edge_frame.schemes

    def set_n_initializer(self, initializer, field=None):
        """Set the initializer for empty node features.

        Initializer is a callable that returns a tensor given the shape, data type
        and device context.

        When a subset of the nodes are assigned a new feature, initializer is
        used to create feature for rest of the nodes.

        Parameters
        ----------
        initializer : callable
            The initializer.
        field : str, optional
            The feature field name. Default is set an initializer for all the
            feature fields.

        Examples
        --------

        .. note:: Here we use pytorch syntax for demo. The general idea applies
            to other frameworks with minor syntax change (e.g. replace
            ``torch.tensor`` with ``mxnet.ndarray``).

        >>> import torch as th
        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(3)

        Set initializer for all node features

        >>> G.set_n_initializer(dgl.init.zero_initializer)

        Set feature for partial nodes

        >>> G.nodes[[0, 2]].data['x'] = th.ones((2, 5))
        >>> G.ndata
        {'x' : tensor([[1., 1., 1., 1., 1.],
                       [0., 0., 0., 0., 0.],
                       [1., 1., 1., 1., 1.]])}

        Note
        -----
        User defined initializer must follow the signature of
        :func:`dgl.init.base_initializer() <dgl.init.base_initializer>`

        """
        self._node_frame.set_initializer(initializer, field)

    def set_e_initializer(self, initializer, field=None):
        """Set the initializer for empty edge features.

        Initializer is a callable that returns a tensor given the shape, data
        type and device context.

        When a subset of the edges are assigned a new feature, initializer is
        used to create feature for rest of the edges.

        Parameters
        ----------
        initializer : callable
            The initializer.
        field : str, optional
            The feature field name. Default is set an initializer for all the
            feature fields.

        Examples
        --------

        .. note:: Here we use pytorch syntax for demo. The general idea applies
            to other frameworks with minor syntax change (e.g. replace
            ``torch.tensor`` with ``mxnet.ndarray``).

        >>> import torch as th
        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(3)
        >>> G.add_edges([0, 1], 2)  # 0->2, 1->2

        Set initializer for edge features

        >>> G.set_e_initializer(dgl.init.zero_initializer)

        Set feature for partial edges

        >>> G.edges[1, 2].data['y'] = th.ones((1, 4))
        >>> G.edata
        {'y' : tensor([[0., 0., 0., 0.],
                       [1., 1., 1., 1.]])}

        Note
        -----
        User defined initializer must follow the signature of
        :func:`dgl.init.base_initializer() <dgl.init.base_initializer>`
        """
        self._edge_frame.set_initializer(initializer, field)

    @property
    def nodes(self):
        """Return a node view that can used to set/get feature data.

        Examples
        --------

        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(3)

        Get nodes in graph G:

        >>> G.nodes()
        tensor([0, 1, 2])

        Get feature dictionary of all nodes:

        >>> G.nodes[:].data
        {}

        The above can be abbreviated as

        >>> G.ndata
        {}

        Init all 3 nodes with zero vector(len=5)

        .. note:: Here we use pytorch syntax for demo. The general idea applies
            to other frameworks with minor syntax change (e.g. replace
            ``torch.tensor`` with ``mxnet.ndarray``).

        >>> import torch as th
        >>> G.ndata['x'] = th.zeros((3, 5))
        >>> G.ndata['x']
        {'x' : tensor([[0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0.]])}

        Use G.nodes to get/set features for some nodes.

        >>> G.nodes[[0, 2]].data['x'] = th.ones((2, 5))
        >>> G.ndata
        {'x' : tensor([[1., 1., 1., 1., 1.],
                       [0., 0., 0., 0., 0.],
                       [1., 1., 1., 1., 1.]])}

        See Also
        --------
        dgl.DGLGraph.ndata

        """
        return NodeView(self)

    @property
    def ndata(self):
        """Return the data view of all the nodes.

        DGLGraph.ndata is an abbreviation of DGLGraph.nodes[:].data

        See Also
        --------
        dgl.DGLGraph.nodes
        """
        return self.nodes[:].data

    @property
    def edges(self):
        """Return a edges view that can used to set/get feature data.

        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(3)
        >>> G.add_edges([0, 1], 2)  # 0->2, 1->2

        Get edges in graph G:

        >>> G.edges()
        (tensor([0, 1]), tensor([2, 2]))

        Get feature dictionary of all edges:

        >>> G.edges[:].data
        {}

        The above can be abbreviated as

        >>> G.edata
        {}

        Init 2 edges with zero vector(len=4)

        .. note:: Here we use pytorch syntax for demo. The general idea applies
            to other frameworks with minor syntax change (e.g. replace
            ``torch.tensor`` with ``mxnet.ndarray``).

        >>> import torch as th
        >>> G.edata['y'] = th.zeros((2, 4))
        >>> G.edata
        {'y' : tensor([[0., 0., 0., 0.],
                       [0., 0., 0., 0.]])}

        Use G.edges to get/set features for some edges.

        >>> G.edges[1, 2].data['y'] = th.ones((1, 4))
        >>> G.edata
        {'y' : tensor([[0., 0., 0., 0.],
                       [1., 1., 1., 1.]])}

        See Also
        --------
        dgl.DGLGraph.edata
        """
        return EdgeView(self)

    @property
    def edata(self):
        """Return the data view of all the edges.

        DGLGraph.data is an abbreviation of DGLGraph.edges[:].data

        See Also
        --------
        dgl.DGLGraph.edges
        """
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

        Once registered, ``func`` will be used as the default
        message function in message passing operations, including
        :func:`send`, :func:`send_and_recv`, :func:`pull`,
        :func:`push`, :func:`update_all`.

        Parameters
        ----------
        func : callable
            Message function on the edge. The function should be
            an :mod:`Edge UDF <dgl.udf>`.

        See Also
        --------
        send
        send_and_recv
        pull
        push
        update_all
        """
        self._message_func = func

    def register_reduce_func(self, func):
        """Register global message reduce function.

        Once registered, ``func`` will be used as the default
        message reduce function in message passing operations, including
        :func:`recv`, :func:`send_and_recv`, :func:`push`, :func:`pull`,
        :func:`update_all`.

        Parameters
        ----------
        func : callable
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.

        See Also
        --------
        recv
        send_and_recv
        push
        pull
        update_all
        """
        self._reduce_func = func

    def register_apply_node_func(self, func):
        """Register global node apply function.

        Once registered, ``func`` will be used as the default apply
        node function. Related operations include :func:`apply_nodes`,
        :func:`recv`, :func:`send_and_recv`, :func:`push`, :func:`pull`,
        :func:`update_all`.

        Parameters
        ----------
        func : callable
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.

        See Also
        --------
        apply_nodes
        register_apply_edge_func
        """
        self._apply_node_func = func

    def register_apply_edge_func(self, func):
        """Register global edge apply function.

        Once registered, ``func`` will be used as the default apply
        edge function in :func:`apply_edges`.

        Parameters
        ----------
        func : callable
            Apply function on the edge. The function should be
            an :mod:`Edge UDF <dgl.udf>`.

        See Also
        --------
        apply_edges
        register_apply_node_func
        """
        self._apply_edge_func = func

    def apply_nodes(self, func="default", v=ALL, inplace=False):
        """Apply the function on the nodes to update their features.

        If None is provided for ``func``, nothing will happen.

        Parameters
        ----------
        func : callable or None, optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        v : int, iterable of int, tensor, optional
            The node (ids) on which to apply ``func``. The default
            value is all the nodes.
        inplace : bool, optional
            If True, update will be done in place, but autograd will break.

        Examples
        --------

        .. note:: Here we use pytorch syntax for demo. The general idea applies
            to other frameworks with minor syntax change (e.g. replace
            ``torch.tensor`` with ``mxnet.ndarray``).

        >>> import torch as th
        >>> g = dgl.DGLGraph()
        >>> g.add_nodes(3)
        >>> g.ndata['x'] = th.ones(3, 1)

        >>> # Increment the node feature by 1.
        >>> def increment_feature(nodes): return {'x': nodes.data['x'] + 1}
        >>> g.apply_nodes(func=increment_feature, v=[0, 2]) # Apply func to nodes 0, 2
        >>> g.ndata
        {'x': tensor([[2.],
                      [1.],
                      [2.]])}

        See Also
        --------
        register_apply_node_func
        apply_edges
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
        """Apply the function on the edges to update their features.

        If None is provided for ``func``, nothing will happen.

        Parameters
        ----------
        func : callable, optional
            Apply function on the edge. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        edges : valid edges type, optional
            Edges on which to apply ``func``. See :func:`send` for valid
            edges type. Default is all the edges.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.

        Notes
        -----
        On multigraphs, if :math:`u` and :math:`v` are specified, then all the edges
        between :math:`u` and :math:`v` will be updated.

        Examples
        --------

        .. note:: Here we use pytorch syntax for demo. The general idea applies
            to other frameworks with minor syntax change (e.g. replace
            ``torch.tensor`` with ``mxnet.ndarray``).

        >>> import torch as th

        >>> g = dgl.DGLGraph()
        >>> g.add_nodes(3)
        >>> g.add_edges([0, 1], [1, 2])   # 0 -> 1, 1 -> 2
        >>> g.edata['y'] = th.ones(2, 1)

        >>> # Doubles the edge feature.
        >>> def double_feature(edges): return {'y': edges.data['y'] * 2}
        >>> g.apply_edges(func=double_feature, edges=0) # Apply func to the first edge.
        >>> g.edata
        {'y': tensor([[2.],   # 2 * 1
                      [1.]])}

        See Also
        --------
        apply_nodes
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

    def send(self, edges=ALL, message_func="default"):
        """Send messages along the given edges.

        ``edges`` can be any of the following types:

        * ``int`` : Specify one edge using its edge id.
        * ``pair of int`` : Specify one edge using its endpoints.
        * ``int iterable`` / ``tensor`` : Specify multiple edges using their edge ids.
        * ``pair of int iterable`` / ``pair of tensors`` :
          Specify multiple edges using their endpoints.

        The UDF returns messages on the edges and can be later fetched in
        the destination node's ``mailbox``. Receiving will consume the messages.
        See :func:`recv` for example.

        If multiple ``send`` are triggered on the same edge without ``recv``. Messages
        generated by the later ``send`` will overwrite previous messages.

        Parameters
        ----------
        edges : valid edges type, optional
            Edges on which to apply ``message_func``. Default is sending along all
            the edges.
        message_func : callable
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.

        Notes
        -----
        On multigraphs, if :math:`u` and :math:`v` are specified, then the messages will be sent
        along all edges between :math:`u` and :math:`v`.

        Examples
        --------
        See the *message passing* example in :class:`DGLGraph` or :func:`recv`.
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
             v=ALL,
             reduce_func="default",
             apply_node_func="default",
             inplace=False):
        """Receive and reduce incoming messages and update the features of node(s) :math:`v`.

        Optionally, apply a function to update the node features after receive.

        * `reduce_func` will be skipped for nodes with no incoming message.
        * If all ``v`` have no incoming message, this will downgrade to an :func:`apply_nodes`.
        * If some ``v`` have no incoming message, their new feature value will be calculated
          by the column initializer (see :func:`set_n_initializer`). The feature shapes and
          dtypes will be inferred.

        The node features will be updated by the result of the ``reduce_func``.

        Messages are consumed once received.

        The provided UDF maybe called multiple times so it is recommended to provide
        function with no side effect.

        Parameters
        ----------
        v : node, container or tensor, optional
            The node to be updated. Default is receiving all the nodes.
        reduce_func : callable, optional
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        apply_node_func : callable
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.

        Examples
        --------
        Create a graph object for demo.

        .. note:: Here we use pytorch syntax for demo. The general idea applies
            to other frameworks with minor syntax change (e.g. replace
            ``torch.tensor`` with ``mxnet.ndarray``).

        >>> import torch as th
        >>> g = dgl.DGLGraph()
        >>> g.add_nodes(3)
        >>> g.ndata['x'] = th.tensor([[1.], [2.], [3.]])
        >>> g.add_edges([0, 1], [1, 2])

        >>> # Define the function for sending node features as messages.
        >>> def send_source(edges): return {'m': edges.src['x']}
        >>> # Set the function defined to be the default message function.
        >>> g.register_message_func(send_source)

        >>> # Sum the messages received and use this to replace the original node feature.
        >>> def simple_reduce(nodes): return {'x': nodes.mailbox['m'].sum(1)}
        >>> # Set the function defined to be the default message reduce function.
        >>> g.register_reduce_func(simple_reduce)

        Send and receive messages. Note that although node :math:`0` has no incoming edges,
        its feature gets changed from :math:`1` to :math:`0` as it is also included in
        ``g.nodes()``.

        >>> g.send(g.edges())
        >>> g.recv(g.nodes())
        >>> g.ndata['x']
        tensor([[0.],
                [1.],
                [2.]])

        Once messages are received, one will need another call of :func:`send` again before
        another call of :func:`recv`. Otherwise, nothing will happen.

        >>> g.recv(g.nodes())
        >>> g.ndata['x']
        tensor([[0.],
                [1.],
                [2.]])
        """
        if reduce_func == "default":
            reduce_func = self._reduce_func
        if apply_node_func == "default":
            apply_node_func = self._apply_node_func
        assert reduce_func is not None

        if self._msg_frame.num_rows == 0:
            # no message has ever been sent
            return

        if is_all(v):
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
        """Send messages along edges and let destinations receive them.

        Optionally, apply a function to update the node features after receive.

        This is a convenient combination for performing
        ``send(self, self.edges, message_func)`` and
        ``recv(self, dst, reduce_func, apply_node_func)``, where ``dst``
        are the destinations of the ``edges``.

        Parameters
        ----------
        edges : valid edges type
            Edges on which to apply ``func``. See :func:`send` for valid
            edges type.
        message_func : callable, optional
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        reduce_func : callable, optional
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        apply_node_func : callable, optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.

        Examples
        --------

        .. note:: Here we use pytorch syntax for demo. The general idea applies
            to other frameworks with minor syntax change (e.g. replace
            ``torch.tensor`` with ``mxnet.ndarray``).

        >>> import torch as th
        >>> g = dgl.DGLGraph()
        >>> g.add_nodes(3)
        >>> g.ndata['x'] = th.tensor([[1.], [2.], [3.]])
        >>> g.add_edges([0, 1], [1, 2])

        >>> # Define the function for sending node features as messages.
        >>> def send_source(edges): return {'m': edges.src['x']}
        >>> # Set the function defined to be the default message function.
        >>> g.register_message_func(send_source)

        >>> # Sum the messages received and use this to replace the original node feature.
        >>> def simple_reduce(nodes): return {'x': nodes.mailbox['m'].sum(1)}
        >>> # Set the function defined to be the default message reduce function.
        >>> g.register_reduce_func(simple_reduce)

        Send and receive messages.

        >>> g.send_and_recv(g.edges())
        >>> g.ndata['x']
        tensor([[1.],
                [1.],
                [2.]])

        Note that the feature of node :math:`0` remains the same as it has no
        incoming edges.

        Notes
        -----
        On multigraphs, if u and v are specified, then the messages will be sent
        and received along all edges between u and v.

        See Also
        --------
        send
        recv
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
        """Pull messages from the node(s)' predecessors and then update their features.

        Optionally, apply a function to update the node features after receive.

        * `reduce_func` will be skipped for nodes with no incoming message.
        * If all ``v`` have no incoming message, this will downgrade to an :func:`apply_nodes`.
        * If some ``v`` have no incoming message, their new feature value will be calculated
          by the column initializer (see :func:`set_n_initializer`). The feature shapes and
          dtypes will be inferred.

        Parameters
        ----------
        v : int, iterable of int, or tensor
            The node(s) to be updated.
        message_func : callable, optional
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        reduce_func : callable, optional
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        apply_node_func : callable, optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.

        Examples
        --------
        Create a graph for demo.

        .. note:: Here we use pytorch syntax for demo. The general idea applies
            to other frameworks with minor syntax change (e.g. replace
            ``torch.tensor`` with ``mxnet.ndarray``).

        >>> import torch as th
        >>> g = dgl.DGLGraph()
        >>> g.add_nodes(3)
        >>> g.ndata['x'] = th.tensor([[0.], [1.], [2.]])

        Use the built-in message function :func:`~dgl.function.copy_src` for copying
        node features as the message.

        >>> m_func = dgl.function.copy_src('x', 'm')
        >>> g.register_message_func(m_func)

        Use the built-int message reducing function :func:`~dgl.function.sum`, which
        sums the messages received and replace the old node features with it.

        >>> m_reduce_func = dgl.function.sum('m', 'x')
        >>> g.register_reduce_func(m_reduce_func)

        As no edges exist, nothing happens.

        >>> g.pull(g.nodes())
        >>> g.ndata['x']
        tensor([[0.],
                [1.],
                [2.]])

        Add edges ``0 -> 1, 1 -> 2``. Pull messages for the node :math:`2`.

        >>> g.add_edges([0, 1], [1, 2])
        >>> g.pull(2)
        >>> g.ndata['x']
        tensor([[0.],
                [1.],
                [1.]])

        The feature of node :math:`2` changes but the feature of node :math:`1`
        remains the same as we did not :func:`pull` (and reduce) messages for it.

        See Also
        --------
        push
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
        """Send message from the node(s) to their successors and update them.

        Optionally, apply a function to update the node features after receive.

        Parameters
        ----------
        u : int, iterable of int, or tensor
            The node(s) to push messages out.
        message_func : callable, optional
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        reduce_func : callable, optional
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        apply_node_func : callable, optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.

        Examples
        --------
        Create a graph for demo.

        .. note:: Here we use pytorch syntax for demo. The general idea applies
            to other frameworks with minor syntax change (e.g. replace
            ``torch.tensor`` with ``mxnet.ndarray``).

        >>> import torch as th
        >>> g = dgl.DGLGraph()
        >>> g.add_nodes(3)
        >>> g.ndata['x'] = th.tensor([[1.], [2.], [3.]])

        Use the built-in message function :func:`~dgl.function.copy_src` for copying
        node features as the message.

        >>> m_func = dgl.function.copy_src('x', 'm')
        >>> g.register_message_func(m_func)

        Use the built-int message reducing function :func:`~dgl.function.sum`, which
        sums the messages received and replace the old node features with it.

        >>> m_reduce_func = dgl.function.sum('m', 'x')
        >>> g.register_reduce_func(m_reduce_func)

        As no edges exist, nothing happens.

        >>> g.push(g.nodes())
        >>> g.ndata['x']
        tensor([[1.],
                [2.],
                [3.]])

        Add edges ``0 -> 1, 1 -> 2``. Send messages from the node :math:`1`. and update.

        >>> g.add_edges([0, 1], [1, 2])
        >>> g.push(1)
        >>> g.ndata['x']
        tensor([[1.],
                [2.],
                [2.]])

        The feature of node :math:`2` changes but the feature of node :math:`1`
        remains the same as we did not :func:`push` for node :math:`0`.

        See Also
        --------
        pull
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
        """Send messages through all edges and update all nodes.

        Optionally, apply a function to update the node features after receive.

        This is a convenient combination for performing
        ``send(self, self.edges(), message_func)`` and
        ``recv(self, self.nodes(), reduce_func, apply_node_func)``.

        Parameters
        ----------
        message_func : callable, optional
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        reduce_func : callable, optional
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        apply_node_func : callable, optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.

        See Also
        --------
        send
        recv
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
        """Propagate messages using graph traversal by triggering
        :func:`pull()` on nodes.

        The traversal order is specified by the ``nodes_generator``. It generates
        node frontiers, which is a list or a tensor of nodes. The nodes in the
        same frontier will be triggered together, while nodes in different frontiers
        will be triggered according to the generating order.

        Parameters
        ----------
        node_generators : iterable, each element is a list or a tensor of node ids
            The generator of node frontiers. It specifies which nodes perform
            :func:`pull` at each timestep.
        message_func : callable, optional
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        reduce_func : callable, optional
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        apply_node_func : callable, optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.

        Examples
        --------
        Create a graph for demo.

        .. note:: Here we use pytorch syntax for demo. The general idea applies
            to other frameworks with minor syntax change (e.g. replace
            ``torch.tensor`` with ``mxnet.ndarray``).

        >>> import torch as th
        >>> g = dgl.DGLGraph()
        >>> g.add_nodes(4)
        >>> g.ndata['x'] = th.tensor([[1.], [2.], [3.], [4.]])
        >>> g.add_edges([0, 1, 1, 2], [1, 2, 3, 3])

        Prepare message function and message reduce function for demo.

        >>> def send_source(edges): return {'m': edges.src['x']}
        >>> g.register_message_func(send_source)
        >>> def simple_reduce(nodes): return {'x': nodes.mailbox['m'].sum(1)}
        >>> g.register_reduce_func(simple_reduce)

        First pull messages for nodes :math:`1, 2` with edges ``0 -> 1`` and
        ``1 -> 2``; and then pull messages for node :math:`3` with edges
        ``1 -> 3`` and ``2 -> 3``.

        >>> g.prop_nodes([[1, 2], [3]])
        >>> g.ndata['x']
        tensor([[1.],
                [1.],
                [2.],
                [3.]])

        In the first stage, we pull messages for nodes :math:`1, 2`.
        The feature of node :math:`1` is replaced by that of node :math:`0`, i.e. 1
        The feature of node :math:`2` is replaced by that of node :math:`1`, i.e. 2.
        Both of the replacement happen simultaneously.

        In the second stage, we pull messages for node :math:`3`.
        The feature of node :math:`3` becomes the sum of node :math:`1`'s feature and
        :math:`2`'s feature, i.e. 1 + 2 = 3.

        See Also
        --------
        prop_edges
        """
        for node_frontier in nodes_generator:
            self.pull(node_frontier,
                    message_func, reduce_func, apply_node_func)

    def prop_edges(self,
                   edges_generator,
                   message_func="default",
                   reduce_func="default",
                   apply_node_func="default"):
        """Propagate messages using graph traversal by triggering
        :func:`send_and_recv()` on edges.

        The traversal order is specified by the ``edges_generator``. It generates
        edge frontiers. The edge frontiers should be of *valid edges type*.
        See :func:`send` for more details.

        Edges in the same frontier will be triggered together, while edges in
        different frontiers will be triggered according to the generating order.

        Parameters
        ----------
        edges_generator : generator
            The generator of edge frontiers.
        message_func : callable, optional
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        reduce_func : callable, optional
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        apply_node_func : callable, optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.

        Examples
        --------
        Create a graph for demo.

        .. note:: Here we use pytorch syntax for demo. The general idea applies
            to other frameworks with minor syntax change (e.g. replace
            ``torch.tensor`` with ``mxnet.ndarray``).

        >>> import torch as th
        >>> g = dgl.DGLGraph()
        >>> g.add_nodes(4)
        >>> g.ndata['x'] = th.tensor([[1.], [2.], [3.], [4.]])
        >>> g.add_edges([0, 1, 1, 2], [1, 2, 3, 3])

        Prepare message function and message reduce function for demo.

        >>> def send_source(edges): return {'m': edges.src['x']}
        >>> g.register_message_func(send_source)
        >>> def simple_reduce(nodes): return {'x': nodes.mailbox['m'].sum(1)}
        >>> g.register_reduce_func(simple_reduce)

        First propagate messages for edges ``0 -> 1``, ``1 -> 3`` and then
        propagate messages for edges ``1 -> 2``, ``2 -> 3``.

        >>> g.prop_edges([([0, 1], [1, 3]), ([1, 2], [2, 3])])
        >>> g.ndata['x']
        tensor([[1.],
                [1.],
                [1.],
                [3.]])

        In the first stage, the following happens simultaneously.

            - The feature of node :math:`1` is replaced by that of
              node :math:`0`, i.e. 1.
            - The feature of node :math:`3` is replaced by that of
              node :math:`1`, i.e. 2.

        In the second stage, the following happens simultaneously.

            - The feature of node :math:`2` is replaced by that of
              node :math:`1`, i.e. 1.
            - The feature of node :math:`3` is replaced by that of
              node :math:`2`, i.e. 3.

        See Also
        --------
        prop_nodes
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
        ``[self.subgraph(nodes_list) for nodes_list in nodes]``

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

        By default, a row of returned adjacency matrix represents the
        destination of an edge and the column represents the source.

        When transpose is True, a row represents the source and a column
        represents a destination.

        Parameters
        ----------
        transpose : bool, optional (default=False)
            A flag to transpose the returned adjacency matrix.
        ctx : context, optional (default=cpu)
            The context of returned adjacency matrix.

        Returns
        -------
        SparseTensor
            The adjacency matrix.
        """
        return self._graph.adjacency_matrix(transpose, ctx)[0]

    def incidence_matrix(self, type, ctx=F.cpu()):
        """Return the incidence matrix representation of this graph.

        An incidence matrix is an n x m sparse matrix, where n is
        the number of nodes and m is the number of edges. Each nnz
        value indicating whether the edge is incident to the node
        or not.

        There are three types of an incidence matrix :math:`I`:

        * ``in``:

            - :math:`I[v, e] = 1` if :math:`e` is the in-edge of :math:`v`
              (or :math:`v` is the dst node of :math:`e`);
            - :math:`I[v, e] = 0` otherwise.

        * ``out``:

            - :math:`I[v, e] = 1` if :math:`e` is the out-edge of :math:`v`
              (or :math:`v` is the src node of :math:`e`);
            - :math:`I[v, e] = 0` otherwise.

        * ``both``:

            - :math:`I[v, e] = 1` if :math:`e` is the in-edge of :math:`v`;
            - :math:`I[v, e] = -1` if :math:`e` is the out-edge of :math:`v`;
            - :math:`I[v, e] = 0` otherwise (including self-loop).

        Parameters
        ----------
        type : str
            Can be either ``in``, ``out`` or ``both``
        ctx : context, optional (default=cpu)
            The context of returned incidence matrix.

        Returns
        -------
        SparseTensor
            The incidence matrix.
        """
        return self._graph.incidence_matrix(type, ctx)[0]

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
            A function of signature ``func(nodes) -> tensor``.
            ``nodes`` are :class:`NodeBatch` objects as in :mod:`~dgl.udf`.
            The ``tensor`` returned should be a 1-D boolean tensor with
            each element indicating whether the corresponding node in
            the batch satisfies the predicate.
        nodes : int, iterable or tensor of ints
            The nodes to filter on. Default value is all the nodes.

        Returns
        -------
        tensor
            The filtered nodes.

        Examples
        --------
        Construct a graph object for demo.

        .. note:: Here we use pytorch syntax for demo. The general idea applies
            to other frameworks with minor syntax change (e.g. replace
            ``torch.tensor`` with ``mxnet.ndarray``).

        >>> import torch as th
        >>> g = dgl.DGLGraph()
        >>> g.add_nodes(3)
        >>> g.ndata['x'] = th.tensor([[1.], [-1.], [1.]])

        Define a function for filtering nodes with feature :math:`1`.

        >>> def has_feature_one(nodes): return (nodes.data['x'] == 1).squeeze(1)

        Filter the nodes with feature :math:`1`.

        >>> g.filter_nodes(has_feature_one)
        tensor([0, 2])

        See Also
        --------
        filter_edges
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
            A function of signature ``func(edges) -> tensor``.
            ``edges`` are :class:`EdgeBatch` objects as in :mod:`~dgl.udf`.
            The ``tensor`` returned should be a 1-D boolean tensor with
            each element indicating whether the corresponding edge in
            the batch satisfies the predicate.
        edges : valid edges type
            Edges on which to apply ``func``. See :func:`send` for valid
            edges type. Default value is all the edges.

        Returns
        -------
        tensor
            The filtered edges represented by their ids.

        Examples
        --------
        Construct a graph object for demo.

        .. note:: Here we use pytorch syntax for demo. The general idea applies
            to other frameworks with minor syntax change (e.g. replace
            ``torch.tensor`` with ``mxnet.ndarray``).

        >>> import torch as th
        >>> g = dgl.DGLGraph()
        >>> g.add_nodes(3)
        >>> g.ndata['x'] = th.tensor([[1.], [-1.], [1.]])
        >>> g.add_edges([0, 1, 2], [2, 2, 1])

        Define a function for filtering edges whose destinations have
        node feature :math:`1`.

        >>> def has_dst_one(edges): return (edges.dst['x'] == 1).squeeze(1)

        Filter the edges whose destination nodes have feature :math:`1`.

        >>> g.filter_edges(has_dst_one)
        tensor([0, 1])

        See Also
        --------
        filter_nodes
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
