"""Base graph class specialized for neural networks on graphs."""
# pylint: disable=too-many-lines
from __future__ import absolute_import

from collections import defaultdict
from contextlib import contextmanager
from typing import Iterable
from functools import wraps
import networkx as nx

import dgl
from ..base import ALL, NID, EID, is_all, DGLError, dgl_warning
from .. import backend as F
from .. import init
from .frame import FrameRef, Frame, Scheme, sync_frame_initializer
from .. import graph_index
from .runtime import ir, scheduler, Runtime, GraphAdapter
from .. import utils
from .view import NodeView, EdgeView
from .udf import NodeBatch, EdgeBatch

__all__ = ['DGLGraph', 'batch', 'unbatch']

class DGLBaseGraph(object):
    """Base graph class.

    DGL graph is always directional. Undirected graph can be represented using
    two bi-directional edges.

    Nodes are identified by consecutive integers starting from zero.

    Edges can be specified by two end points (u, v) or the integer id assigned
    when the edges are added.  Edge IDs are automatically assigned by the order
    of addition, i.e. the first edge being added has an ID of 0, the second
    being 1, so on so forth.

    Parameters
    ----------
    graph : graph index, optional
        Data to initialize graph.
    """

    is_block = False        # for compatibility with DGLHeteroGraph

    def __init__(self, graph):
        self._graph = graph

    def number_of_nodes(self):
        """Return the number of nodes in the graph.

        Returns
        -------
        int
            The number of nodes
        """
        return self._graph.number_of_nodes()

    def number_of_src_nodes(self):
        """Return the number of nodes in the graph.

        For compatibility with heterographs.

        Returns
        -------
        int
            The number of nodes
        """
        return self._graph.number_of_nodes()

    def number_of_dst_nodes(self):
        """Return the number of nodes in the graph.

        For compatibility with heterographs.

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

    @property
    def is_readonly(self):
        """True if the graph is readonly, False otherwise.
        """
        return self._graph.is_readonly()

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

    def edge_id(self, u, v, force_multi=None, return_array=False):
        """Return the edge ID, or an array of edge IDs, between source node
        `u` and destination node `v`.

        Parameters
        ----------
        u : int
            The source node ID.
        v : int
            The destination node ID.
        force_multi : bool
            Deprecated (Will be deleted in the future).
            If False, will return a single edge ID.
            If True, will always return an array.
        return_array : bool
            If False, will return a single edge ID.
            If True, will always return an array.

        Returns
        -------
        int or tensor
            The edge ID if return_array is False.
            The edge ID array otherwise.

        Notes
        -----
        If multiply edges exist between `u` and `v` and return_array is False,
        the result is undefined.

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

        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(3)

        Adding edges (0, 1), (0, 2), (0, 1), (0, 2), so edge ID 0 and 2 both
        connect from 0 and 1, while edge ID 1 and 3 both connect from 0 and 2.

        >>> G.add_edges([0, 0, 0, 0], [1, 2, 1, 2])
        >>> G.edge_id(0, 1, return_array=True)
        tensor([0, 2])

        See Also
        --------
        edge_ids
        """
        idx = self._graph.edge_id(u, v)
        if force_multi is not None:
            dgl_warning("force_multi will be deprecated." \
                        "Please use return_array instead")
            return_array = force_multi

        if return_array:
            return idx.tousertensor()
        else:
            assert len(idx) == 1, "For return_array=False, there should be one and " \
                "only one edge between u and v, but get {} edges. " \
                "Please use return_array=True instead".format(len(idx))
            return idx[0]

    def edge_ids(self, u, v, force_multi=None, return_uv=False):
        """Return all edge IDs between source node array `u` and destination
        node array `v`.

        Parameters
        ----------
        u : list, tensor
            The source node ID array.
        v : list, tensor
            The destination node ID array.
        force_multi : bool
            Deprecated (Will be deleted in the future).
            Whether to always treat the graph as a multigraph.
        return_uv : bool
            Whether return e or (eu, ev, e)

        Returns
        -------
        tensor, or (tensor, tensor, tensor)
            If 'return_uv` is False, return a single edge ID array `e`.
            `e[i]` is the edge ID between `u[i]` and `v[i]`.
            Otherwise, return three arrays `(eu, ev, e)`.  `e[i]` is the ID
            of an edge between `eu[i]` and `ev[i]`.  All edges between `u[i]`
            and `v[i]` are returned.

        Notes
        -----
        If the graph is a simple graph, `return_uv` is False, and no edge
        exist between some pairs of `u[i]` and `v[i]`, the result is undefined.

        If the graph is a multi graph, `return_uv` is False, and multi edges
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

        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(4)
        >>> G.add_edges([0, 0, 0], [1, 1, 2])   # (0, 1), (0, 1), (0, 2)

        Get all edges between (0, 1), (0, 2), (0, 3).  Note that there is no
        edge between 0 and 3:

        >>> G.edge_ids([0, 0, 0], [1, 2, 3], return_uv=True)
        (tensor([0, 0, 0]), tensor([1, 1, 2]), tensor([0, 1, 2]))

        See Also
        --------
        edge_id
        """
        u = utils.toindex(u)
        v = utils.toindex(v)
        src, dst, eid = self._graph.edge_ids(u, v)
        if force_multi is not None:
            dgl_warning("force_multi will be deprecated, " \
                        "Please use return_uv instead")
            return_uv = force_multi

        if return_uv:
            return src.tousertensor(), dst.tousertensor(), eid.tousertensor()
        else:
            assert len(eid) == max(len(u), len(v)), "If return_uv=False, there should be one and " \
                "only one edge between each u and v, expect {} edges but get {}. " \
                "Please use return_uv=True instead".format(max(len(u), len(v)), len(eid))
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
            ``eid[i]`` is the ID of an outbound edge from ``eu[i]`` to ``ev[i]``.
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

    def all_edges(self, form='uv', order=None):
        """Return all the edges.

        Parameters
        ----------
        form : str, optional
            The return form. Currently support:

            - 'all' : a tuple (u, v, eid)
            - 'uv'  : a pair (u, v), default
            - 'eid' : one eid tensor
        order : string
            The order of the returned edges. Currently support:

            - 'srcdst' : sorted by their src and dst ids.
            - 'eid'    : sorted by edge Ids.
            - None     : the arbitrary order.

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
        src, dst, eid = self._graph.edges(order)
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
            v = utils.toindex(slice(0, self.number_of_nodes()))
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
            v = utils.toindex(slice(0, self.number_of_nodes()))
        else:
            v = utils.toindex(v)
        return self._graph.out_degrees(v).tousertensor()

    @property
    def idtype(self):
        """Return the dtype of the graph index

        Returns
        ---------
        backend dtype object
            th.int32/th.int64 or tf.int32/tf.int64 etc.
        """
        return getattr(F, self._graph.dtype)


    @property
    def _idtype_str(self):
        """The dtype of graph index

        Returns
        -------
        backend dtype object
            th.int32/th.int64 or tf.int32/tf.int64 etc.
        """
        return self._graph.dtype


def mutation(func):
    """A decorator to decorate functions that might change graph structure."""
    @wraps(func)
    def inner(g, *args, **kwargs):
        if g.is_readonly:
            raise DGLError("Readonly graph. Mutation is not allowed. "
                           "To mutate it, call g.readonly(False) first.")
        if g.batch_size > 1:
            dgl_warning("The graph has batch_size > 1, and mutation would break"
                        " batching related properties, call `flatten` to remove"
                        " batching information of the graph.")
        if g._parent is not None:
            dgl_warning("The graph is a subgraph of a parent graph, and mutation"
                        " would break subgraph related properties, call `detach"
                        "_parent` to remove its connection with its parent.")
        func(g, *args, **kwargs)
    return inner


class DGLGraph(DGLBaseGraph):
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

    DGL graph accepts graph data of multiple formats:

    * NetworkX graph,
    * scipy matrix,
    * DGLGraph.

    If the input graph data is DGLGraph, the constructed DGLGraph only contains
    its graph index.

    Parameters
    ----------
    graph_data : graph data, optional
        Data to initialize graph.
    node_frame : FrameRef, optional
        Node feature storage.
    edge_frame : FrameRef, optional
        Edge feature storage.
    multigraph : bool, optional
        Deprecated (Will be deleted in the future).
        Whether the graph would be a multigraph. If none, the flag will be
        set to True. (default: None)
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
                 multigraph=None,
                 readonly=False,
                 sort_csr=False,
                 batch_num_nodes=None,
                 batch_num_edges=None,
                 parent=None):
        # graph
        if isinstance(graph_data, DGLGraph):
            gidx = graph_data._graph
            if sort_csr:
                gidx.sort_csr()
        else:
            if multigraph is not None:
                dgl_warning("multigraph will be deprecated." \
                            "DGL will treat all graphs as multigraph in the future.")
            gidx = graph_index.create_graph_index(graph_data, readonly)
            if sort_csr:
                gidx.sort_csr()
        super(DGLGraph, self).__init__(gidx)

        # node and edge frame
        if node_frame is None:
            self._node_frame = FrameRef(Frame(num_rows=self.number_of_nodes()))
        else:
            self._node_frame = node_frame
        if edge_frame is None:
            self._edge_frame = FrameRef(Frame(num_rows=self.number_of_edges()))
        else:
            self._edge_frame = edge_frame
        # message indicator:
        # if self._msg_index[eid] == 1, then edge eid has message
        self._msg_index = None
        # message frame
        self._msg_frame = FrameRef(Frame(num_rows=self.number_of_edges()))
        # set initializer for message frame
        self._msg_frame.set_initializer(init.zero_initializer)
        # registered functions
        self._message_func = None
        self._reduce_func = None
        self._apply_node_func = None
        self._apply_edge_func = None

        # batched graph
        self._batch_num_nodes = batch_num_nodes
        self._batch_num_edges = batch_num_edges

        # set parent if the graph is a subgraph.
        self._parent = parent

    def __setstate__(self, state):
        # Compatibility with pickles from DGL 0.4.2-
        if '_batch_num_nodes' not in state:
            state = state.copy()
            state.setdefault('_batch_num_nodes', None)
            state.setdefault('_batch_num_edges', None)
            state.setdefault('_parent', None)
        self.__dict__.update(state)

    def _create_subgraph(self, sgi, induced_nodes, induced_edges):
        """Internal function to create a subgraph from index."""
        subg = DGLGraph(graph_data=sgi.graph,
                        readonly=True,
                        parent=self)
        subg.ndata[NID] = induced_nodes.tousertensor()
        subg.edata[EID] = induced_edges.tousertensor()
        return subg

    def _get_msg_index(self):
        if self._msg_index is None:
            self._msg_index = utils.zero_index(size=self.number_of_edges())
        return self._msg_index

    def _set_msg_index(self, index):
        self._msg_index = index

    @mutation
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
        if data is None:
            # Initialize feature placeholders if there are features existing
            self._node_frame.add_rows(num)
        else:
            self._node_frame.append(data)

    @mutation
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
        # resize msg_index and msg_frame
        if self._msg_index is not None:
            self._msg_index = self._msg_index.append_zeros(1)
        self._msg_frame.add_rows(1)

    @mutation
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
        num = max(len(u), len(v))
        if data is None:
            # Initialize feature placeholders if there are features existing
            # NOTE: use max due to edge broadcasting syntax
            self._edge_frame.add_rows(num)
        else:
            self._edge_frame.append(data)
        # initialize feature placeholder for messages
        if self._msg_index is not None:
            self._msg_index = self._msg_index.append_zeros(num)
        self._msg_frame.add_rows(num)

    @mutation
    def remove_nodes(self, vids):
        """Remove multiple nodes, edges that have connection with these nodes would also be removed.

        Parameters
        ----------
        vids: list, tensor
            The id of nodes to remove.

        Notes
        -----
        The nodes and edges in the graph would be re-indexed after the removal.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import torch as th
        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(5, {'x': th.arange(5) * 2})
        >>> G.add_edges([0, 1, 2, 3, 4], [1, 2, 3, 4, 0], {'x': th.arange(15).view(5, 3)})
        >>> G.nodes()
        tensor([0, 1, 2, 3, 4])
        >>> G.edges()
        (tensor([0, 1, 2, 3, 4]), tensor([1, 2, 3, 4, 0]))
        >>> G.ndata['x']
        tensor([0, 2, 4, 6, 8])
        >>> G.edata['x']
        tensor([[ 0,  1,  2],
                [ 3,  4,  5],
                [ 6,  7,  8],
                [ 9, 10, 11],
                [12, 13, 14]])
        >>> G.remove_nodes([2, 3])
        >>> G.nodes()
        tensor([0, 1, 2]
        >>> G.edges()
        (tensor([0, 2]), tensor([1, 0]))
        >>> G.ndata['x']
        tensor([0, 2, 8])
        >>> G.edata['x']
        tensor([[ 0,  1,  2],
                [12, 13, 14]])

        See Also
        --------
        add_nodes
        add_edges
        remove_edges
        """
        induced_nodes = utils.set_diff(utils.toindex(self.nodes()), utils.toindex(vids))
        sgi = self._graph.node_subgraph(induced_nodes)

        num_nodes = len(sgi.induced_nodes)
        num_edges = len(sgi.induced_edges)
        if isinstance(self._node_frame, FrameRef):
            self._node_frame = FrameRef(Frame(self._node_frame[sgi.induced_nodes],
                                              num_rows=num_nodes))
        else:
            self._node_frame = FrameRef(self._node_frame, sgi.induced_nodes)

        if isinstance(self._edge_frame, FrameRef):
            self._edge_frame = FrameRef(Frame(self._edge_frame[sgi.induced_edges],
                                              num_rows=num_edges))
        else:
            self._edge_frame = FrameRef(self._edge_frame, sgi.induced_edges)

        self._graph = sgi.graph

    @mutation
    def remove_edges(self, eids):
        """Remove multiple edges.

        Parameters
        ----------
        eids: list, tensor
            The id of edges to remove.

        Notes
        -----
        The edges in the graph would be re-indexed after the removal.  The nodes are preserved.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import torch as th
        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(5)
        >>> G.add_edges([0, 1, 2, 3, 4], [1, 2, 3, 4, 0], {'x': th.arange(15).view(5, 3)})
        >>> G.nodes()
        tensor([0, 1, 2, 3, 4])
        >>> G.edges()
        (tensor([0, 1, 2, 3, 4]), tensor([1, 2, 3, 4, 0]))
        >>> G.edata['x']
        tensor([[ 0,  1,  2],
                [ 3,  4,  5],
                [ 6,  7,  8],
                [ 9, 10, 11],
                [12, 13, 14]])
        >>> G.remove_edges([1, 2])
        >>> G.nodes()
        tensor([0, 1, 2, 3, 4])
        >>> G.edges()
        (tensor([0, 3, 4]), tensor([1, 4, 0]))
        >>> G.edata['x']
        tensor([[ 0,  1,  2],
                [ 9, 10, 11],
                [12, 13, 14]])

        See Also
        --------
        add_nodes
        add_edges
        remove_nodes
        """
        induced_edges = utils.set_diff(
            utils.toindex(range(self.number_of_edges())), utils.toindex(eids))
        sgi = self._graph.edge_subgraph(induced_edges, preserve_nodes=True)

        num_nodes = len(sgi.induced_nodes)
        num_edges = len(sgi.induced_edges)
        if isinstance(self._node_frame, FrameRef):
            self._node_frame = FrameRef(Frame(self._node_frame[sgi.induced_nodes],
                                              num_rows=num_nodes))
        else:
            self._node_frame = FrameRef(self._node_frame, sgi.induced_nodes)

        if isinstance(self._edge_frame, FrameRef):
            self._edge_frame = FrameRef(Frame(self._edge_frame[sgi.induced_edges],
                                              num_rows=num_edges))
        else:
            self._edge_frame = FrameRef(self._edge_frame, sgi.induced_edges)

        self._graph = sgi.graph

    @property
    def parent_nid(self):
        """Get the parent node ids.

        The returned tensor can be used as a map from the node id
        in this subgraph to the node id in the parent graph.

        Returns
        -------
        Tensor
            The parent node id array.

        Notes
        -----
        The parent node id information is stored in ``_ID`` field in the
        node frame of the graph, so please do not manually change
        this field.
        """
        if self._parent is None:
            raise DGLError("We only support parent_nid for subgraphs.")
        return self.ndata[NID]

    @property
    def parent_eid(self):
        """Get the parent edge ids.

        The returned tensor can be used as a map from the edge id
        in this subgraph to the edge id in the parent graph.

        Returns
        -------
        Tensor
            The parent edge id array.

        Notes
        -----
        The parent edge id information is stored in ``_ID`` field in the
        edge frame of the graph, so please do not manually change
        this field.
        """
        if self._parent is None:
            raise DGLError("We only support parent_eid for subgraphs.")
        return self.edata[EID]

    def copy_to_parent(self, inplace=False):
        """Write node/edge features to the parent graph.

        Parameters
        ----------
        inplace : bool
            If true, use inplace write (no gradient but faster)

        Examples
        --------
        >>> import dgl
        >>> import torch as th
        >>> g = dgl.DGLGraph()
        >>> g.add_nodes(5)                  # Create a DGLGraph with 5 nodes
        >>> g.add_edges([0,1,2,3,4], [1,2,3,4,0])
        >>> subg.ndata['h'] = th.rand(4, 3)
        >>> subg.edata['h'] = th.rand(3, 3)
        >>> subg.ndata
        {'_ID': tensor([0, 1, 3, 4]), 'h': tensor([[0.3803, 0.9351, 0.0611],
                [0.6492, 0.4327, 0.3610],
                [0.7471, 0.4257, 0.4130],
                [0.9766, 0.6280, 0.6075]])}
        >>> subg.edata
        {'_ID': tensor([0, 3, 4]), 'h': tensor([[0.8192, 0.2409, 0.6278],
                [0.9600, 0.3501, 0.8037],
                [0.6521, 0.9029, 0.4901]])}
        >>> g
        DGLGraph(num_nodes=5, num_edges=5,
                ndata_schemes={}
                edata_schemes={})
        >>> subg.copy_to_parent()
        >>> g.ndata
        {'h': tensor([[0.3803, 0.9351, 0.0611],
                [0.6492, 0.4327, 0.3610],
                [0.0000, 0.0000, 0.0000],
                [0.7471, 0.4257, 0.4130],
                [0.9766, 0.6280, 0.6075]])}
        >>> g.edata
        {'h': tensor([[0.8192, 0.2409, 0.6278],
                [0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000],
                [0.9600, 0.3501, 0.8037],
                [0.6521, 0.9029, 0.4901]])}

        Notes
        -----
        This API excludes the ``_ID`` field in both node frame and edge frame.
        This being said if user take a subgraph ``sg`` of a graph ``g`` and
        apply :func:`~dgl.copy_from_parent` on ``sg``, it would not polluate the
        ``_ID`` field of node/edge frame of ``g``.

        See Also
        --------
        """
        if self._parent is None:
            raise DGLError("We only support copy_to_parent for subgraphs.")
        nids = self.ndata.pop(NID)
        eids = self.edata.pop(EID)
        self._parent._node_frame.update_rows(
            utils.toindex(nids), self._node_frame, inplace=inplace)
        if self._parent._edge_frame.num_rows != 0:
            self._parent._edge_frame.update_rows(
                utils.toindex(eids), self._edge_frame, inplace=inplace)
        self.ndata[NID] = nids
        self.edata[EID] = eids

    def copy_from_parent(self):
        """Copy node/edge features from the parent graph.

        All old features will be removed.

        Examples
        --------
        >>> import dgl
        >>> import torch as th
        >>> g = dgl.DGLGraph()
        >>> g.add_nodes(5)                  # Create a DGLGraph with 5 nodes
        >>> g.add_edges([0,1,2,3,4], [1,2,3,4,0])
        >>> g.ndata['h'] = th.rand(5, 3)
        >>> g.ndata['h']
        tensor([[0.3749, 0.5681, 0.4749],
                [0.6312, 0.7955, 0.3682],
                [0.0215, 0.0303, 0.0282],
                [0.8840, 0.6842, 0.3645],
                [0.9253, 0.8427, 0.6626]])
        >>> g.edata['h'] = th.rand(5, 3)
        >>> g.edata['h']
        tensor([[0.0659, 0.8552, 0.9208],
                [0.8238, 0.0332, 0.7864],
                [0.1629, 0.4149, 0.1363],
                [0.0648, 0.6582, 0.4400],
                [0.4321, 0.1612, 0.7893]])
        >>> g
        DGLGraph(num_nodes=5, num_edges=5,
                ndata_schemes={'h': Scheme(shape=(3,), dtype=torch.float32)}
                edata_schemes={'h': Scheme(shape=(3,), dtype=torch.float32)})
        >>> subg = g.subgraph([0,1,3,4])    # Take subgraph induced by node 0,1,3,4
        >>> subg                            # '_ID' field records node/edge mapping
        DGLGraph(num_nodes=4, num_edges=3,
                ndata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)}
                edata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)})
        >>> subg.copy_from_parent()
        >>> subg.ndata
        {'h': tensor([[0.3749, 0.5681, 0.4749],
                [0.6312, 0.7955, 0.3682],
                [0.8840, 0.6842, 0.3645],
                [0.9253, 0.8427, 0.6626]]), '_ID': tensor([0, 1, 3, 4])}
        >>> subg.edata
        {'h': tensor([[0.0659, 0.8552, 0.9208],
                [0.0648, 0.6582, 0.4400],
                [0.4321, 0.1612, 0.7893]]), '_ID': tensor([0, 3, 4])}

        Notes
        -----
        This API excludes the ``_ID`` field in both node frame and edge frame.
        This being said if user take a subgraph ``sg1`` of a subgraph ``sg``
        whose ``_ID`` field in node/edge frame is not None and
        apply :func:`~dgl.copy_from_parent` on ``sg1``, it would not polluate
        the ``_ID`` field of node/edge frame of ``sg1``.

        See Also
        --------
        subgraph
        edge_subgraph
        parent_nid
        parent_eid
        copy_to_parent
        map_to_subgraph_nid
        """
        if self._parent is None:
            raise DGLError("We only support copy_from_parent for subgraphs.")
        nids = self.ndata[NID]
        eids = self.edata[EID]
        if self._parent._node_frame.num_rows != 0 and self._parent._node_frame.num_columns != 0:
            self._node_frame = FrameRef(Frame(
                self._parent._node_frame[utils.toindex(nids)]))
        if self._parent._edge_frame.num_rows != 0 and self._parent._edge_frame.num_columns != 0:
            self._edge_frame = FrameRef(Frame(
                self._parent._edge_frame[utils.toindex(eids)]))
        self.ndata[NID] = nids
        self.edata[NID] = eids

    def map_to_subgraph_nid(self, parent_vids):
        """Map the node Ids in the parent graph to the node Ids in the subgraph.

        Parameters
        ----------
        parent_vids : list, tensor
            The node ID array in the parent graph.

        Returns
        -------
        tensor
            The node ID array in the subgraph.

        Examples
        --------
        >>> import dgl
        >>> g = dgl.DGLGraph()
        >>> g.add_nodes(5)
        >>> sg = g.subgrph([0,2,4])
        >>> sg.map_to_subgraph([2,4])
        tensor([1, 2])

        See Also
        --------
        subgraph
        edge_subgraph
        parent_nid
        parent_eid
        copy_to_parent
        copy_from_parent
        """
        if self._parent is None:
            raise DGLError("We only support map_to_subgraph_nid for subgraphs.")
        v = graph_index.map_to_subgraph_nid(
            utils.toindex(self.ndata[NID]), utils.toindex(parent_vids))
        return v.tousertensor()

    def flatten(self):
        """Remove all batching information of the graph, and regard the current
        graph as an independent graph rather then a batched graph.
        Graph topology and attributes would not be affected.

        User can change the structure of the flattened graph.

        Examples
        --------
        >>> import dgl
        >>> import torch as th
        >>> g_list = []
        >>> for _ in range(3)            # Create three graphs, each with #nodes 4
        >>>     g = dgl.DGLGraph()
        >>>     g.add_nodes(4)
        >>>     g.add_edges([0,1,2,3], [1,2,3,0])
        >>>     g.ndata['h'] = th.rand(4, 3)
        >>>     g_list.append(g)
        >>> bg = dgl.batch(g_list)
        >>> bg.ndata
        {'h': tensor([[0.0463, 0.1251, 0.5967],
                [0.8633, 0.9812, 0.8601],
                [0.7828, 0.3624, 0.7845],
                [0.2169, 0.8761, 0.3237],
                [0.1752, 0.1478, 0.5611],
                [0.5279, 0.2556, 0.2304],
                [0.8950, 0.8203, 0.5604],
                [0.2999, 0.2946, 0.2676],
                [0.3419, 0.2935, 0.6618],
                [0.8137, 0.8927, 0.8953],
                [0.6229, 0.7153, 0.5041],
                [0.5659, 0.0612, 0.2351]])}
        >>> bg.batch_size
        3
        >>> bg.batch_num_nodes
        [4, 4, 4]
        >>> bg.batch_num_edges
        [4, 4, 4]
        >>> bg.flatten()
        >>> bg.batch_size
        1
        >>> bg.batch_num_nodes
        [12]
        >>> bg.batch_num_edges
        [12]
        >>> bg.remove_nodes([1,3,5,7,9,11])
        >>> bg.ndata
        {'h': tensor([[0.0463, 0.1251, 0.5967],
                [0.7828, 0.3624, 0.7845],
                [0.1752, 0.1478, 0.5611],
                [0.8950, 0.8203, 0.5604],
                [0.3419, 0.2935, 0.6618],
                [0.6229, 0.7153, 0.5041]])}
        """
        self._batch_num_nodes = None
        self._batch_num_edges = None

    def detach_parent(self):
        """Detach the current graph from its parent, and regard the current graph
        as an independent graph rather then a subgraph.
        Graph topology and attributes would not be affected.

        User can change the structure of the detached graph.

        Examples
        --------
        >>> import dgl
        >>> import torch as th
        >>> g = dgl.DGLGraph()              # Graph 1
        >>> g.add_nodes(5)
        >>> g.ndata['h'] = th.rand(5, 3)
        >>> g.ndata['h']
        {'h': tensor([[0.9595, 0.7450, 0.5495],
                [0.8253, 0.2902, 0.4393],
                [0.3783, 0.4548, 0.6075],
                [0.2323, 0.0936, 0.6580],
                [0.1624, 0.3484, 0.3750]])}
        >>> subg = g.subgraph([0,1,3])      # Create a subgraph
        >>> subg.parent                     # Get the parent reference of subg
        DGLGraph(num_nodes=5, num_edges=0,
                 ndata_schemes={'h': Scheme(shape=(3,), dtype=torch.float32)}
                 edata_schemes={})
        >>> subg.copy_from_parent()
        >>> subg.detach_parent()            # Detach the subgraph from its parent
        >>> subg.parent == None
        True
        >>> subg.add_nodes(1)               # Change the structure of the subgraph
        >>> subg
        DGLGraph(num_nodes=4, num_edges=0,
                 ndata_schemes={'h': Scheme(shape=(3,), dtype=torch.float32)}
                 edata_schemes={})
        >>> subg.ndata
        {'h': tensor([[0.9595, 0.7450, 0.5495],
                [0.8253, 0.2902, 0.4393],
                [0.2323, 0.0936, 0.6580],
                [0.0000, 0.0000, 0.0000]])}
        """
        self._parent = None
        self.ndata.pop(NID)
        self.edata.pop(EID)

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
        self._msg_index = None
        self._msg_frame.clear()

    def clear_cache(self):
        """Clear all cached graph structures such as adjmat.

        By default, all graph structure related sparse matrices (e.g. adjmat, incmat)
        are cached so they could be reused with the cost of extra memory consumption.
        This function can be used to clear the cached matrices if memory is an issue.
        """
        self._graph.clear_cache()

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
                feat_dict = self.get_n_repr(nid)
                attr.update({key: F.squeeze(feat_dict[key], 0) for key in node_attrs})
        if edge_attrs is not None:
            for _, _, attr in nx_graph.edges(data=True):
                eid = attr['id']
                feat_dict = self.get_e_repr(eid)
                attr.update({key: F.squeeze(feat_dict[key], 0) for key in edge_attrs})
        return nx_graph

    def from_networkx(self, nx_graph, node_attrs=None, edge_attrs=None):
        """Convert from networkx graph.

        If 'id' edge attribute exists, the edge will be added follows
        the edge id order. Otherwise, order is undefined.

        Parameters
        ----------
        nx_graph : networkx.DiGraph
            If the node labels of ``nx_graph`` are not consecutive
            integers, its nodes will be relabeled using consecutive integers.
            The new node ordering will inherit that of ``sorted(nx_graph.nodes())``
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
        # Relabel nodes using consecutive integers
        nx_graph = nx.convert_node_labels_to_integers(nx_graph, ordering='sorted')
        # With to_directed we will get a directed version of the original networkx
        # graph, with the original nodes, edges and their attributes preserved.
        # This is particularly helpful when we are also converting the edge attributes
        # as the reversed edges (u, v) will be created with the same attributes as the
        # original edges (v, u).
        if not nx_graph.is_directed():
            nx_graph = nx_graph.to_directed()

        self.clear()
        self._graph = graph_index.from_networkx(nx_graph, self.is_readonly)
        self._node_frame.add_rows(self.number_of_nodes())
        self._edge_frame.add_rows(self.number_of_edges())
        self._msg_frame.add_rows(self.number_of_edges())

        # copy attributes
        def _batcher(lst):
            if F.is_tensor(lst[0]):
                return F.cat([F.unsqueeze(x, 0) for x in lst], dim=0)
            else:
                return F.tensor(lst)
        if node_attrs is not None:
            # mapping from feature name to a list of tensors to be concatenated
            attr_dict = defaultdict(list)
            for nid in range(self.number_of_nodes()):
                for attr in node_attrs:
                    attr_dict[attr].append(nx_graph.nodes[nid][attr])
            for attr in node_attrs:
                self._node_frame[attr] = _batcher(attr_dict[attr])
        if edge_attrs is not None:
            has_edge_id = 'id' in next(iter(nx_graph.edges(data=True)))[-1]
            # mapping from feature name to a list of tensors to be concatenated
            attr_dict = defaultdict(lambda: [None] * self.number_of_edges())
            # each defaultdict value is initialized to be a list of None
            # None here serves as placeholder to be replaced by feature with
            # corresponding edge id
            if has_edge_id:
                num_edges = self.number_of_edges()
                for _, _, attrs in nx_graph.edges(data=True):
                    if attrs['id'] >= num_edges:
                        raise DGLError('Expect the pre-specified edge ids to be'
                                       ' smaller than the number of edges --'
                                       ' {}, got {}.'.format(num_edges, attrs['id']))
                    for key in edge_attrs:
                        attr_dict[key][attrs['id']] = attrs[key]
            else:
                # XXX: assuming networkx iteration order is deterministic
                #      so the order is the same as graph_index.from_networkx
                for eid, (_, _, attrs) in enumerate(nx_graph.edges(data=True)):
                    for key in edge_attrs:
                        attr_dict[key][eid] = attrs[key]
            for attr in edge_attrs:
                for val in attr_dict[attr]:
                    if val is None:
                        raise DGLError('Not all edges have attribute {}.'.format(attr))
                self._edge_frame[attr] = _batcher(attr_dict[attr])

    def from_scipy_sparse_matrix(self, spmat, multigraph=None):
        """ Convert from scipy sparse matrix.

        Parameters
        ----------
        spmat : scipy sparse matrix
            The graph's adjacency matrix

        multigraph : bool, optional
            Deprecated (Will be deleted in the future).
            Whether the graph would be a multigraph. If the input scipy sparse matrix is CSR,
            this argument is ignored.

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
        if multigraph is not None:
            dgl_warning("multigraph will be deprecated." \
                        "DGL will treat all graphs as multigraph in the future.")

        self._graph = graph_index.from_scipy_sparse_matrix(spmat, self.is_readonly)
        self._node_frame.add_rows(self.number_of_nodes())
        self._edge_frame.add_rows(self.number_of_edges())
        self._msg_frame.add_rows(self.number_of_edges())

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
    def srcdata(self):
        """Compatibility interface with heterogeneous graphs; identical to ``ndata``"""
        return self.ndata

    @property
    def dstdata(self):
        """Compatibility interface with heterogeneous graphs; identical to ``ndata``"""
        return self.ndata

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

    @property
    def batch_size(self):
        """Number of graphs in this batch.

        Returns
        -------
        int
            Number of graphs in this batch."""
        return 1 if self.batch_num_nodes is None else len(self.batch_num_nodes)

    @property
    def batch_num_nodes(self):
        """Number of nodes of each graph in this batch.

        Returns
        -------
        list
            Number of nodes of each graph in this batch."""
        if self._batch_num_nodes is None:
            return [self.number_of_nodes()]
        else:
            return self._batch_num_nodes

    @property
    def batch_num_edges(self):
        """Number of edges of each graph in this batch.

        Returns
        -------
        list
            Number of edges of each graph in this batch."""
        if self._batch_num_edges is None:
            return [self.number_of_edges()]
        else:
            return self._batch_num_edges

    @property
    def parent(self):
        """If current graph is a subgraph of a parent graph, return
        its parent graph, else return None.

        Returns
        -------
        DGLGraph or None
            The parent graph of current graph.
        """
        return self._parent

    def init_ndata(self, ndata_name, shape, dtype, ctx=F.cpu()):
        """Create node embedding.

        It first creates the node embedding in the server and maps it to the current process
        with shared memory.

        Parameters
        ----------
        ndata_name : string
            The name of node embedding
        shape : tuple
            The shape of the node embedding
        dtype : string
            The data type of the node embedding. The currently supported data types
            are "float32" and "int32".
        ctx : DGLContext
            The column context.
        """
        scheme = Scheme(tuple(shape[1:]), F.data_type_dict[dtype])
        self._node_frame._frame.add_column(ndata_name, scheme, ctx)

    def init_edata(self, edata_name, shape, dtype, ctx=F.cpu()):
        """Create edge embedding.

        It first creates the edge embedding in the server and maps it to the current process
        with shared memory.

        Parameters
        ----------
        edata_name : string
            The name of edge embedding
        shape : tuple
            The shape of the edge embedding
        dtype : string
            The data type of the edge embedding. The currently supported data types
            are "float32" and "int32".
        ctx : DGLContext
            The column context.
        """
        scheme = Scheme(tuple(shape[1:]), F.data_type_dict[dtype])
        self._edge_frame._frame.add_column(edata_name, scheme, ctx)


    def set_n_repr(self, data, u=ALL, inplace=False):
        """Set node(s) representation.

        `data` is a dictionary from the feature name to feature tensor. Each tensor
        is of shape (B, D1, D2, ...), where B is the number of nodes to be updated,
        and (D1, D2, ...) be the shape of the node representation tensor. The
        length of the given node ids must match B (i.e, len(u) == B).

        All update will be done out of place to work with autograd unless the
        inplace flag is true.

        Parameters
        ----------
        data : dict of tensor
            Node representation.
        u : node, container or tensor
            The node(s).
        inplace : bool
            If True, update will be done in place, but autograd will break.
        """
        # sanity check
        if not utils.is_dict_like(data):
            raise DGLError('Expect dictionary type for feature data.'
                           ' Got "%s" instead.' % type(data))
        if is_all(u):
            num_nodes = self.number_of_nodes()
        else:
            u = utils.toindex(u)
            num_nodes = len(u)
        for key, val in data.items():
            nfeats = F.shape(val)[0]
            if nfeats != num_nodes:
                raise DGLError('Expect number of features to match number of nodes (len(u)).'
                               ' Got %d and %d instead.' % (nfeats, num_nodes))
        # set
        if is_all(u):
            for key, val in data.items():
                self._node_frame[key] = val
        else:
            self._node_frame.update_rows(u, data, inplace=inplace)

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

    def set_e_repr(self, data, edges=ALL, inplace=False):
        """Set edge(s) representation.

        `data` is a dictionary from the feature name to feature tensor. Each tensor
        is of shape (B, D1, D2, ...), where B is the number of edges to be updated,
        and (D1, D2, ...) be the shape of the edge representation tensor.

        All update will be done out of place to work with autograd unless the
        inplace flag is true.

        Parameters
        ----------
        data : tensor or dict of tensor
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
        if not utils.is_dict_like(data):
            raise DGLError('Expect dictionary type for feature data.'
                           ' Got "%s" instead.' % type(data))

        if is_all(eid):
            num_edges = self.number_of_edges()
        else:
            eid = utils.toindex(eid)
            num_edges = len(eid)
        for key, val in data.items():
            nfeats = F.shape(val)[0]
            if nfeats != num_edges:
                raise DGLError('Expect number of features to match number of edges.'
                               ' Got %d and %d instead.' % (nfeats, num_edges))
        # set
        if is_all(eid):
            # update column
            for key, val in data.items():
                self._edge_frame[key] = val
        else:
            # update row
            self._edge_frame.update_rows(eid, data, inplace=inplace)

    def get_e_repr(self, edges=ALL):
        """Get edge(s) representation.

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
            scheduler.schedule_apply_nodes(v=v,
                                           apply_func=func,
                                           node_frame=self._node_frame,
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
            u, v, _ = self._graph.edges('eid')
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
            scheduler.schedule_apply_edges(AdaptedDGLGraph(self), u, v, eid, func, inplace)
            Runtime.run(prog)

    def group_apply_edges(self, group_by, func, edges=ALL, inplace=False):
        """Group the edges by nodes and apply the function on the grouped edges to
         update their features.

        Parameters
        ----------
        group_by : str
            Specify how to group edges. Expected to be either 'src' or 'dst'
        func : callable
            Apply function on the edge. The function should be
            an :mod:`Edge UDF <dgl.udf>`. The input of `Edge UDF` should
            be (bucket_size, degrees, *feature_shape), and
            return the dict with values of the same shapes.
        edges : valid edges type, optional
            Edges on which to group and apply ``func``. See :func:`send` for valid
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
        >>> g.add_nodes(4)
        >>> g.add_edges(0, [1, 2, 3])
        >>> g.add_edges(1, [2, 3])
        >>> g.add_edges(2, [2, 3])
        >>> g.edata['feat'] = th.randn((g.number_of_edges(), 1))

        >>> # Softmax over the out edges of each node
        >>> # Second dimension of edges.data is the degree dimension
        >>> def softmax_feat(edges): return {'norm_feat': th.softmax(edges.data['feat'], dim=1)}
        >>> g.group_apply_edges(func=softmax_feat, group_by='src') # Apply func to the first edge.
        >>> u, v, eid = g.out_edges(1, form='all')
        >>> in_feat = g.edata['feat'][eid]
        >>> out_feat = g.edata['norm_feat'][eid]
        >>> print(out_feat - th.softmax(in_feat, 0))
            tensor([[0.],
            [0.]])

        See Also
        --------
        apply_edges
        """
        assert func is not None

        if group_by not in ('src', 'dst'):
            raise DGLError("Group_by should be either src or dst")

        if is_all(edges):
            u, v, _ = self._graph.edges('eid')
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
            scheduler.schedule_group_apply_edge(graph=AdaptedDGLGraph(self),
                                                u=u, v=v, eid=eid,
                                                apply_func=func,
                                                group_by=group_by,
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
            eid = utils.toindex(slice(0, self.number_of_edges()))
            u, v, _ = self._graph.edges('eid')
        elif isinstance(edges, tuple):
            u, v = edges
            u = utils.toindex(u)
            v = utils.toindex(v)
            # Rewrite u, v to handle edge broadcasting and multigraph.
            u, v, eid = self._graph.edge_ids(u, v)
        else:
            eid = utils.toindex(edges)
            u, v, _ = self._graph.find_edges(eid)

        if len(eid) == 0:
            # no edge to be triggered
            return

        with ir.prog() as prog:
            scheduler.schedule_send(graph=AdaptedDGLGraph(self), u=u, v=v, eid=eid,
                                    message_func=message_func)
            Runtime.run(prog)

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

        if is_all(v):
            v = F.arange(0, self.number_of_nodes())
        elif isinstance(v, int):
            v = [v]
        v = utils.toindex(v)
        if len(v) == 0:
            # no vertex to be triggered.
            return

        with ir.prog() as prog:
            scheduler.schedule_recv(graph=AdaptedDGLGraph(self),
                                    recv_nodes=v,
                                    reduce_func=reduce_func,
                                    apply_func=apply_node_func,
                                    inplace=inplace)
            Runtime.run(prog)

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
            scheduler.schedule_snr(graph=AdaptedDGLGraph(self),
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
            scheduler.schedule_pull(graph=AdaptedDGLGraph(self),
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
            scheduler.schedule_push(graph=AdaptedDGLGraph(self),
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
            scheduler.schedule_update_all(graph=AdaptedDGLGraph(self),
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
            self.pull(node_frontier, message_func, reduce_func, apply_node_func)

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
            self.send_and_recv(edge_frontier, message_func, reduce_func, apply_node_func)

    def subgraph(self, nodes):
        """Return the subgraph induced on given nodes.

        Parameters
        ----------
        nodes : list, or iterable
            A node ID array to construct subgraph.
            All nodes must exist in the graph.

        Returns
        -------
        G : DGLGraph
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
        subgraphs
        edge_subgraph
        parent_nid
        parent_eid
        copy_from_parent
        copy_to_parent
        map_to_subgraph_nid
        """
        induced_nodes = utils.toindex(nodes)
        sgi = self._graph.node_subgraph(induced_nodes)
        return self._create_subgraph(sgi, sgi.induced_nodes, sgi.induced_edges)

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
        G : A list of DGLGraph
            The subgraphs.

        See Also
        --------
        subgraph
        parent_nid
        parent_eid
        copy_from_parent
        copy_to_parent
        map_to_subgraph_nid
        """
        induced_nodes = [utils.toindex(n) for n in nodes]
        sgis = self._graph.node_subgraphs(induced_nodes)
        return [self._create_subgraph(
            sgi, sgi.induced_nodes, sgi.induced_edges) for sgi in sgis]

    def edge_subgraph(self, edges, preserve_nodes=False):
        """Return the subgraph induced on given edges.

        Parameters
        ----------
        edges : list, or iterable
            An edge ID array to construct subgraph.
            All edges must exist in the subgraph.
        preserve_nodes : bool
            Indicates whether to preserve all nodes or not.
            If true, keep the nodes which have no edge connected in the subgraph;
            If false, all nodes without edge connected to it would be removed.

        Returns
        -------
        G : DGLGraph
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
        >>> SG = G.edge_subgraph([0, 4], preserve_nodes=True)
        >>> SG.nodes()
        tensor([0, 1, 2, 3, 4])
        >>> SG.edges()
        (tensor([0, 4]), tensor([1, 0]))
        >>> SG.parent_nid
        tensor([0, 1, 2, 3, 4])
        >>> SG.parent_eid
        tensor([0, 4])

        See Also
        --------
        subgraph
        copy_from_parent
        copy_to_parent
        map_to_subgraph_nid
        """
        induced_edges = utils.toindex(edges)
        sgi = self._graph.edge_subgraph(induced_edges, preserve_nodes=preserve_nodes)
        return self._create_subgraph(sgi, sgi.induced_nodes, sgi.induced_edges)

    def adjacency_matrix_scipy(self, transpose=None, fmt='csr', return_edge_ids=None):
        """Return the scipy adjacency matrix representation of this graph.

        By default, a row of returned adjacency matrix represents the destination
        of an edge and the column represents the source.

        When transpose is True, a row represents the source and a column represents
        a destination.

        Parameters
        ----------
        transpose : bool, optional (default=False)
            A flag to transpose the returned adjacency matrix.
        fmt : str, optional (default='csr')
            Indicates the format of returned adjacency matrix.
        return_edge_ids : bool, optional (default=True)
            If True, the elements in the adjacency matrix are edge ids.
            Note that one of the element is 0.  Proceed with caution.
            If False, the elements will be always 1.

        Returns
        -------
        scipy.sparse.spmatrix
            The scipy representation of adjacency matrix.

        """
        if transpose is None:
            dgl_warning(
                "Currently adjacency_matrix() returns a matrix with destination as rows"
                " by default.  In 0.5 the result will have source as rows"
                " (i.e. transpose=True)")
            transpose = False
        return self._graph.adjacency_matrix_scipy(transpose, fmt, return_edge_ids)

    def adjacency_matrix(self, transpose=None, ctx=F.cpu()):
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
        if transpose is None:
            dgl_warning(
                "Currently adjacency_matrix() returns a matrix with destination as rows"
                " by default.  In 0.5 the result will have source as rows"
                " (i.e. transpose=True)")
            transpose = False
        return self._graph.adjacency_matrix(transpose, ctx)[0]

    def incidence_matrix(self, typestr, ctx=F.cpu()):
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
        typestr : str
            Can be either ``in``, ``out`` or ``both``
        ctx : context, optional (default=cpu)
            The context of returned incidence matrix.

        Returns
        -------
        SparseTensor
            The incidence matrix.
        """
        return self._graph.incidence_matrix(typestr, ctx)[0]

    def line_graph(self, backtracking=True, shared=False):
        """Return the line graph of this graph.

        See :func:`~dgl.transform.line_graph`.
        """
        return dgl.line_graph(self, backtracking, shared)

    def reverse(self, share_ndata=False, share_edata=False):
        """Return the reverse of this graph.

        See :func:`~dgl.transform.reverse`.
        """
        return dgl.reverse(self, share_ndata, share_edata)

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
        nbatch = NodeBatch(v, n_repr)
        n_mask = F.copy_to(predicate(nbatch), F.cpu())

        if is_all(nodes):
            return F.nonzero_1d(n_mask)
        else:
            nodes = F.tensor(nodes)
            return F.boolean_mask(nodes, n_mask)

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
            u, v, _ = self._graph.edges('eid')
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

        src_data = self.get_n_repr(u)
        edge_data = self.get_e_repr(eid)
        dst_data = self.get_n_repr(v)
        ebatch = EdgeBatch((u, v, eid), src_data, edge_data, dst_data)
        e_mask = F.copy_to(predicate(ebatch), F.cpu())

        if is_all(edges):
            return F.nonzero_1d(e_mask)
        else:
            edges = F.tensor(edges)
            return F.boolean_mask(edges, e_mask)

    def readonly(self, readonly_state=True):
        """Set this graph's readonly state in-place.

        Parameters
        ----------
        readonly_state : bool, optional
            New readonly state of the graph, defaults to True.

        Examples
        --------
        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(3)
        >>> G.add_edge(0, 1)
        >>> G.readonly()
        >>> try:
        >>>     G.add_nodes(5)
        >>>     fail = False
        >>> except:
        >>>     fail = True
        >>>
        >>> fail
        True
        >>> G.readonly(False)
        >>> G.add_nodes(5)
        >>> G.number_of_nodes()
        8
        """
        if readonly_state != self.is_readonly:
            self._graph.readonly(readonly_state)

    def __repr__(self):
        ret = ('DGLGraph(num_nodes={node}, num_edges={edge},\n'
               '         ndata_schemes={ndata}\n'
               '         edata_schemes={edata})')
        return ret.format(node=self.number_of_nodes(), edge=self.number_of_edges(),
                          ndata=str(self.node_attr_schemes()),
                          edata=str(self.edge_attr_schemes()))

    # pylint: disable=invalid-name
    def to(self, ctx, **kwargs):
        """Move both ndata and edata to the targeted mode (cpu/gpu)
        Framework agnostic

        Parameters
        ----------
        ctx : framework-specific context object
            The context to move data to.

        Returns
        -------
        g : DGLGraph
          Moved DGLGraph of the targeted mode.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import torch
        >>> G = dgl.DGLGraph()
        >>> G.add_nodes(5, {'h': torch.ones((5, 2))})
        >>> G.add_edges([0, 1], [1, 2], {'m' : torch.ones((2, 2))})
        >>> G.add_edges([0, 1], [1, 2], {'m' : torch.ones((2, 2))})
        >>> G = G.to(torch.device('cuda:0'))
        """
        for k in self.ndata.keys():
            self.ndata[k] = F.copy_to(self.ndata[k], ctx, **kwargs)
        for k in self.edata.keys():
            self.edata[k] = F.copy_to(self.edata[k], ctx, **kwargs)
        return self
    # pylint: enable=invalid-name

    def local_var(self):
        """Return a graph object that can be used in a local function scope.

        The returned graph object shares the feature data and graph structure of this graph.
        However, any out-place mutation to the feature data will not reflect to this graph,
        thus making it easier to use in a function scope.

        If set, the local graph object will use same initializers for node features and
        edge features.

        Examples
        --------
        The following example uses PyTorch backend.

        Avoid accidentally overriding existing feature data. This is quite common when
        implementing a NN module:

        >>> def foo(g):
        >>>     g = g.local_var()
        >>>     g.ndata['h'] = torch.ones((g.number_of_nodes(), 3))
        >>>     return g.ndata['h']
        >>>
        >>> g = ... # some graph
        >>> g.ndata['h'] = torch.zeros((g.number_of_nodes(), 3))
        >>> newh = foo(g)  # get tensor of all ones
        >>> print(g.ndata['h'])  # still get tensor of all zeros

        Automatically garbage collect locally-defined tensors without the need to manually
        ``pop`` the tensors.

        >>> def foo(g):
        >>>     g = g.local_var()
        >>>     # This 'xxx' feature will stay local and be GCed when the function exits
        >>>     g.ndata['xxx'] = torch.ones((g.number_of_nodes(), 3))
        >>>     return g.ndata['xxx']
        >>>
        >>> g = ... # some graph
        >>> xxx = foo(g)
        >>> print('xxx' in g.ndata)
        False

        Notes
        -----
        Internally, the returned graph shares the same feature tensors, but construct a new
        dictionary structure (aka. Frame) so adding/removing feature tensors from the returned
        graph will not reflect to the original graph. However, inplace operations do change
        the shared tensor values, so will be reflected to the original graph. This function
        also has little overhead when the number of feature tensors in this graph is small.

        See Also
        --------
        local_var

        Returns
        -------
        DGLGraph
            The graph object that can be used as a local variable.
        """
        local_node_frame = FrameRef(Frame(self._node_frame._frame))
        local_edge_frame = FrameRef(Frame(self._edge_frame._frame))
        # Use same per-column initializers and default initializer.
        # If registered, a column (based on key) initializer will be used first,
        # otherwise the default initializer will be used.
        sync_frame_initializer(local_node_frame._frame, self._node_frame._frame)
        sync_frame_initializer(local_edge_frame._frame, self._edge_frame._frame)
        return DGLGraph(graph_data=self._graph,
                        node_frame=local_node_frame,
                        edge_frame=local_edge_frame,
                        readonly=self.is_readonly,
                        batch_num_nodes=self.batch_num_nodes,
                        batch_num_edges=self.batch_num_edges,
                        parent=self._parent)

    @contextmanager
    def local_scope(self):
        """Enter a local scope context for this graph.

        By entering a local scope, any out-place mutation to the feature data will
        not reflect to the original graph, thus making it easier to use in a function scope.

        If set, the local scope will use same initializers for node features and
        edge features.

        Examples
        --------
        The following example uses PyTorch backend.

        Avoid accidentally overriding existing feature data. This is quite common when
        implementing a NN module:

        >>> def foo(g):
        >>>     with g.local_scope():
        >>>         g.ndata['h'] = torch.ones((g.number_of_nodes(), 3))
        >>>         return g.ndata['h']
        >>>
        >>> g = ... # some graph
        >>> g.ndata['h'] = torch.zeros((g.number_of_nodes(), 3))
        >>> newh = foo(g)  # get tensor of all ones
        >>> print(g.ndata['h'])  # still get tensor of all zeros

        Automatically garbage collect locally-defined tensors without the need to manually
        ``pop`` the tensors.

        >>> def foo(g):
        >>>     with g.local_scope():
        >>>     # This 'xxx' feature will stay local and be GCed when the function exits
        >>>         g.ndata['xxx'] = torch.ones((g.number_of_nodes(), 3))
        >>>         return g.ndata['xxx']
        >>>
        >>> g = ... # some graph
        >>> xxx = foo(g)
        >>> print('xxx' in g.ndata)
        False

        See Also
        --------
        local_var
        """
        old_nframe = self._node_frame
        old_eframe = self._edge_frame
        self._node_frame = FrameRef(Frame(self._node_frame._frame))
        self._edge_frame = FrameRef(Frame(self._edge_frame._frame))
        # Use same per-column initializers and default initializer.
        # If registered, a column (based on key) initializer will be used first,
        # otherwise the default initializer will be used.
        sync_frame_initializer(self._node_frame._frame, old_nframe._frame)
        sync_frame_initializer(self._edge_frame._frame, old_eframe._frame)
        yield
        self._node_frame = old_nframe
        self._edge_frame = old_eframe

    @property
    def is_homogeneous(self):
        """Return if the graph is homogeneous."""
        return True

############################################################
# Batch/Unbatch APIs
############################################################

def batch(graph_list, node_attrs=ALL, edge_attrs=ALL):
    """Batch a collection of :class:`~dgl.DGLGraph` and return a batched
    :class:`DGLGraph` object that is independent of the :attr:`graph_list` so that
    one can perform message passing and readout over a batch of graphs
    simultaneously, the batch size of the returned graph is the length of
    :attr:`graph_list`.

    The nodes and edges are re-indexed with a new id in the batched graph with the
    rule below:

    ======  ==========  ========================  ===  ==========================
    item    Graph 1     Graph 2                   ...  Graph k
    ======  ==========  ========================  ===  ==========================
    raw id  0, ..., N1       0, ..., N2           ...  ..., Nk
    new id  0, ..., N1  N1 + 1, ..., N1 + N2 + 1  ...  ..., N1 + ... + Nk + k - 1
    ======  ==========  ========================  ===  ==========================

    To modify the features in the batched graph has no effect on the original
    graphs. See the examples below about how to work around.

    Parameters
    ----------
    graph_list : iterable
        A collection of :class:`~dgl.DGLGraph` to be batched.
    node_attrs : None, str or iterable
        The node attributes to be batched. If ``None``, the returned :class:`DGLGraph`
        object will not have any node attributes. By default, all node attributes will
        be batched. If ``str`` or iterable, this should specify exactly what node
        attributes to be batched.
    edge_attrs : None, str or iterable, optional
        Same as for the case of :attr:`node_attrs`

    Returns
    -------
    DGLGraph
        One single batched graph.

    Examples
    --------
    Create two :class:`~dgl.DGLGraph` objects.
    **Instantiation:**

    >>> import dgl
    >>> import torch as th
    >>> g1 = dgl.DGLGraph()
    >>> g1.add_nodes(2)                                # Add 2 nodes
    >>> g1.add_edge(0, 1)                              # Add edge 0 -> 1
    >>> g1.ndata['hv'] = th.tensor([[0.], [1.]])       # Initialize node features
    >>> g1.edata['he'] = th.tensor([[0.]])             # Initialize edge features
    >>> g2 = dgl.DGLGraph()
    >>> g2.add_nodes(3)                                # Add 3 nodes
    >>> g2.add_edges([0, 2], [1, 1])                   # Add edges 0 -> 1, 2 -> 1
    >>> g2.ndata['hv'] = th.tensor([[2.], [3.], [4.]]) # Initialize node features
    >>> g2.edata['he'] = th.tensor([[1.], [2.]])       # Initialize edge features

    Merge two :class:`~dgl.DGLGraph` objects into one :class:`DGLGraph` object.
    When merging a list of graphs, we can choose to include only a subset of the attributes.

    >>> bg = dgl.batch([g1, g2], edge_attrs=None)
    >>> bg.edata
    {}

    Below one can see that the nodes are re-indexed. The edges are re-indexed in
    the same way.

    >>> bg.nodes()
    tensor([0, 1, 2, 3, 4])
    >>> bg.ndata['hv']
    tensor([[0.],
            [1.],
            [2.],
            [3.],
            [4.]])

    **Property:**
    We can still get a brief summary of the graphs that constitute the batched graph.

    >>> bg.batch_size
    2
    >>> bg.batch_num_nodes
    [2, 3]
    >>> bg.batch_num_edges
    [1, 2]

    **Readout:**
    Another common demand for graph neural networks is graph readout, which is a
    function that takes in the node attributes and/or edge attributes for a graph
    and outputs a vector summarizing the information in the graph.
    DGL also supports performing readout for a batch of graphs at once.
    Below we take the built-in readout function :func:`sum_nodes` as an example, which
    sums over a particular kind of node attribute for each graph.

    >>> dgl.sum_nodes(bg, 'hv') # Sum the node attribute 'hv' for each graph.
    tensor([[1.],               # 0 + 1
            [9.]])              # 2 + 3 + 4

    **Message passing:**
    For message passing and related operations, batched :class:`DGLGraph` acts exactly
    the same as a single :class:`~dgl.DGLGraph` with batch size 1.

    **Update Attributes:**
    Updating the attributes of the batched graph has no effect on the original graphs.

    >>> bg.edata['he'] = th.zeros(3, 2)
    >>> g2.edata['he']
    tensor([[1.],
            [2.]])}

    Instead, we can decompose the batched graph back into a list of graphs and use them
    to replace the original graphs.

    >>> g1, g2 = dgl.unbatch(bg)    # returns a list of DGLGraph objects
    >>> g2.edata['he']
    tensor([[0., 0.],
            [0., 0.]])}

    See Also
    --------
    unbatch
    """
    if len(graph_list) == 1:
        # Need to deepcopy the node/edge frame of original graph.
        graph = graph_list[0]
        return DGLGraph(graph_data=graph._graph,
                        node_frame=graph._node_frame.deepclone(),
                        edge_frame=graph._edge_frame.deepclone(),
                        batch_num_nodes=graph.batch_num_nodes,
                        batch_num_edges=graph.batch_num_edges)

    def _init_attrs(attrs, mode):
        """Collect attributes of given mode (node/edge) from graph_list.

        Parameters
        ----------
        attrs: None or ALL or str or iterator
            The attributes to collect. If ALL, check if all graphs have the same
            attribute set and return the attribute set. If None, return an empty
            list. If it is a string or a iterator of string, return these
            attributes.
        mode: str
            Suggest to collect node attributes or edge attributes.

        Returns
        -------
        Iterable
            The obtained attribute set.
        """
        if mode == 'node':
            nitems_list = [g.number_of_nodes() for g in graph_list]
            attrs_list = [set(g.node_attr_schemes().keys()) for g in graph_list]
        else:
            nitems_list = [g.number_of_edges() for g in graph_list]
            attrs_list = [set(g.edge_attr_schemes().keys()) for g in graph_list]

        if attrs is None:
            return []
        elif is_all(attrs):
            attrs = set()
            # Check if at least a graph has mode items and associated features.
            for i, (g_num_items, g_attrs) in enumerate(zip(nitems_list, attrs_list)):
                if g_num_items > 0 and len(g_attrs) > 0:
                    attrs = g_attrs
                    ref_g_index = i
                    break
            # Check if all the graphs with mode items have the same associated features.
            if len(attrs) > 0:
                for i, (g_num_items, g_attrs) in enumerate(zip(nitems_list, attrs_list)):
                    if g_attrs != attrs and g_num_items > 0:
                        raise ValueError('Expect graph {0} and {1} to have the same {2} '
                                         'attributes when {2}_attrs=ALL, got {3} and {4}.'
                                         .format(ref_g_index, i, mode, attrs, g_attrs))
            return attrs
        elif isinstance(attrs, str):
            return [attrs]
        elif isinstance(attrs, Iterable):
            return attrs
        else:
            raise ValueError('Expected {} attrs to be of type None str or Iterable, '
                             'got type {}'.format(mode, type(attrs)))

    node_attrs = _init_attrs(node_attrs, 'node')
    edge_attrs = _init_attrs(edge_attrs, 'edge')

    # create batched graph index
    batched_index = graph_index.disjoint_union([g._graph for g in graph_list])
    # create batched node and edge frames
    if len(node_attrs) == 0:
        batched_node_frame = FrameRef(Frame(num_rows=batched_index.number_of_nodes()))
    else:
        # NOTE: following code will materialize the columns of the input graphs.
        cols = {key: F.cat([gr._node_frame[key] for gr in graph_list
                            if gr.number_of_nodes() > 0], dim=0)
                for key in node_attrs}
        batched_node_frame = FrameRef(Frame(cols))

    if len(edge_attrs) == 0:
        batched_edge_frame = FrameRef(Frame(num_rows=batched_index.number_of_edges()))
    else:
        cols = {key: F.cat([gr._edge_frame[key] for gr in graph_list
                            if gr.number_of_edges() > 0], dim=0)
                for key in edge_attrs}
        batched_edge_frame = FrameRef(Frame(cols))

    batch_size = 0
    batch_num_nodes = []
    batch_num_edges = []
    for grh in graph_list:
        # handle the input is again a batched graph.
        batch_size += grh.batch_size
        batch_num_nodes += grh.batch_num_nodes
        batch_num_edges += grh.batch_num_edges

    return DGLGraph(graph_data=batched_index,
                    node_frame=batched_node_frame,
                    edge_frame=batched_edge_frame,
                    batch_num_nodes=batch_num_nodes,
                    batch_num_edges=batch_num_edges)

def unbatch(graph):
    """Return the list of graphs in this batch.

    Parameters
    ----------
    graph : DGLGraph
        The batched graph.

    Returns
    -------
    list
        A list of :class:`~dgl.DGLGraph` objects whose attributes are obtained
        by partitioning the attributes of the :attr:`graph`. The length of the
        list is the same as the batch size of :attr:`graph`.

    Notes
    -----
    Unbatching will break each field tensor of the batched graph into smaller
    partitions.

    For simpler tasks such as node/edge state aggregation, try to use
    readout functions.

    See Also
    --------
    batch
    """
    if graph.batch_size == 1:
        # Like dgl.batch, unbatch also deep copies data frame.
        return [DGLGraph(graph_data=graph._graph,
                         node_frame=graph._node_frame.deepclone(),
                         edge_frame=graph._edge_frame.deepclone())]

    bsize = graph.batch_size
    bnn = graph.batch_num_nodes
    bne = graph.batch_num_edges
    pttns = graph_index.disjoint_partition(graph._graph, utils.toindex(bnn))
    # split the frames
    node_frames = [FrameRef(Frame(num_rows=n)) for n in bnn]
    edge_frames = [FrameRef(Frame(num_rows=n)) for n in bne]
    for attr, col in graph._node_frame.items():
        col_splits = F.split(col, bnn, dim=0)
        for i in range(bsize):
            node_frames[i][attr] = col_splits[i]
    for attr, col in graph._edge_frame.items():
        col_splits = F.split(col, bne, dim=0)
        for i in range(bsize):
            edge_frames[i][attr] = col_splits[i]
    return [DGLGraph(graph_data=pttns[i],
                     node_frame=node_frames[i],
                     edge_frame=edge_frames[i]) for i in range(bsize)]


############################################################
# Internal APIs
############################################################

class AdaptedDGLGraph(GraphAdapter):
    """Adapt DGLGraph to interface required by scheduler.

    Parameters
    ----------
    graph : DGLGraph
        Graph
    """
    def __init__(self, graph):
        self.graph = graph

    @property
    def gidx(self):
        return self.graph._graph

    def num_src(self):
        """Number of source nodes."""
        return self.graph.number_of_nodes()

    def num_dst(self):
        """Number of destination nodes."""
        return self.graph.number_of_nodes()

    def num_edges(self):
        """Number of edges."""
        return self.graph.number_of_edges()

    @property
    def srcframe(self):
        """Frame to store source node features."""
        return self.graph._node_frame

    @property
    def dstframe(self):
        """Frame to store source node features."""
        return self.graph._node_frame

    @property
    def edgeframe(self):
        """Frame to store edge features."""
        return self.graph._edge_frame

    @property
    def msgframe(self):
        """Frame to store messages."""
        return self.graph._msg_frame

    @property
    def msgindicator(self):
        """Message indicator tensor."""
        return self.graph._get_msg_index()

    @msgindicator.setter
    def msgindicator(self, val):
        """Set new message indicator tensor."""
        self.graph._set_msg_index(val)

    def in_edges(self, nodes):
        return self.graph._graph.in_edges(nodes)

    def out_edges(self, nodes):
        return self.graph._graph.out_edges(nodes)

    def edges(self, form):
        return self.graph._graph.edges(form)

    def get_immutable_gidx(self, ctx):
        return self.graph._graph.get_immutable_gidx(ctx)

    def bits_needed(self):
        return self.graph._graph.bits_needed()

    @property
    def canonical_etype(self):
        """Canonical edge type (None for homogeneous graph)"""
        return (None, None, None)
