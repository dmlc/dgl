"""Classes for heterogeneous graphs."""
from collections import defaultdict
import networkx as nx
import scipy.sparse as ssp
from . import heterograph_index, graph_index
from . import utils
from . import backend as F
from . import init
from .runtime import ir, scheduler, Runtime
from .frame import Frame, FrameRef
from .view import HeteroNodeView, HeteroNodeDataView, HeteroEdgeView, HeteroEdgeDataView
from .base import ALL, is_all, DGLError

__all__ = ['DGLHeteroGraph']

# TODO: depending on the progress of unifying DGLGraph and Bipartite, we may or may not
# need the code of heterogeneous graph views.
# pylint: disable=unnecessary-pass
class DGLBaseHeteroGraph(object):
    """Base Heterogeneous graph class.

    Parameters
    ----------
    graph : graph index, optional
        The graph index
    ntypes : list[str]
        The node type names
    etypes : list[str]
        The edge type names
    _ntypes_invmap, _etypes_invmap, _view_ntype_idx, _view_etype_idx :
        Internal arguments
    """

    # pylint: disable=unused-argument
    def __init__(self, graph, ntypes, etypes,
                 _ntypes_invmap=None, _etypes_invmap=None,
                 _view_ntype_idx=None, _view_etype_idx=None):
        super(DGLBaseHeteroGraph, self).__init__()

        self._graph = graph
        self._ntypes = ntypes
        self._etypes = etypes
        # inverse mapping from ntype str to int
        self._ntypes_invmap = _ntypes_invmap or \
            {ntype: i for i, ntype in enumerate(ntypes)}
        # inverse mapping from etype str to int
        self._etypes_invmap = _etypes_invmap or \
            {etype: i for i, etype in enumerate(etypes)}

        # Indicates which node/edge type (int) it is viewing.
        self._view_ntype_idx = _view_ntype_idx
        self._view_etype_idx = _view_etype_idx

        self._cache = {}

    def _create_view(self, ntype_idx, etype_idx):
        return DGLBaseHeteroGraph(
            self._graph, self._ntypes, self._etypes,
            self._ntypes_invmap, self._etypes_invmap,
            ntype_idx, etype_idx)

    @property
    def is_node_type_view(self):
        """Whether this is a node type view of a heterograph."""
        return self._view_ntype_idx is not None

    @property
    def is_edge_type_view(self):
        """Whether this is an edge type view of a heterograph."""
        return self._view_etype_idx is not None

    @property
    def is_view(self):
        """Whether this is a node/view of a heterograph."""
        return self.is_node_type_view or self.is_edge_type_view

    @property
    def all_node_types(self):
        """Return the list of node types of the entire heterograph."""
        return self._ntypes

    @property
    def all_edge_types(self):
        """Return the list of edge types of the entire heterograph."""
        return self._etypes

    @property
    def metagraph(self):
        """Return the metagraph as networkx.MultiDiGraph.

        The nodes are labeled with node type names.
        The edges have their keys holding the edge type names.
        """
        nx_graph = self._graph.metagraph.to_networkx()
        nx_return_graph = nx.MultiDiGraph()
        for u_v in nx_graph.edges:
            etype = self._etypes[nx_graph.edges[u_v]['id']]
            srctype = self._ntypes[u_v[0]]
            dsttype = self._ntypes[u_v[1]]
            assert etype[0] == srctype
            assert etype[2] == dsttype
            nx_return_graph.add_edge(srctype, dsttype, etype[1])
        return nx_return_graph

    def _endpoint_types(self, etype):
        """Return the source and destination node type (int) of given edge
        type (int)."""
        return self._graph.metagraph.find_edge(etype)

    def _node_types(self):
        if self.is_node_type_view:
            return [self._view_ntype_idx]
        elif self.is_edge_type_view:
            srctype_idx, dsttype_idx = self._endpoint_types(self._view_etype_idx)
            return [srctype_idx, dsttype_idx] if srctype_idx != dsttype_idx else [srctype_idx]
        else:
            return range(len(self._ntypes))

    def node_types(self):
        """Return the list of node types appearing in the current view.

        Returns
        -------
        list[str]
            List of node types

        Examples
        --------
        Getting all node types.
        >>> g.node_types()
        ['user', 'game', 'developer']

        Getting all node types appearing in the subgraph induced by "users"
        (which should only yield "user").
        >>> g['user'].node_types()
        ['user']

        The node types appearing in subgraph induced by "plays" relationship,
        which should only give "user" and "game".
        >>> g['plays'].node_types()
        ['user', 'game']
        """
        ntypes = self._node_types()
        if isinstance(ntypes, range):
            # assuming that the range object always covers the entire node type list
            return self._ntypes
        else:
            return [self._ntypes[i] for i in ntypes]

    def _edge_types(self):
        if self.is_node_type_view:
            etype_indices = self._graph.metagraph.edge_id(
                self._view_ntype_idx, self._view_ntype_idx)
            return etype_indices
        elif self.is_edge_type_view:
            return [self._view_etype_idx]
        else:
            return range(len(self._etypes))

    def edge_types(self):
        """Return the list of edge types appearing in the current view.

        Returns
        -------
        list[str]
            List of edge types

        Examples
        --------
        Getting all edge types.
        >>> g.edge_types()
        ['follows', 'plays', 'develops']

        Getting all edge types appearing in subgraph induced by "users".
        >>> g['user'].edge_types()
        ['follows']

        The edge types appearing in subgraph induced by "plays" relationship,
        which should only give "plays".
        >>> g['plays'].edge_types()
        ['plays']
        """
        etypes = self._edge_types()
        if isinstance(etypes, range):
            return self._etypes
        else:
            return [self._etypes[i] for i in etypes]

    @property
    @utils.cached_member('_cache', '_current_ntype_idx')
    def _current_ntype_idx(self):
        """Checks the uniqueness of node type in the view and get the index
        of that node type.

        This allows reading/writing node frame data.
        """
        node_types = self._node_types()
        assert len(node_types) == 1, "only available for subgraphs with one node type"
        return node_types[0]

    @property
    @utils.cached_member('_cache', '_current_etype_idx')
    def _current_etype_idx(self):
        """Checks the uniqueness of edge type in the view and get the index
        of that edge type.

        This allows reading/writing edge frame data and message passing routines.
        """
        edge_types = self._edge_types()
        assert len(edge_types) == 1, "only available for subgraphs with one edge type"
        return edge_types[0]

    @property
    @utils.cached_member('_cache', '_current_srctype_idx')
    def _current_srctype_idx(self):
        """Checks the uniqueness of edge type in the view and get the index
        of the source type.

        This allows reading/writing edge frame data and message passing routines.
        """
        srctype_idx, _ = self._endpoint_types(self._current_etype_idx)
        return srctype_idx

    @property
    @utils.cached_member('_cache', '_current_dsttype_idx')
    def _current_dsttype_idx(self):
        """Checks the uniqueness of edge type in the view and get the index
        of the destination type.

        This allows reading/writing edge frame data and message passing routines.
        """
        _, dsttype_idx = self._endpoint_types(self._current_etype_idx)
        return dsttype_idx

    def number_of_nodes(self, ntype):
        """Return the number of nodes of the given type in the heterograph.

        Parameters
        ----------
        ntype : str
            The node type

        Returns
        -------
        int
            The number of nodes

        Examples
        --------
        >>> g['user'].number_of_nodes()
        3
        """
        return self._graph.number_of_nodes(self._ntypes_invmap[ntype])

    def _number_of_src_nodes(self):
        """Return number of source nodes (only used in scheduler)"""
        return self._graph.number_of_nodes(self._current_srctype_idx)

    def _number_of_dst_nodes(self):
        """Return number of destination nodes (only used in scheduler)"""
        return self._graph.number_of_nodes(self._current_dsttype_idx)

    @property
    def is_multigraph(self):
        """True if the graph is a multigraph, False otherwise.
        """
        assert not self.is_view, 'not supported on views'
        return self._graph.is_multigraph()

    @property
    def is_readonly(self):
        """True if the graph is readonly, False otherwise.
        """
        return self._graph.is_readonly()

    def _number_of_edges(self):
        """Return number of edges in the current view (only used for scheduler)"""
        return self._graph.number_of_edges(self._current_etype_idx)

    def number_of_edges(self, etype):
        """Return the number of edges of the given type in the heterograph.

        Parameters
        ----------
        etype : (str, str, str)
            The source-edge-destination type triplet

        Returns
        -------
        int
            The number of edges

        Examples
        --------
        >>> g.number_of_edges(('user', 'plays', 'game'))
        4
        """
        return self._graph.number_of_edges(self._etypes_invmap[etype])

    def has_node(self, ntype, vid):
        """Return True if the graph contains node `vid` of type `ntype`.

        Parameters
        ----------
        ntype : str
            The node type.
        vid : int
            The node ID.

        Returns
        -------
        bool
            True if the node exists

        Examples
        --------
        >>> g.has_node('user', 0)
        True
        >>> g.has_node('user', 4)
        False

        See Also
        --------
        has_nodes
        """
        return self._graph.has_node(self._ntypes_invmap[ntype], vid)

    def has_nodes(self, ntype, vids):
        """Return a 0-1 array ``a`` given the node ID array ``vids``.

        ``a[i]`` is 1 if the graph contains node ``vids[i]`` of type ``ntype``, 0 otherwise.

        Parameters
        ----------
        ntype : str
            The node type.
        vid : list or tensor
            The array of node IDs.

        Returns
        -------
        a : tensor
            0-1 array indicating existence

        Examples
        --------
        The following example uses PyTorch backend.

        >>> g.has_nodes('user', [0, 1, 2, 3, 4])
        tensor([1, 1, 1, 0, 0])

        See Also
        --------
        has_node
        """
        vids = utils.toindex(vids)
        rst = self._graph.has_nodes(self._ntypes_invmap[ntype], vids)
        return rst.tousertensor()

    def has_edge_between(self, etype, u, v):
        """Return True if the edge (u, v) of type ``etype`` is in the graph.

        Parameters
        ----------
        etype : (str, str, str)
            The source-edge-destination type triplet
        u : int
            The node ID of source type.
        v : int
            The node ID of destination type.

        Returns
        -------
        bool
            True if the edge is in the graph, False otherwise.

        Examples
        --------
        Check whether Alice plays Tetris
        >>> g.has_edge_between(('user', 'plays', 'game'), 0, 1)
        True

        And whether Alice plays Minecraft
        >>> g.has_edge_between(('user', 'plays', 'game'), 0, 2)
        False

        See Also
        --------
        has_edges_between
        """
        return self._graph.has_edge_between(self._etypes_invmap[etype], u, v)

    def has_edges_between(self, etype, u, v):
        """Return a 0-1 array ``a`` given the source node ID array ``u`` and
        destination node ID array ``v``.

        ``a[i]`` is 1 if the graph contains edge ``(u[i], v[i])`` of type ``etype``, 0 otherwise.

        Parameters
        ----------
        etype : (str, str, str)
            The source-edge-destination type triplet
        u : list, tensor
            The node ID array of source type.
        v : list, tensor
            The node ID array of destination type.

        Returns
        -------
        a : tensor
            0-1 array indicating existence.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> g.has_edges_between(('user', 'plays', 'game'), [0, 0], [1, 2])
        tensor([1, 0])

        See Also
        --------
        has_edge_between
        """
        u = utils.toindex(u)
        v = utils.toindex(v)
        rst = self._graph.has_edges_between(self._etypes_invmap[etype], u, v)
        return rst.tousertensor()

    def predecessors(self, etype, v):
        """Return the predecessors of node `v` in the graph with the same
        edge type.

        Node `u` is a predecessor of `v` if an edge `(u, v)` exist in the
        graph.

        Parameters
        ----------
        etype : (str, str, str)
            The source-edge-destination type triplet
        v : int
            The node of destination type.

        Returns
        -------
        tensor
            Array of predecessor node IDs of source node type.

        Examples
        --------
        The following example uses PyTorch backend.

        Query who plays Tetris:
        >>> g.predecessors(('user', 'plays', 'game'), 0)
        tensor([0, 1])

        This indicates User #0 (Alice) and User #1 (Bob).

        See Also
        --------
        successors
        """
        return self._graph.predecessors(self._etypes_invmap[etype], v).tousertensor()

    def successors(self, etype, v):
        """Return the successors of node `v` in the graph with the same edge
        type.

        Node `u` is a successor of `v` if an edge `(v, u)` exist in the
        graph.

        Parameters
        ----------
        etype : (str, str, str)
            The source-edge-destination type triplet
        v : int
            The node of source type.

        Returns
        -------
        tensor
            Array of successor node IDs of destination node type.

        Examples
        --------
        The following example uses PyTorch backend.

        Asks which game Alice plays:
        >>> g.successors(('user', 'plays', 'game'), 0)
        tensor([0])

        This indicates Game #0 (Tetris).

        See Also
        --------
        predecessors
        """
        return self._graph.successors(self._etypes_invmap[etype], v).tousertensor()

    def edge_id(self, etype, u, v, force_multi=False):
        """Return the edge ID, or an array of edge IDs, between source node
        `u` and destination node `v`.

        Only works if the graph has one edge type.  For multiple types,
        query with

        Parameters
        ----------
        etype : (str, str, str)
            The source-edge-destination type triplet
        u : int
            The node ID of source type.
        v : int
            The node ID of destination type.
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

        Find the edge ID of "Bob plays Tetris"
        >>> g.edge_id(('user', 'plays', 'game'), 1, 0)
        1

        See Also
        --------
        edge_ids
        """
        idx = self._graph.edge_id(self._etypes_invmap[etype], u, v)
        return idx.tousertensor() if force_multi or self._graph.is_multigraph() else idx[0]

    def edge_ids(self, etype, u, v, force_multi=False):
        """Return all edge IDs between source node array `u` and destination
        node array `v`.

        Only works if the graph has one edge type.  For multiple types,
        query with

        Parameters
        ----------
        etype : (str, str, str)
            The source-edge-destination type triplet
        u : list, tensor
            The node ID array of source type.
        v : list, tensor
            The node ID array of destination type.
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

        Find the edge IDs of "Alice plays Tetris" and "Bob plays Minecraft".
        >>> g.edge_ids(('user', 'plays', 'game'), [0, 1], [0, 1])
        tensor([0, 2])

        See Also
        --------
        edge_id
        """
        u = utils.toindex(u)
        v = utils.toindex(v)
        src, dst, eid = self._graph.edge_ids(self._etypes_invmap[etype], u, v)
        if force_multi or self._graph.is_multigraph():
            return src.tousertensor(), dst.tousertensor(), eid.tousertensor()
        else:
            return eid.tousertensor()

    def find_edges(self, etype, eid):
        """Given an edge ID array, return the source and destination node ID
        array `s` and `d`.  `s[i]` and `d[i]` are source and destination node
        ID for edge `eid[i]`.

        Parameters
        ----------
        etype : (str, str, str)
            The source-edge-destination type triplet
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

        Find the user and game of gameplay #0 and #2:
        >>> g.find_edges(('user', 'plays', 'game'), [0, 2])
        (tensor([0, 1]), tensor([0, 1]))
        """
        eid = utils.toindex(eid)
        src, dst, _ = self._graph.find_edges(self._etypes_invmap[etype], eid)
        return src.tousertensor(), dst.tousertensor()

    def in_edges(self, etype, v, form='uv'):
        """Return the inbound edges of the node(s).

        Parameters
        ----------
        etype : (str, str, str)
            The source-edge-destination type triplet
        v : int, list, tensor
            The node(s) of destination type.
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

        Find the gameplay IDs of game #0 (Tetris)
        >>> g.in_edges(('user', 'plays', 'game'), 0, 'eid')
        tensor([0, 1])
        """
        v = utils.toindex(v)
        src, dst, eid = self._graph.in_edges(self._etypes_invmap[etype], v)
        if form == 'all':
            return (src.tousertensor(), dst.tousertensor(), eid.tousertensor())
        elif form == 'uv':
            return (src.tousertensor(), dst.tousertensor())
        elif form == 'eid':
            return eid.tousertensor()
        else:
            raise DGLError('Invalid form:', form)

    def out_edges(self, etype, v, form='uv'):
        """Return the outbound edges of the node(s).

        Parameters
        ----------
        etype : (str, str, str)
            The source-edge-destination type triplet
        v : int, list, tensor
            The node(s) of source type.
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

        Find the gameplay IDs of user #0 (Alice)
        >>> g.out_edges(('user', 'plays', 'game'), 0, 'eid')
        tensor([0])
        """
        v = utils.toindex(v)
        src, dst, eid = self._graph.out_edges(self._etypes_invmap[etype], v)
        if form == 'all':
            return (src.tousertensor(), dst.tousertensor(), eid.tousertensor())
        elif form == 'uv':
            return (src.tousertensor(), dst.tousertensor())
        elif form == 'eid':
            return eid.tousertensor()
        else:
            raise DGLError('Invalid form:', form)

    def all_edges(self, etype, form='uv', order=None):
        """Return all the edges.

        Parameters
        ----------
        etype : (str, str, str)
            The source-edge-destination type triplet
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

        Find the user-game pairs for all gameplays:
        >>> g.all_edges(('user', 'plays', 'game'), 'uv')
        (tensor([0, 1, 1, 2]), tensor([0, 0, 1, 1]))
        """
        src, dst, eid = self._graph.edges(self._etypes_invmap[etype], order)
        if form == 'all':
            return (src.tousertensor(), dst.tousertensor(), eid.tousertensor())
        elif form == 'uv':
            return (src.tousertensor(), dst.tousertensor())
        elif form == 'eid':
            return eid.tousertensor()
        else:
            raise DGLError('Invalid form:', form)

    def in_degree(self, etype, v):
        """Return the in-degree of node ``v``.

        Parameters
        ----------
        etype : (str, str, str)
            The source-edge-destination type triplet
        v : int
            The node ID of destination type.

        Returns
        -------
        etype : (str, str, str)
            The source-edge-destination type triplet
        int
            The in-degree.

        Examples
        --------
        Find how many users are playing Game #0 (Tetris):
        >>> g.in_degree(('user', 'plays', 'game'), 0)
        2

        See Also
        --------
        in_degrees
        """
        return self._graph.in_degree(self._etypes_invmap[etype], v)

    def in_degrees(self, etype, v=ALL):
        """Return the array `d` of in-degrees of the node array `v`.

        `d[i]` is the in-degree of node `v[i]`.

        Parameters
        ----------
        etype : (str, str, str)
            The source-edge-destination type triplet
        v : list, tensor, optional.
            The node ID array of destination type. Default is to return the
            degrees of all the nodes.

        Returns
        -------
        d : tensor
            The in-degree array.

        Examples
        --------
        The following example uses PyTorch backend.

        Find how many users are playing Game #0 and #1 (Tetris and Minecraft):
        >>> g.in_degrees(('user', 'plays', 'game'), [0, 1])
        tensor([2, 2])

        See Also
        --------
        in_degree
        """
        etype_idx = self._etypes_invmap[etype]
        _, dsttype_idx = self._endpoint_types(etype_idx)
        if is_all(v):
            v = utils.toindex(slice(0, self._graph.number_of_nodes(dsttype_idx)))
        else:
            v = utils.toindex(v)
        return self._graph.in_degrees(etype_idx, v).tousertensor()

    def out_degree(self, etype, v):
        """Return the out-degree of node `v`.

        Parameters
        ----------
        etype : (str, str, str)
            The source-edge-destination type triplet
        v : int
            The node ID of source type.

        Returns
        -------
        int
            The out-degree.

        Examples
        --------
        Find how many games User #0 Alice is playing
        >>> g.out_degree(('user', 'plays', 'game'), 0)
        1

        See Also
        --------
        out_degrees
        """
        return self._graph.out_degree(self._etypes_invmap[etype], v)

    def out_degrees(self, etype, v=ALL):
        """Return the array `d` of out-degrees of the node array `v`.

        `d[i]` is the out-degree of node `v[i]`.

        Parameters
        ----------
        etype : (str, str, str)
            The source-edge-destination type triplet
        v : list, tensor
            The node ID array of source type. Default is to return the degrees
            of all the nodes.

        Returns
        -------
        d : tensor
            The out-degree array.

        Examples
        --------
        The following example uses PyTorch backend.

        Find how many games User #0 and #1 (Alice and Bob) are playing
        >>> g.out_degrees(('user', 'plays', 'game'), [0, 1])
        tensor([1, 2])

        See Also
        --------
        out_degree
        """
        etype_idx = self._etypes_invmap[etype]
        srctype_idx, _ = self._endpoint_types(etype_idx)
        if is_all(v):
            v = utils.toindex(slice(0, self._graph.number_of_nodes(srctype_idx)))
        else:
            v = utils.toindex(v)
        return self._graph.out_degrees(etype_idx, v).tousertensor()


def bipartite_from_edge_list(u, v, num_src=None, num_dst=None):
    """Create a bipartite graph component of a heterogeneous graph with a
    list of edges.

    Parameters
    ----------
    u, v : list[int]
        List of source and destination node IDs.
    num_src : int, optional
        The number of nodes of source type.

        By default, the value is the maximum of the source node IDs in the
        edge list plus 1.
    num_dst : int, optional
        The number of nodes of destination type.

        By default, the value is the maximum of the destination node IDs in
        the edge list plus 1.
    """
    num_src = num_src or (max(u) + 1)
    num_dst = num_dst or (max(v) + 1)
    u = utils.toindex(u)
    v = utils.toindex(v)
    return heterograph_index.create_bipartite_from_coo(num_src, num_dst, u, v)

def bipartite_from_scipy(spmat, with_edge_id=False):
    """Create a bipartite graph component of a heterogeneous graph with a
    scipy sparse matrix.

    Parameters
    ----------
    spmat : scipy sparse matrix
        The bipartite graph matrix whose rows represent sources and columns
        represent destinations.
    with_edge_id : bool
        If True, the entries in the sparse matrix are treated as edge IDs.
        Otherwise, the entries are ignored and edges will be added in
        (source, destination) order.
    """
    spmat = spmat.tocsr()
    num_src, num_dst = spmat.shape
    indptr = utils.toindex(spmat.indptr)
    indices = utils.toindex(spmat.indices)
    data = utils.toindex(spmat.data if with_edge_id else list(range(len(indices))))
    return heterograph_index.create_bipartite_from_csr(num_src, num_dst, indptr, indices, data)


class DGLHeteroGraph(DGLBaseHeteroGraph):
    """Base heterogeneous graph class.

    A Heterogeneous graph is defined as a graph with node types and edge
    types.

    If two edges share the same edge type, then their source nodes, as well
    as their destination nodes, also have the same type (the source node
    types don't have to be the same as the destination node types).

    Parameters
    ----------
    graph_data :
        The graph data.  It can be one of the followings:

        * (nx.MultiDiGraph, dict[str, list[tuple[int, int]]])
        * (nx.MultiDiGraph, dict[str, scipy.sparse.matrix])

          The first element is the metagraph of the heterogeneous graph, as a
          networkx directed graph.  Its nodes represent the node types, and
          its edges represent the edge types.  The edge type name should be
          stored as edge keys.

          The second element is a mapping from edge type to edge list.  The
          edge list can be either a list of (u, v) pairs, or a scipy sparse
          matrix whose rows represents sources and columns represents
          destinations.  The edges will be added in the (source, destination)
          order.
    node_frames : dict[str, dict[str, Tensor]]
        The node frames for each node type
    edge_frames : dict[str, dict[str, Tensor]]
        The edge frames for each edge type
    multigraph : bool
        Whether the heterogeneous graph is a multigraph.
    readonly : bool
        Whether the heterogeneous graph is readonly.

    Examples
    --------
    Suppose that we want to construct the following heterogeneous graph:

    .. graphviz::

       digraph G {
           Alice -> Bob [label=follows]
           Bob -> Carol [label=follows]
           Alice -> Tetris [label=plays]
           Bob -> Tetris [label=plays]
           Bob -> Minecraft [label=plays]
           Carol -> Minecraft [label=plays]
           Nintendo -> Tetris [label=develops]
           Mojang -> Minecraft [label=develops]
           {rank=source; Alice; Bob; Carol}
           {rank=sink; Nintendo; Mojang}
       }

    One can analyze the graph and figure out the metagraph as follows:

    .. graphviz::

       digraph G {
           User -> User [label=follows]
           User -> Game [label=plays]
           Developer -> Game [label=develops]
       }

    Suppose that one maps the users, games and developers to the following
    IDs:

        User name   Alice   Bob     Carol
        User ID     0       1       2

        Game name   Tetris  Minecraft
        Game ID     0       1

        Developer name  Nintendo    Mojang
        Developer ID    0           1

    One can construct the graph as follows:

    >>> mg = nx.MultiDiGraph([('user', 'user', 'follows'),
    ...                       ('user', 'game', 'plays'),
    ...                       ('developer', 'game', 'develops')])
    >>> g = DGLHeteroGraph(
    ...         mg, {
    ...             'follows': [(0, 1), (1, 2)],
    ...             'plays': [(0, 0), (1, 0), (1, 1), (2, 1)],
    ...             'develops': [(0, 0), (1, 1)]})

    Then one can query the graph structure as follows:

    >>> g['user'].number_of_nodes()
    3
    >>> g['plays'].number_of_edges()
    4
    >>> g['develops'].out_degrees() # out-degrees of source nodes of 'develops' relation
    tensor([1, 1])
    >>> g['develops'].in_edges(0)   # in-edges of destination node 0 of 'develops' relation
    (tensor([0]), tensor([0]))

    Notes
    -----
    Currently, all heterogeneous graphs are readonly.
    """
    # pylint: disable=unused-argument
    def __init__(
            self,
            graph_data=None,
            node_frames=None,
            edge_frames=None,
            multigraph=None,
            readonly=True,
            _view_ntype_idx=None,
            _view_etype_idx=None):
        assert readonly, "Only readonly heterogeneous graphs are supported"

        # Creating a view of another graph?
        if isinstance(graph_data, DGLHeteroGraph):
            super(DGLHeteroGraph, self).__init__(
                graph_data._graph, graph_data._ntypes, graph_data._etypes,
                graph_data._ntypes_invmap, graph_data._etypes_invmap,
                graph_data._view_ntype_idx, graph_data._view_etype_idx)
            self._node_frames = graph_data._node_frames
            self._edge_frames = graph_data._edge_frames
            self._msg_frames = graph_data._msg_frames
            self._msg_indices = graph_data._msg_indices
            self._view_ntype_idx = _view_ntype_idx
            self._view_etype_idx = _view_etype_idx
            return

        if isinstance(graph_data, tuple):
            metagraph, edges_by_type = graph_data
            if not isinstance(metagraph, nx.MultiDiGraph):
                raise TypeError('Metagraph should be networkx.MultiDiGraph')

            # create metagraph graph index
            srctypes, dsttypes, etypes = [], [], []
            ntypes = []
            ntypes_invmap = {}
            etypes_invmap = {}
            for srctype, dsttype, etype in metagraph.edges(keys=True):
                srctypes.append(srctype)
                dsttypes.append(dsttype)
                etypes_invmap[(srctype, etype, dsttype)] = len(etypes_invmap)
                etypes.append((srctype, etype, dsttype))

                if srctype not in ntypes_invmap:
                    ntypes_invmap[srctype] = len(ntypes_invmap)
                    ntypes.append(srctype)
                if dsttype not in ntypes_invmap:
                    ntypes_invmap[dsttype] = len(ntypes_invmap)
                    ntypes.append(dsttype)

            srctypes = [ntypes_invmap[srctype] for srctype in srctypes]
            dsttypes = [ntypes_invmap[dsttype] for dsttype in dsttypes]

            metagraph_index = graph_index.create_graph_index(
                list(zip(srctypes, dsttypes)), None, True)  # metagraph is always immutable

            # create base bipartites
            bipartites = []
            num_nodes = defaultdict(int)
            # count the number of nodes for each type
            for etype_triplet in etypes:
                srctype, etype, dsttype = etype_triplet
                edges = edges_by_type[etype_triplet]
                if ssp.issparse(edges):
                    num_src, num_dst = edges.shape
                elif isinstance(edges, list):
                    u, v = zip(*edges)
                    num_src = max(u) + 1
                    num_dst = max(v) + 1
                else:
                    raise TypeError('unknown edge list type %s' % type(edges))
                num_nodes[srctype] = max(num_nodes[srctype], num_src)
                num_nodes[dsttype] = max(num_nodes[dsttype], num_dst)
            # create actual objects
            for etype_triplet in etypes:
                srctype, etype, dsttype = etype_triplet
                edges = edges_by_type[etype_triplet]
                if ssp.issparse(edges):
                    bipartite = bipartite_from_scipy(edges)
                elif isinstance(edges, list):
                    u, v = zip(*edges)
                    bipartite = bipartite_from_edge_list(
                        u, v, num_nodes[srctype], num_nodes[dsttype])
                bipartites.append(bipartite)

            hg_index = heterograph_index.create_heterograph(metagraph_index, bipartites)

            super(DGLHeteroGraph, self).__init__(hg_index, ntypes, etypes)
        else:
            raise TypeError('Unrecognized graph data type %s' % type(graph_data))

        # node and edge frame
        if node_frames is None:
            self._node_frames = [
                FrameRef(Frame(num_rows=self._graph.number_of_nodes(i)))
                for i in range(len(self._ntypes))]
        else:
            self._node_frames = node_frames

        if edge_frames is None:
            self._edge_frames = [
                FrameRef(Frame(num_rows=self._graph.number_of_edges(i)))
                for i in range(len(self._etypes))]
        else:
            self._edge_frames = edge_frames

        # message indicators
        self._msg_indices = [None] * len(self._etypes)
        self._msg_frames = []
        for i in range(len(self._etypes)):
            frame = FrameRef(Frame(num_rows=self._graph.number_of_edges(i)))
            frame.set_initializer(init.zero_initializer)
            self._msg_frames.append(frame)

    def _create_view(self, ntype_idx, etype_idx):
        return DGLHeteroGraph(
            graph_data=self, _view_ntype_idx=ntype_idx, _view_etype_idx=etype_idx)

    def _get_msg_index(self):
        if self._msg_indices[self._current_etype_idx] is None:
            self._msg_indices[self._current_etype_idx] = utils.zero_index(
                size=self._graph.number_of_edges(self._current_etype_idx))
        return self._msg_indices[self._current_etype_idx]

    def _set_msg_index(self, index):
        self._msg_indices[self._current_etype_idx] = index

    def __getitem__(self, key):
        if key in self._etypes_invmap:
            return self._create_view(None, self._etypes_invmap[key])
        else:
            raise KeyError(key)

    @property
    def _node_frame(self):
        # overrides DGLGraph._node_frame
        return self._node_frames[self._current_ntype_idx]

    @property
    def _edge_frame(self):
        # overrides DGLGraph._edge_frame
        return self._edge_frames[self._current_etype_idx]

    @property
    def _src_frame(self):
        # overrides DGLGraph._src_frame
        return self._node_frames[self._current_srctype_idx]

    @property
    def _dst_frame(self):
        # overrides DGLGraph._dst_frame
        return self._node_frames[self._current_dsttype_idx]

    @property
    def _msg_frame(self):
        # overrides DGLGraph._msg_frame
        return self._msg_frames[self._current_etype_idx]

    def add_nodes(self, node_type, num, data=None):
        """Add multiple new nodes of the same node type

        Parameters
        ----------
        node_type : str
            Type of the added nodes.  Must appear in the metagraph.
        num : int
            Number of nodes to be added.
        data : dict, optional
            Feature data of the added nodes.

        Examples
        --------
        The variable ``g`` is constructed from the example in
        DGLBaseHeteroGraph.

        >>> g['game'].number_of_nodes()
        2
        >>> g.add_nodes(3, 'game')  # add 3 new games
        >>> g['game'].number_of_nodes()
        5
        """
        pass

    def add_edge(self, etype, u, v, data=None):
        """Add an edge of ``etype`` between u of the source node type, and v
        of the destination node type..

        Parameters
        ----------
        etype : (str, str, str)
            The source-edge-destination type triplet
        u : int
            The source node ID of type ``utype``.  Must exist in the graph.
        v : int
            The destination node ID of type ``vtype``.  Must exist in the
            graph.
        data : dict, optional
            Feature data of the added edge.

        Examples
        --------
        The variable ``g`` is constructed from the example in
        DGLBaseHeteroGraph.

        >>> g['plays'].number_of_edges()
        4
        >>> g.add_edge(2, 0, 'plays')
        >>> g['plays'].number_of_edges()
        5
        """
        pass

    def add_edges(self, u, v, etype, data=None):
        """Add multiple edges of ``etype`` between list of source nodes ``u``
        and list of destination nodes ``v`` of type ``vtype``.  A single edge
        is added between every pair of ``u[i]`` and ``v[i]``.

        Parameters
        ----------
        u : list, tensor
            The source node IDs of type ``utype``.  Must exist in the graph.
        v : list, tensor
            The destination node IDs of type ``vtype``.  Must exist in the
            graph.
        etype : (str, str, str)
            The source-edge-destination type triplet
        data : dict, optional
            Feature data of the added edge.

        Examples
        --------
        The variable ``g`` is constructed from the example in
        DGLBaseHeteroGraph.

        >>> g['plays'].number_of_edges()
        4
        >>> g.add_edges([0, 2], [1, 0], 'plays')
        >>> g['plays'].number_of_edges()
        6
        """
        pass

    def from_networkx(
            self,
            nx_graph,
            node_type_attr_name='type',
            edge_type_attr_name='type',
            node_id_attr_name='id',
            edge_id_attr_name='id',
            node_attrs=None,
            edge_attrs=None):
        """Convert from networkx graph.

        The networkx graph must satisfy the metagraph.  That is, for any
        edge in the networkx graph, the source/destination node type must
        be the same as the source/destination node of the edge type in
        the metagraph.  An error will be raised otherwise.

        Parameters
        ----------
        nx_graph : networkx.DiGraph
            The networkx graph.
        node_type_attr_name : str
            The node attribute name for the node type.
            The attribute contents must be strings.
        edge_type_attr_name : str
            The edge attribute name for the edge type.
            The attribute contents must be strings.
        node_id_attr_name : str
            The node attribute name for node type-specific IDs.
            The attribute contents must be integers.
            If the IDs of the same type are not consecutive integers, its
            nodes will be relabeled using consecutive integers.  The new
            node ordering will inherit that of the sorted IDs.
        edge_id_attr_name : str or None
            The edge attribute name for edge type-specific IDs.
            The attribute contents must be integers.
            If the IDs of the same type are not consecutive integers, its
            nodes will be relabeled using consecutive integers.  The new
            node ordering will inherit that of the sorted IDs.

            If None is provided, the edge order would be arbitrary.
        node_attrs : iterable of str, optional
            The node attributes whose data would be copied.
        edge_attrs : iterable of str, optional
            The edge attributes whose data would be copied.
        """
        pass

    def node_attr_schemes(self, ntype):
        """Return the node feature schemes.

        Each feature scheme is a named tuple that stores the shape and data type
        of the node feature.

        Parameters
        ----------
        ntype : str
            The node type

        Returns
        -------
        dict of str to schemes
            The schemes of node feature columns.

        Examples
        --------
        The following uses PyTorch backend.

        >>> g.ndata['user']['h'] = torch.randn(3, 4)
        >>> g.node_attr_schemes('user')
        {'h': Scheme(shape=(4,), dtype=torch.float32)}
        """
        return self._node_frames[self._ntypes_invmap[ntype]].schemes

    def edge_attr_schemes(self, etype):
        """Return the edge feature schemes.

        Each feature scheme is a named tuple that stores the shape and data type
        of the edge feature.

        Parameters
        ----------
        etype : (str, str, str)
            The source-edge-destination type triplet

        Returns
        -------
        dict of str to schemes
            The schemes of node feature columns.

        Examples
        --------
        The following uses PyTorch backend.

        >>> g.edata['user', 'plays', 'game']['h'] = torch.randn(4, 4)
        >>> g.edge_attr_schemes(('user', 'plays', 'game'))
        {'h': Scheme(shape=(4,), dtype=torch.float32)}
        """
        return self._edge_frames[self._etypes_invmap[etype]].schemes

    @property
    def nodes(self):
        """Return a node view that can used to set/get feature data of a
        single node type.

        Examples
        --------
        To set features of User #0 and #2 in a heterogeneous graph:
        >>> g.nodes['user'][[0, 2]].data['h'] = torch.zeros(2, 5)
        """
        return HeteroNodeView(self)

    @property
    def ndata(self):
        """Return the data view of all the nodes of a single node type.

        Examples
        --------
        To set features of games in a heterogeneous graph:
        >>> g.ndata['game']['h'] = torch.zeros(2, 5)
        """
        return HeteroNodeDataView(self)

    @property
    def edges(self):
        """Return an edges view that can used to set/get feature data of a
        single edge type.

        Examples
        --------
        To set features of gameplays #1 (Bob -> Tetris) and #3 (Carol ->
        Minecraft) in a heterogeneous graph:
        >>> g.edges['user', 'plays', 'game'][[1, 3]].data['h'] = torch.zeros(2, 5)
        """
        return HeteroEdgeView(self)

    @property
    def edata(self):
        """Return the data view of all the edges of a single edge type.

        Examples
        --------
        >>> g.edata['developer', 'develops', 'game']['h'] = torch.zeros(2, 5)
        """
        return HeteroEdgeDataView(self)

    def set_n_repr(self, ntype, data, u=ALL, inplace=False):
        """Set node(s) representation of a single node type.

        `data` is a dictionary from the feature name to feature tensor. Each tensor
        is of shape (B, D1, D2, ...), where B is the number of nodes to be updated,
        and (D1, D2, ...) be the shape of the node representation tensor. The
        length of the given node ids must match B (i.e, len(u) == B).

        All update will be done out of place to work with autograd unless the
        inplace flag is true.

        Parameters
        ----------
        ntype : str
            The node type
        data : dict of tensor
            Node representation.
        u : node, container or tensor
            The node(s).
        inplace : bool
            If True, update will be done in place, but autograd will break.
        """
        ntype = self._ntypes_invmap[ntype]
        if is_all(u):
            num_nodes = self._graph.number_of_nodes(ntype)
        else:
            u = utils.toindex(u)
            num_nodes = len(u)
        for key, val in data.items():
            nfeats = F.shape(val)[0]
            if nfeats != num_nodes:
                raise DGLError('Expect number of features to match number of nodes (len(u)).'
                               ' Got %d and %d instead.' % (nfeats, num_nodes))

        if is_all(u):
            for key, val in data.items():
                self._node_frames[ntype][key] = val
        else:
            self._node_frames[ntype].update_rows(u, data, inplace=inplace)

    def get_n_repr(self, ntype, u=ALL):
        """Get node(s) representation of a single node type.

        The returned feature tensor batches multiple node features on the first dimension.

        Parameters
        ----------
        ntype : str
            The node type
        u : node, container or tensor
            The node(s).

        Returns
        -------
        dict
            Representation dict from feature name to feature tensor.
        """
        if len(self.node_attr_schemes(ntype)) == 0:
            return dict()
        ntype_idx = self._ntypes_invmap[ntype]
        if is_all(u):
            return dict(self._node_frames[ntype_idx])
        else:
            u = utils.toindex(u)
            return self._node_frames[ntype_idx].select_rows(u)

    def pop_n_repr(self, ntype, key):
        """Get and remove the specified node repr of a given node type.

        Parameters
        ----------
        ntype : str
            The node type
        key : str
            The attribute name.

        Returns
        -------
        Tensor
            The popped representation
        """
        ntype = self._ntypes_invmap[ntype]
        return self._node_frames[ntype].pop(key)

    def set_e_repr(self, etype, data, edges=ALL, inplace=False):
        """Set edge(s) representation of a single edge type.

        `data` is a dictionary from the feature name to feature tensor. Each tensor
        is of shape (B, D1, D2, ...), where B is the number of edges to be updated,
        and (D1, D2, ...) be the shape of the edge representation tensor.

        All update will be done out of place to work with autograd unless the
        inplace flag is true.

        Parameters
        ----------
        etype : (str, str, str)
            The source-edge-destination type triplet
        data : tensor or dict of tensor
            Edge representation.
        edges : edges
            Edges can be either

            * A pair of endpoint nodes (u, v), where u is the node ID of source
              node type and v is that of destination node type.
            * A tensor of edge ids of the given type.

            The default value is all the edges.
        inplace : bool
            If True, update will be done in place, but autograd will break.
        """
        etype_idx = self._etypes_invmap[etype]
        # parse argument
        if is_all(edges):
            eid = ALL
        elif isinstance(edges, tuple):
            u, v = edges
            u = utils.toindex(u)
            v = utils.toindex(v)
            # Rewrite u, v to handle edge broadcasting and multigraph.
            _, _, eid = self._graph.edge_ids(etype_idx, u, v)
        else:
            eid = utils.toindex(edges)

        # sanity check
        if not utils.is_dict_like(data):
            raise DGLError('Expect dictionary type for feature data.'
                           ' Got "%s" instead.' % type(data))

        if is_all(eid):
            num_edges = self._graph.number_of_edges(etype_idx)
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
                self._edge_frames[etype_idx][key] = val
        else:
            # update row
            self._edge_frames[etype_idx].update_rows(eid, data, inplace=inplace)

    def get_e_repr(self, etype, edges=ALL):
        """Get edge(s) representation.

        Parameters
        ----------
        etype : (str, str, str)
            The source-edge-destination type triplet
        edges : edges
            Edges can be a pair of endpoint nodes (u, v), or a
            tensor of edge ids. The default value is all the edges.

        Returns
        -------
        dict
            Representation dict
        """
        etype_idx = self._etypes_invmap[etype]
        if len(self.edge_attr_schemes(etype)) == 0:
            return dict()
        # parse argument
        if is_all(edges):
            eid = ALL
        elif isinstance(edges, tuple):
            u, v = edges
            u = utils.toindex(u)
            v = utils.toindex(v)
            # Rewrite u, v to handle edge broadcasting and multigraph.
            _, _, eid = self._graph.edge_ids(etype_idx, u, v)
        else:
            eid = utils.toindex(edges)

        if is_all(eid):
            return dict(self._edge_frames[etype_idx])
        else:
            eid = utils.toindex(eid)
            return self._edge_frames[etype_idx].select_rows(eid)

    def pop_e_repr(self, etype, key):
        """Get and remove the specified edge repr of a single edge type.

        Parameters
        ----------
        etype : (str, str, str)
            The source-edge-destination type triplet
        key : str
          The attribute name.

        Returns
        -------
        Tensor
            The popped representation
        """
        etype = self._etypes_invmap[etype]
        self._edge_frames[etype].pop(key)

    def register_message_func(self, func):
        """Register global message function for each edge type provided.

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
        raise NotImplementedError

    def register_reduce_func(self, func):
        """Register global message reduce function for each edge type provided.

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
        raise NotImplementedError

    def register_apply_node_func(self, func):
        """Register global node apply function for each node type provided.

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
        raise NotImplementedError

    def register_apply_edge_func(self, func):
        """Register global edge apply function for each edge type provided.

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
        raise NotImplementedError

    def apply_nodes(self, func, v=ALL, inplace=False):
        """Apply the function on the nodes with the same type to update their
        features.

        If None is provided for ``func``, nothing will happen.

        Parameters
        ----------
        func : dict[str, callable] or None
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        v : dict[str, int or iterable of int or tensor], optional
            The (type-specific) node (ids) on which to apply ``func``.
        inplace : bool, optional
            If True, update will be done in place, but autograd will break.

        Examples
        --------
        >>> g.ndata['user']['h'] = torch.ones(3, 5)
        >>> g.apply_nodes({'user': lambda nodes: {'h': nodes.data['h'] * 2}})
        >>> g.ndata['user']['h']
        tensor([[2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.]])
        """
        for ntype, nfunc in func.items():
            if is_all(v):
                v_ntype = utils.toindex(slice(0, self.number_of_nodes(ntype)))
            else:
                v_ntype = utils.toindex(v[ntype])
            with ir.prog() as prog:
                scheduler.schedule_apply_nodes(
                    graph=self._create_view(self._ntypes_invmap[ntype], None),
                    v=v_ntype,
                    apply_func=nfunc,
                    inplace=inplace)
                Runtime.run(prog)

    def apply_edges(self, func, edges=ALL, inplace=False):
        """Apply the function on the edges with the same type to update their
        features.

        If None is provided for ``func``, nothing will happen.

        Parameters
        ----------
        func : dict[(str, str, str), callable] or None
            Apply function on the edge. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        edges : dict[(str, str, str), any valid edge specification], optional
            Edges on which to apply ``func``. See :func:`send` for valid
            edge specification.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.

        Examples
        --------
        >>> g.edata['user', 'plays', 'game']['h'] = torch.ones(4, 5)
        >>> g.apply_edges(
        ...     {('user', 'plays', 'game'): lambda edges: {'h': edges.data['h'] * 2}})
        >>> g.edata['user', 'plays', 'game']['h']
        tensor([[2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.]])
        """
        for etype, efunc in func.items():
            etype_idx = self._etypes_invmap[etype]
            if is_all(edges):
                u, v, _ = self._graph.edges(etype_idx, 'eid')
                eid = utils.toindex(slice(0, self.number_of_edges(etype)))
            elif isinstance(edges, tuple):
                u, v = edges
                u = utils.toindex(u)
                v = utils.toindex(v)
                # Rewrite u, v to handle edge broadcasting and multigraph.
                u, v, eid = self._graph.edge_ids(etype_idx, u, v)
            else:
                eid = utils.toindex(edges)
                u, v, _ = self._graph.find_edges(etype_idx, eid)

            with ir.prog() as prog:
                scheduler.schedule_apply_edges(
                    graph=self._create_view(None, etype_idx),
                    u=u,
                    v=v,
                    eid=eid,
                    apply_func=efunc,
                    inplace=inplace)
                Runtime.run(prog)

    def group_apply_edges(self, group_by, func, edges=ALL, inplace=False):
        """Group the edges by nodes and apply the function of the grouped
        edges to update their features.  The edges are of the same edge type
        (hence having the same source and destination node type).

        Parameters
        ----------
        group_by : str
            Specify how to group edges. Expected to be either 'src' or 'dst'
        func : dict[(str, str, str), callable]
            Apply function on the edge.  The function should be
            an :mod:`Edge UDF <dgl.udf>`. The input of `Edge UDF` should
            be (bucket_size, degrees, *feature_shape), and
            return the dict with values of the same shapes.
        edges : dict[(str, str, str), valid edges type], optional
            Edges on which to group and apply ``func``. See :func:`send` for valid
            edges type. Default is all the edges.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.
        """
        if group_by not in ('src', 'dst'):
            raise DGLError("Group_by should be either src or dst")

        for etype, efunc in func.items():
            etype_idx = self._etypes_invmap[etype]
            if is_all(edges):
                u, v, _ = self._graph.edges(etype_idx)
                eid = utils.toindex(slice(0, self.number_of_edges(etype)))
            elif isinstance(edges, tuple):
                u, v = edges
                u = utils.toindex(u)
                v = utils.toindex(v)
                # Rewrite u, v to handle edge broadcasting and multigraph.
                u, v, eid = self._graph.edge_ids(etype_idx, u, v)
            else:
                eid = utils.toindex(edges)
                u, v, _ = self._graph.find_edges(etype_idx, eid)

            with ir.prog() as prog:
                scheduler.schedule_group_apply_edge(
                    graph=self._create_view(None, etype_idx),
                    u=u,
                    v=v,
                    eid=eid,
                    apply_func=efunc,
                    group_by=group_by,
                    inplace=inplace)
                Runtime.run(prog)

    def send(self, edges=ALL, message_func=None):
        """Send messages along the given edges with the same edge type.

        ``edges`` can be any of the following types:

        * ``int`` : Specify one edge using its edge id (of the given edge type).
        * ``pair of int`` : Specify one edge using its endpoints (of source node type
          and destination node type respectively).
        * ``int iterable`` / ``tensor`` : Specify multiple edges using their edge ids.
        * ``pair of int iterable`` / ``pair of tensors`` :
          Specify multiple edges using their endpoints.

        Only works if the graph has one edge type.  For multiple types,
        use

        .. code::

           g['edgetype'].send(edges, message_func)

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
        """
        assert not utils.is_dict_like(message_func), \
            "multiple-type message passing is not implemented"
        assert message_func is not None

        if is_all(edges):
            eid = utils.toindex(slice(0, self._graph.number_of_edges(self._current_etype_idx)))
            u, v, _ = self._graph.edges(self._current_etype_idx)
        elif isinstance(edges, tuple):
            u, v = edges
            u = utils.toindex(u)
            v = utils.toindex(v)
            # Rewrite u, v to handle edge broadcasting and multigraph.
            u, v, eid = self._graph.edge_ids(self._current_etype_idx, u, v)
        else:
            eid = utils.toindex(edges)
            u, v, _ = self._graph.find_edges(self._current_etype_idx, eid)

        if len(eid) == 0:
            # no edge to be triggered
            return

        with ir.prog() as prog:
            scheduler.schedule_send(graph=self, u=u, v=v, eid=eid,
                                    message_func=message_func)
            Runtime.run(prog)

    def recv(self,
             v=ALL,
             reduce_func=None,
             apply_node_func=None,
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

        Only works if the graph has one edge type.  For multiple types,
        use

        .. code::

           g['edgetype'].recv(v, reduce_func, apply_node_func, inplace)

        Parameters
        ----------
        v : int, container or tensor, optional
            The node(s) to be updated. Default is receiving all the nodes.
        reduce_func : callable, optional
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        apply_node_func : callable
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.
        """
        assert not utils.is_dict_like(reduce_func) and \
            not utils.is_dict_like(apply_node_func), \
            "multiple-type message passing is not implemented"
        assert reduce_func is not None

        if is_all(v):
            v = F.arange(0, self._graph.number_of_nodes(self._current_dsttype_idx))
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

    def send_and_recv(self,
                      edges,
                      message_func=None,
                      reduce_func=None,
                      apply_node_func=None,
                      inplace=False):
        """Send messages along edges with the same edge type, and let destinations
        receive them.

        Optionally, apply a function to update the node features after receive.

        This is a convenient combination for performing
        ``send(self, self.edges, message_func)`` and
        ``recv(self, dst, reduce_func, apply_node_func)``, where ``dst``
        are the destinations of the ``edges``.

        Only works if the graph has one edge type.  For multiple types,
        use

        .. code::

           g['edgetype'].send_and_recv(edges, message_func, reduce_func, apply_node_func, inplace)

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
        """
        assert not utils.is_dict_like(message_func) and \
            not utils.is_dict_like(reduce_func) and \
            not utils.is_dict_like(apply_node_func), \
            "multiple-type message passing is not implemented"
        assert message_func is not None
        assert reduce_func is not None

        if isinstance(edges, tuple):
            u, v = edges
            u = utils.toindex(u)
            v = utils.toindex(v)
            # Rewrite u, v to handle edge broadcasting and multigraph.
            u, v, eid = self._graph.edge_ids(self._current_etype_idx, u, v)
        else:
            eid = utils.toindex(edges)
            u, v, _ = self._graph.find_edges(self._current_etype_idx, eid)

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
             message_func=None,
             reduce_func=None,
             apply_node_func=None,
             inplace=False):
        """Pull messages from the node(s)' predecessors and then update their features.

        Optionally, apply a function to update the node features after receive.

        * `reduce_func` will be skipped for nodes with no incoming message.
        * If all ``v`` have no incoming message, this will downgrade to an :func:`apply_nodes`.
        * If some ``v`` have no incoming message, their new feature value will be calculated
          by the column initializer (see :func:`set_n_initializer`). The feature shapes and
          dtypes will be inferred.

        Only works if the graph has one edge type.  For multiple types,
        use

        .. code::

           g['edgetype'].pull(v, message_func, reduce_func, apply_node_func, inplace)

        Parameters
        ----------
        v : int, container or tensor, optional
            The node(s) to be updated. Default is receiving all the nodes.
        message_func : callable, optional
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        reduce_func : callable, optional
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        apply_node_func : callable, optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        """
        assert not utils.is_dict_like(message_func) and \
            not utils.is_dict_like(reduce_func) and \
            not utils.is_dict_like(apply_node_func), \
            "multiple-type message passing is not implemented"
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
             message_func=None,
             reduce_func=None,
             apply_node_func=None,
             inplace=False):
        """Send message from the node(s) to their successors and update them.

        Optionally, apply a function to update the node features after receive.

        Only works if the graph has one edge type.  For multiple types,
        use

        .. code::

           g['edgetype'].push(e, message_func, reduce_func, apply_node_func, inplace)

        Parameters
        ----------
        u : int, container or tensor
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
        """
        assert not utils.is_dict_like(message_func) and \
            not utils.is_dict_like(reduce_func) and \
            not utils.is_dict_like(apply_node_func), \
            "multiple-type message passing is not implemented"
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
                   message_func=None,
                   reduce_func=None,
                   apply_node_func=None):
        """Send messages through all edges and update all nodes.

        Optionally, apply a function to update the node features after receive.

        This is a convenient combination for performing
        ``send(self, self.edges(), message_func)`` and
        ``recv(self, self.nodes(), reduce_func, apply_node_func)``.

        Only works if the graph has one edge type.  For multiple types,
        use

        .. code::

           g['edgetype'].update_all(message_func, reduce_func, apply_node_func)

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
        """
        assert not utils.is_dict_like(message_func) and \
            not utils.is_dict_like(reduce_func) and \
            not utils.is_dict_like(apply_node_func), \
            "multiple-type message passing is not implemented"
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
                   message_func=None,
                   reduce_func=None,
                   apply_node_func=None):
        """Node propagation in heterogeneous graph is not supported.
        """
        raise NotImplementedError('not supported')

    def prop_edges(self,
                   edges_generator,
                   message_func=None,
                   reduce_func=None,
                   apply_node_func=None):
        """Edge propagation in heterogeneous graph is not supported.
        """
        raise NotImplementedError('not supported')

    def subgraph(self, nodes):
        """Return the subgraph induced on given nodes.

        Parameters
        ----------
        nodes : dict[str, list or iterable]
            A dictionary of node types to node ID array to construct
            subgraph.
            All nodes must exist in the graph.

        Returns
        -------
        G : DGLHeteroSubGraph
            The subgraph.
            The nodes are relabeled so that node `i` of type `t` in the
            subgraph is mapped to the ``nodes[i]`` of type `t` in the
            original graph.
            The edges are also relabeled.
            One can retrieve the mapping from subgraph node/edge ID to parent
            node/edge ID via `parent_nid` and `parent_eid` properties of the
            subgraph.
        """
        pass

    def subgraphs(self, nodes):
        """Return a list of subgraphs, each induced in the corresponding given
        nodes in the list.

        Equivalent to
        ``[self.subgraph(nodes_list) for nodes_list in nodes]``

        Parameters
        ----------
        nodes : a list of dict[str, list or iterable]
            A list of type-ID dictionaries to construct corresponding
            subgraphs.  The dictionaries are of the same form as
            :func:`subgraph`.
            All nodes in all the list items must exist in the graph.

        Returns
        -------
        G : A list of DGLHeteroSubGraph
            The subgraphs.
        """
        pass

    def edge_subgraph(self, edges):
        """Return the subgraph induced on given edges.

        Parameters
        ----------
        edges : dict[etype, list or iterable]
            A dictionary of edge types to edge ID array to construct
            subgraph.
            All edges must exist in the subgraph.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.

        Returns
        -------
        G : DGLHeteroSubGraph
            The subgraph.
            The edges are relabeled so that edge `i` of type `t` in the
            subgraph is mapped to the ``edges[i]`` of type `t` in the
            original graph.
            One can retrieve the mapping from subgraph node/edge ID to parent
            node/edge ID via `parent_nid` and `parent_eid` properties of the
            subgraph.
        """
        pass

    def adjacency_matrix_scipy(self, etype, transpose=False, fmt='csr'):
        """Return the scipy adjacency matrix representation of edges with the
        given edge type.

        By default, a row of returned adjacency matrix represents the destination
        of an edge and the column represents the source.

        When transpose is True, a row represents the source and a column represents
        a destination.

        The elements in the adajency matrix are edge ids.

        Parameters
        ----------
        etype : tuple[str, str, str]
            The edge type, characterized by a triplet of source type name,
            destination type name, and edge type name.
        transpose : bool, optional (default=False)
            A flag to transpose the returned adjacency matrix.
        fmt : str, optional (default='csr')
            Indicates the format of returned adjacency matrix.

        Returns
        -------
        scipy.sparse.spmatrix
            The scipy representation of adjacency matrix.
        """
        pass

    def adjacency_matrix(self, etype, transpose=False, ctx=F.cpu()):
        """Return the adjacency matrix representation of edges with the
        given edge type.

        By default, a row of returned adjacency matrix represents the
        destination of an edge and the column represents the source.

        When transpose is True, a row represents the source and a column
        represents a destination.

        Parameters
        ----------
        etype : tuple[str, str, str]
            The edge type, characterized by a triplet of source type name,
            destination type name, and edge type name.
        transpose : bool, optional (default=False)
            A flag to transpose the returned adjacency matrix.
        ctx : context, optional (default=cpu)
            The context of returned adjacency matrix.

        Returns
        -------
        SparseTensor
            The adjacency matrix.
        """
        pass

    def incidence_matrix(self, etype, typestr, ctx=F.cpu()):
        """Return the incidence matrix representation of edges with the given
        edge type.

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
        etype : tuple[str, str, str]
            The edge type, characterized by a triplet of source type name,
            destination type name, and edge type name.
        typestr : str
            Can be either ``in``, ``out`` or ``both``
        ctx : context, optional (default=cpu)
            The context of returned incidence matrix.

        Returns
        -------
        SparseTensor
            The incidence matrix.
        """
        pass

    def filter_nodes(self, ntype, predicate, nodes=ALL):
        """Return a tensor of node IDs with the given node type that satisfy
        the given predicate.

        Parameters
        ----------
        ntype : str
            The node type.
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
        """
        pass

    def filter_edges(self, etype, predicate, edges=ALL):
        """Return a tensor of edge IDs with the given edge type that satisfy
        the given predicate.

        Parameters
        ----------
        etype : tuple[str, str, str]
            The edge type, characterized by a triplet of source type name,
            destination type name, and edge type name.
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
        """
        pass

    def readonly(self, readonly_state=True):
        """Set this graph's readonly state in-place.

        Parameters
        ----------
        readonly_state : bool, optional
            New readonly state of the graph, defaults to True.
        """
        pass

    # TODO: replace this after implementing frame
    # pylint: disable=useless-super-delegation
    def __repr__(self):
        return super(DGLHeteroGraph, self).__repr__()

# pylint: disable=abstract-method
class DGLHeteroSubGraph(DGLHeteroGraph):
    """
    Parameters
    ----------
    parent : DGLHeteroGraph
        The parent graph.
    parent_nid : dict[str, utils.Index]
        The type-specific parent node IDs for each type.
    parent_eid : dict[etype, utils.Index]
        The type-specific parent edge IDs for each type.
    graph_idx : GraphIndex
        The graph index
    shared : bool, optional
        Whether the subgraph shares node/edge features with the parent graph
    """
    # pylint: disable=unused-argument, super-init-not-called
    def __init__(
            self,
            parent,
            parent_nid,
            parent_eid,
            graph_idx,
            shared=False):
        pass

    @property
    def parent_nid(self):
        """Get the parent node ids.

        The returned tensor dictionary can be used as a map from the node id
        in this subgraph to the node id in the parent graph.

        Returns
        -------
        dict[str, Tensor]
            The parent node id array for each type.
        """
        pass

    @property
    def parent_eid(self):
        """Get the parent edge ids.

        The returned tensor dictionary can be used as a map from the edge id
        in this subgraph to the edge id in the parent graph.

        Returns
        -------
        dict[etype, Tensor]
            The parent edge id array for each type.
            The edge types are characterized by a triplet of source type
            name, destination type name, and edge type name.
        """
        pass

    def copy_to_parent(self, inplace=False):
        """Write node/edge features to the parent graph.

        Parameters
        ----------
        inplace : bool
            If true, use inplace write (no gradient but faster)
        """
        pass

    def copy_from_parent(self):
        """Copy node/edge features from the parent graph.

        All old features will be removed.
        """
        pass

    def map_to_subgraph_nid(self, parent_vids):
        """Map the node IDs in the parent graph to the node IDs in the
        subgraph.

        Parameters
        ----------
        parent_vids : dict[str, list or tensor]
            The dictionary of node types to parent node ID array.

        Returns
        -------
        dict[str, tensor]
            The node ID array in the subgraph of each node type.
        """
        pass
