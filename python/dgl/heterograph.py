"""Classes for heterogeneous graphs."""
import networkx as nx
from . import heterograph_index, graph_index
from . import utils
from . import backend as F
from . import init
from .runtime import ir, scheduler, Runtime
from .frame import Frame, FrameRef
from .view import NodeView, EdgeView
from .base import ALL, DEFAULT_NODE_TYPE, DEFAULT_EDGE_TYPE, is_all, DGLError

__all__ = ['DGLHeteroGraph', 'DGLBaseBipartite', 'DGLGraph2']

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
    _ntypes_invmap, _etypes_invmap, _view_etype_idx :
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

        # Indicates which node/edge type (int) it is viewing (e.g. g[ntype])
        # The behavior of interfaces will change accordingly.
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
        return self._view_ntype_idx is not None

    @property
    def is_edge_type_view(self):
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
        The edges have a field "type" holding the edge type names.
        """
        nx_graph = self._graph.metagraph.to_networkx()
        for u_v in nx_graph.edges:
            nx_graph.edges[u_v]['type'] = self._etypes[nx_graph.edges[u_v]['id']]
        nx.relabel_nodes(
            nx_graph,
            {i: ntype for i, ntype in enumerate(self._ntypes)},
            copy=False)
        return nx_graph

    def _endpoint_types(self, etype):
        """Return the source and destination node type (int) of given edge
        type (int)."""
        return self._graph.metagraph.find_edge(etype)

    def endpoint_types(self, etype):
        """Return the source and destination node type of the given edge type.

        Parameters
        ----------
        etype : str
            The edge type.

        Returns
        -------
        srctype, dsttype : str, str
            The source node type and destination node type.
        """
        etype_idx = self._etypes_invmap[etype]
        srctype_idx, dsttype_idx = self._endpoint_types(etype_idx)
        return self._ntypes[srctype_idx], self._ntypes[dsttype_idx]

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
        if self.is_node_type_view:
            return [self._ntypes[self._view_ntype_idx]]
        elif self.is_edge_type_view:
            srctype_idx, dsttype_idx = self._endpoint_types(self._view_etype_idx)
            srctype = self._ntypes[srctype_idx]
            dsttype = self._ntypes[dsttype_idx]
            return [srctype, dsttype] if srctype != dsttype else [srctype]
        else:
            return self._ntypes

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
        if self.is_node_type_view:
            etype_indices = self._graph.metagraph.edge_id(
                self._view_ntype_idx, self._view_ntype_idx).tonumpy()
            return [self._etypes[etype_idx] for etype_idx in etype_indices]
        elif self.is_edge_type_view:
            return [self._etypes[self._view_etype_idx]]
        else:
            return self._etypes

    @property
    @utils.cached_member('_cache', '_current_ntype_idx')
    def _current_ntype_idx(self):
        """Checks the uniqueness of node type in the view and get the index
        of that node type.

        This allows reading/writing node frame data.
        """
        node_types = self.node_types()
        assert len(node_types) == 1, "only available for subgraphs with one node type"
        return self._ntypes_invmap[node_types[0]]

    @property
    @utils.cached_member('_cache', '_current_ntype_idx')
    def _current_etype_idx(self):
        """Checks the uniqueness of edge type in the view and get the index
        of that edge type.

        This allows reading/writing edge frame data and message passing routines.
        """
        edge_types = self.edge_types()
        assert len(edge_types) == 1, "only available for subgraphs with one edge type"
        return self._etypes_invmap[edge_types[0]]

    @property
    @utils.cached_member('_cache', '_current_srctype_idx')
    def _current_srctype_idx(self):
        """Checks the uniqueness of edge type in the view and get the index
        of the source type.

        This allows reading/writing edge frame data and message passing routines.
        """
        srctype_idx, dsttype_idx = self._endpoint_types(self._current_etype_idx)
        return srctype_idx

    @property
    @utils.cached_member('_cache', '_current_dsttype_idx')
    def _current_dsttype_idx(self):
        """Checks the uniqueness of edge type in the view and get the index
        of the destination type.

        This allows reading/writing edge frame data and message passing routines.
        """
        srctype_idx, dsttype_idx = self._endpoint_types(self._current_etype_idx)
        return dsttype_idx

    def number_of_nodes(self):
        """Return the number of nodes in the current view of the heterograph.

        Returns
        -------
        int
            The number of nodes

        Examples
        --------

        >>> g.number_of_nodes('user')
        3
        """
        # TODO: relax to multiple types
        return self._graph.number_of_nodes(self._current_ntype_idx)

    def _number_of_src_nodes(self):
        return self._graph.number_of_nodes(self._current_srctype_idx)

    def _number_of_dst_nodes(self):
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

    def number_of_edges(self):
        """Return the number of edges in the graph.

        Returns
        -------
        int
            The number of edges
        """
        # TODO: relax to multiple types
        return self._graph.number_of_edges(self._current_etype_idx)

    def has_node(self, vid):
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
        return self._graph.has_node(self._current_ntype_idx, vid)

    def has_nodes(self, vids):
        """Return a 0-1 array ``a`` given the node ID array ``vids``.

        ``a[i]`` is 1 if the graph contains node ``vids[i]``, 0 otherwise.

        Only works if the graph has one node type.  For multiple types,
        query with

        .. code::

           g.has_nodes(ntype, vids)

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

        >>> g.has_nodes('user', [0, 1, 2, 3, 4])
        tensor([1, 1, 1, 0, 0])

        See Also
        --------
        has_node
        """
        vids = utils.toindex(vids)
        rst = self._graph.has_nodes(self._current_ntype_idx, vids)
        return rst.tousertensor()

    def has_edge_between(self, u, v):
        """Return True if the edge (u, v) is in the graph.

        Only works if the graph has one edge type.  For multiple types,
        query with

        .. code::

           g['edgetype'].has_edge_between(u, v)

        Parameters
        ----------
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
        >>> g['plays'].has_edge_between(0, 1)
        True

        And whether Alice plays Minecraft
        >>> g['plays'].has_edge_between(0, 2)
        False

        See Also
        --------
        has_edges_between
        """
        return self._graph.has_edge_between(self._current_etype_idx, u, v)

    def has_edges_between(self, u, v):
        """Return a 0-1 array `a` given the source node ID array `u` and
        destination node ID array `v`.

        `a[i]` is 1 if the graph contains edge `(u[i], v[i])`, 0 otherwise.

        Only works if the graph has one edge type.  For multiple types,
        query with

        .. code::

           g['edgetype'].has_edges_between(u, v)

        Parameters
        ----------
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

        >>> g['plays'].has_edges_between([0, 0], [1, 2])
        tensor([1, 0])

        See Also
        --------
        has_edge_between
        """
        u = utils.toindex(u)
        v = utils.toindex(v)
        rst = self._graph.has_edges_between(self._current_etype_idx, u, v)
        return rst.tousertensor()

    def predecessors(self, v):
        """Return the predecessors of node `v` in the graph with the same
        edge type.

        Node `u` is a predecessor of `v` if an edge `(u, v)` exist in the
        graph.

        Only works if the graph has one edge type.  For multiple types,
        query with

        .. code::

           g['edgetype'].predecessors(v)

        Parameters
        ----------
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
        >>> g['plays'].predecessors(0)
        tensor([0, 1])

        This indicates User #0 (Alice) and User #1 (Bob).

        See Also
        --------
        successors
        """
        return self._graph.predecessors(self._current_etype_idx, v).tousertensor()

    def successors(self, v):
        """Return the successors of node `v` in the graph with the same edge
        type.

        Node `u` is a successor of `v` if an edge `(v, u)` exist in the
        graph.

        Only works if the graph has one edge type.  For multiple types,
        query with

        .. code::

           g['edgetype'].successors(v)

        Parameters
        ----------
        v : int
            The node of source type.

        Returns
        -------
        tensor
            Array of successor node IDs if destination node type.

        Examples
        --------
        The following example uses PyTorch backend.

        Asks which game Alice plays:
        >>> g['plays'].successors(0)
        tensor([0])

        This indicates Game #0 (Tetris).

        See Also
        --------
        predecessors
        """
        return self._graph.successors(self._current_etype_idx, v).tousertensor()

    def edge_id(self, u, v, force_multi=False):
        """Return the edge ID, or an array of edge IDs, between source node
        `u` and destination node `v`.

        Only works if the graph has one edge type.  For multiple types,
        query with

        .. code::

           g['edgetype'].edge_id(u, v)

        Parameters
        ----------
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
        >>> g['plays'].edge_id(1, 0)
        1

        See Also
        --------
        edge_ids
        """
        idx = self._graph.edge_id(self._current_etype_idx, u, v)
        return idx.tousertensor() if force_multi or self._graph.is_multigraph() else idx[0]

    def edge_ids(self, u, v, force_multi=False):
        """Return all edge IDs between source node array `u` and destination
        node array `v`.

        Only works if the graph has one edge type.  For multiple types,
        query with

        .. code::

           g['edgetype'].edge_ids(u, v)

        Parameters
        ----------
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
        >>> g['plays'].edge_ids([0, 1], [0, 1])
        tensor([0, 2])

        See Also
        --------
        edge_id
        """
        u = utils.toindex(u)
        v = utils.toindex(v)
        src, dst, eid = self._graph.edge_ids(self._current_etype_idx, u, v)
        if force_multi or self._graph.is_multigraph():
            return src.tousertensor(), dst.tousertensor(), eid.tousertensor()
        else:
            return eid.tousertensor()

    def find_edges(self, eid):
        """Given an edge ID array, return the source and destination node ID
        array `s` and `d`.  `s[i]` and `d[i]` are source and destination node
        ID for edge `eid[i]`.

        Only works if the graph has one edge type.  For multiple types,
        query with

        .. code::

           g['edgetype'].edge_ids(u, v)

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

        Find the user and game of gameplay #0 and #2:
        >>> g['plays'].find_edges([0, 2])
        (tensor([0, 1]), tensor([0, 1]))
        """
        eid = utils.toindex(eid)
        src, dst, _ = self._graph.find_edges(self._current_etype_idx, eid)
        return src.tousertensor(), dst.tousertensor()

    def in_edges(self, v, form='uv'):
        """Return the inbound edges of the node(s).

        Only works if the graph has one edge type.  For multiple types,
        query with

        .. code::

           g['edgetype'].edge_ids(u, v)

        Parameters
        ----------
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
        >>> g['plays'].in_edges(0, 'eid')
        tensor([0, 1])
        """
        v = utils.toindex(v)
        src, dst, eid = self._graph.in_edges(self._current_etype_idx, v)
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

        Only works if the graph has one edge type.  For multiple types,
        query with

        .. code::

           g['edgetype'].edge_ids(u, v)

        Parameters
        ----------
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
        >>> g['plays'].out_edges(0, 'eid')
        tensor([0])
        """
        v = utils.toindex(v)
        src, dst, eid = self._graph.out_edges(self._current_etype_idx, v)
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

        Only works if the graph has one edge type.  For multiple types,
        query with

        .. code::

           g['edgetype'].edge_ids(u, v)

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

        Find the user-game pairs for all gameplays:
        >>> g['plays'].all_edges('uv')
        (tensor([0, 1, 1, 2]), tensor([0, 0, 1, 1]))
        """
        src, dst, eid = self._graph.edges(self._current_etype_idx, order)
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

        Only works if the graph has one edge type.  For multiple types,
        query with

        .. code::

           g['edgetype'].edge_ids(u, v)

        Parameters
        ----------
        v : int
            The node ID of destination type.

        Returns
        -------
        int
            The in-degree.

        Examples
        --------
        Find how many users are playing Game #0 (Tetris):
        >>> g['plays'].in_degree(0)
        2

        See Also
        --------
        in_degrees
        """
        return self._graph.in_degree(self._current_etype_idx, v)

    def in_degrees(self, v=ALL):
        """Return the array `d` of in-degrees of the node array `v`.

        `d[i]` is the in-degree of node `v[i]`.

        Only works if the graph has one edge type.  For multiple types,
        query with

        .. code::

           g['edgetype'].edge_ids(u, v)

        Parameters
        ----------
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
        >>> g['plays'].in_degrees([0, 1])
        tensor([2, 2])

        See Also
        --------
        in_degree
        """
        _, dsttype_idx = self._endpoint_types(self._current_etype_idx)
        if is_all(v):
            v = utils.toindex(slice(0, self._graph.number_of_nodes(dsttype_idx)))
        else:
            v = utils.toindex(v)
        return self._graph.in_degrees(self._current_etype_idx, v).tousertensor()

    def out_degree(self, v):
        """Return the out-degree of node `v`.

        Only works if the graph has one edge type.  For multiple types,
        query with

        .. code::

           g['edgetype'].edge_ids(u, v)

        Parameters
        ----------
        v : int
            The node ID of source type.

        Returns
        -------
        int
            The out-degree.

        Examples
        --------
        Find how many games User #0 Alice is playing
        >>> g['plays'].out_degree(0)
        1

        See Also
        --------
        out_degrees
        """
        return self._graph.out_degree(self._current_etype_idx, v)

    def out_degrees(self, v=ALL):
        """Return the array `d` of out-degrees of the node array `v`.

        `d[i]` is the out-degree of node `v[i]`.

        Only works if the graph has one edge type.  For multiple types,
        query with

        .. code::

           g['edgetype'].edge_ids(u, v)

        Parameters
        ----------
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
        >>> g['plays'].out_degrees([0, 1])
        tensor([1, 2])

        See Also
        --------
        out_degree
        """
        srctype_idx, _ = self._endpoint_types(self._current_etype_idx)
        if is_all(v):
            v = utils.toindex(slice(0, self._graph.number_of_nodes(srctype_idx)))
        else:
            v = utils.toindex(v)
        return self._graph.out_degrees(self._current_etype_idx, v).tousertensor()


class DGLBaseBipartite(DGLBaseHeteroGraph):
    """Base bipartite graph class.
    """
    def __init__(self, graph, srctype, dsttype, etype):
        super(DGLBaseBipartite, self).__init__(
            graph,
            [srctype, dsttype] if srctype != dsttype else [srctype],
            [etype])
        self.srctype = srctype
        self.dsttype = dsttype

    @classmethod
    def from_csr(cls, srctype, dsttype, etype, num_src, num_dst, indptr, indices,
                 edge_ids):
        """Create a bipartite graph from CSR format.

        Parameters
        ----------
        srctype : str
            Name of source node type.
        dsttype : str
            Name of destination node type.
        etype : str
            Name of edge type.
        num_src : int
            Number of nodes in the source type.
        num_dst : int
            Number of nodes in the destination type.
        indptr, indices, edge_ids : Tensor, Tensor, Tensor
            The indptr, indices, and entries of the CSR matrix.
            The entries are edge IDs of the bipartite graph.
            The rows represent the source nodes and the columns represent the
            destination nodes.
        """
        indptr = utils.toindex(indptr)
        indices = utils.toindex(indices)
        edge_ids = utils.toindex(edge_ids)
        graph = heterograph_index.create_bipartite_from_csr(
            num_src, num_dst, indptr, indices, edge_ids)
        return cls(graph, srctype, dsttype, etype)

    @classmethod
    def from_coo(cls, srctype, dsttype, etype, num_src, num_dst, row, col):
        """Create a bipartite graph from COO format.

        Parameters
        ----------
        srctype : str
            Name of source node type.
        dsttype : str
            Name of destination node type.
        etype : str
            Name of edge type.
        num_src : int
            Number of nodes in the source type.
        num_dst : int
            Number of nodes in the destination type.
        row, col : Tensor, Tensor
            The row indices (source node IDs) and column indices (destination node IDs)
            of the COO matrix, ordered by edge IDs.
        """
        row = utils.toindex(row)
        col = utils.toindex(col)
        graph = heterograph_index.create_bipartite_from_coo(num_src, num_dst, row, col)
        return cls(graph, srctype, dsttype, etype)


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

        * list[DGLBaseBipartite]

          One could directly supply a list of bipartite graphs.  The metagraph
          would be then automatically figured out from the bipartite graph list.

          The bipartite graphs should not share the same edge type names.

          If two bipartite graphs share the same source node type, then the edges
          in the heterogeneous graph will share the same source node set.  This also
          applies to sharing the same destination node type, or having the same type
          for source nodes of one and destination nodes of the other.
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

    >>> follows = DGLBaseBipartite.from_coo(
    ...     'user', 'user', 'follows', 3, 3, [0, 1], [1, 2])
    >>> plays = DGLBaseBipartite.from_coo(
    ...     'user', 'game', 'plays', 3, 2, [0, 1, 1, 2], [0, 0, 1, 1])
    >>> develops = DGLBaseBipartite.from_coo(
    ...     'developer', 'game', 'develops', 2, 2, [0, 1], [0, 1])
    >>> g = DGLHeteroGraph([follows, plays, develops])

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
            self._view_ntype_idx = _view_ntype_idx
            self._view_etype_idx = _view_etype_idx
            return

        if isinstance(graph_data, list):
            if not isinstance(graph_data[0], DGLBaseBipartite):
                raise TypeError('Only list of DGLBaseBipartite is supported')

            srctypes, dsttypes, etypes = [], [], []
            ntypes = []
            ntypes_invmap = {}
            etypes_invmap = {}
            for bipartite in graph_data:
                etype, = bipartite.all_edge_types
                srctype = bipartite.srctype
                dsttype = bipartite.dsttype

                srctypes.append(srctype)
                dsttypes.append(dsttype)
                etypes_invmap[etype] = len(etypes_invmap)
                etypes.append(etype)

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
            hg_index = heterograph_index.create_heterograph(
                metagraph_index, [bipartite._graph for bipartite in graph_data])

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
                size=self.number_of_edges())
        return self._msg_indices[self._current_etype_idx]

    def _set_msg_index(self, index):
        self._msg_indices[self._current_etype_idx] = index

    def __getitem__(self, key):
        if key in self._ntypes_invmap:
            return self._create_view(self._ntypes_invmap[key], None)
        elif key in self._etypes_invmap:
            return self._create_view(None, self._etypes_invmap[key])
        else:
            raise KeyError(key)

    @property
    def _node_frame(self):
        return self._node_frames[self._current_ntype_idx]

    @property
    def _edge_frame(self):
        return self._edge_frames[self._current_etype_idx]

    @property
    def _src_frame(self):
        return self._node_frames[self._current_srctype_idx]

    @property
    def _dst_frame(self):
        return self._node_frames[self._current_dsttype_idx]

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
        etype : str
            The edge type name.  Must exist in the metagraph.
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
        etype : str
            The edge type name.  Must exist in the metagraph.
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

    def node_attr_schemes(self):
        """Return the node feature schemes for a given node type.

        Each feature scheme is a named tuple that stores the shape and data type
        of the node feature

        Parameters
        ----------
        ntype : str
            The node type

        Returns
        -------
        dict of str to schemes
            The schemes of node feature columns.
        """
        return self._node_frame.schemes

    def edge_attr_schemes(self):
        """Return the edge feature schemes for a given edge type.

        Each feature scheme is a named tuple that stores the shape and data type
        of the edge feature

        Parameters
        ----------
        etype : tuple[str, str, str]
            The edge type, characterized by a triplet of source type name,
            destination type name, and edge type name.

        Returns
        -------
        dict of str to schemes
            The schemes of node feature columns.
        """
        return self._edge_frame.schemes

    def set_n_initializer(self, initializer, field=None):
        """Set the initializer for empty node features of given type.

        Initializer is a callable that returns a tensor given the shape, data type
        and device context.

        When a subset of the nodes are assigned a new feature, initializer is
        used to create feature for rest of the nodes.

        Parameters
        ----------
        ntype : str
            The node type name.
        initializer : callable
            The initializer.
        field : str, optional
            The feature field name. Default is set an initializer for all the
            feature fields.
        """
        self._node_frame.set_initializer(initializer, field)

    def set_e_initializer(self, etype, initializer, field=None):
        """Set the initializer for empty edge features of given type.

        Initializer is a callable that returns a tensor given the shape, data
        type and device context.

        When a subset of the edges are assigned a new feature, initializer is
        used to create feature for rest of the edges.

        Parameters
        ----------
        etype : tuple[str, str, str]
            The edge type, characterized by a triplet of source type name,
            destination type name, and edge type name.
        initializer : callable
            The initializer.
        field : str, optional
            The feature field name. Default is set an initializer for all the
            feature fields.
        """
        self._edge_frame.set_initializer(initializer, field)

    @property
    def nodes(self):
        """Return a node view that can used to set/get feature data of a
        single node type.

        Notes
        -----
        An error is raised if the graph contains multiple node types.  Use

            g[ntype]

        to select nodes with type ``ntype``.

        Examples
        --------
        To set features of User #0 and #2 in a heterogeneous graph:
        >>> g['user'].nodes[[0, 2]].data['h'] = torch.zeros(2, 5)
        """
        return NodeView(self)

    @property
    def ndata(self):
        """Return the data view of all the nodes of a single node type.

        Notes
        -----
        An error is raised if the graph contains multiple node types.  Use

            g[ntype]

        to select nodes with type ``ntype``.

        Examples
        --------
        To set features of games in a heterogeneous graph:
        >>> g['game'].ndata['h'] = torch.zeros(2, 5)
        """
        return self.nodes[:].data

    @property
    def edges(self):
        """Return an edges view that can used to set/get feature data of a
        single edge type.

        Notes
        -----
        An error is raised if the graph contains multiple edge types.  Use

            g[src_type, dst_type, edge_type]

        to select edges with type ``(src_type, dst_type, edge_type)``.

        Examples
        --------
        To set features of gameplays #1 (Bob -> Tetris) and #3 (Carol ->
        Minecraft) in a heterogeneous graph:
        >>> g['user', 'game', 'plays'].edges[[1, 3]].data['h'] = torch.zeros(2, 5)
        """
        return EdgeView(self)

    @property
    def edata(self):
        """Return the data view of all the edges of a single edge type.

        Notes
        -----
        An error is raised if the graph contains multiple edge types.  Use

            g[src_type, dst_type, edge_type]

        to select edges with type ``(src_type, dst_type, edge_type)``.

        Examples
        --------
        >>> g['developer', 'game', 'develops'].edata['h'] = torch.zeros(2, 5)
        """
        return self.edges[:].data

    def set_n_repr(self, data, u=ALL, inplace=False):
        """Set node(s) representation of a single node type.

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

        if is_all(u):
            for key, val in data.items():
                self._node_frame[key] = val
        else:
            self._node_frame.update_rows(u, data, inplace=inplace)

    def get_n_repr(self, u=ALL):
        """Get node(s) representation of a single node type.

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
        """Get and remove the specified node repr of a given node type.

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
        """Set edge(s) representation of a single edge type.

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
            Edges can be either

            * A pair of endpoint nodes (u, v), where u is the node ID of source
              node type and v is that of destination node type.
            * A tensor of edge ids of the given type.

            The default value is all the edges.
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

    def pop_e_repr(self, etype, key):
        """Get and remove the specified edge repr of a single edge type.

        Parameters
        ----------
        etype : tuple[str, str, str]
            The edge type, characterized by a triplet of source type name,
            destination type name, and edge type name.
        key : str
          The attribute name.

        Returns
        -------
        Tensor
            The popped representation
        """
        self._edge_frame.pop(key)

    def register_message_func(self, func):
        """Register global message function for each edge type provided.

        Once registered, ``func`` will be used as the default
        message function in message passing operations, including
        :func:`send`, :func:`send_and_recv`, :func:`pull`,
        :func:`push`, :func:`update_all`.

        Parameters
        ----------
        func : callable, dict[etype, callable]
            Message function on the edge. The function should be
            an :mod:`Edge UDF <dgl.udf>`.

            If a dict is provided, the functions will be applied according to
            edge type.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.
            If the graph has more than one edge type and ``func`` is not a
            dict, it will throw an error.

        See Also
        --------
        send
        send_and_recv
        pull
        push
        update_all
        """
        pass

    def register_reduce_func(self, func):
        """Register global message reduce function for each edge type provided.

        Once registered, ``func`` will be used as the default
        message reduce function in message passing operations, including
        :func:`recv`, :func:`send_and_recv`, :func:`push`, :func:`pull`,
        :func:`update_all`.

        Parameters
        ----------
        func : callable, dict[etype, callable]
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.

            If a dict is provided, the messages will be aggregated onto the
            nodes by the edge type of the message.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.
            If the graph has more than one edge type and ``reduce_func`` is not
            a dict, it will throw an error.

        See Also
        --------
        recv
        send_and_recv
        push
        pull
        update_all
        """
        pass

    def register_apply_node_func(self, func):
        """Register global node apply function for each node type provided.

        Once registered, ``func`` will be used as the default apply
        node function. Related operations include :func:`apply_nodes`,
        :func:`recv`, :func:`send_and_recv`, :func:`push`, :func:`pull`,
        :func:`update_all`.

        Parameters
        ----------
        func : callable, dict[str, callable]
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.

            If a dict is provided, the functions will be applied according to
            node type.
            If the graph has more than one node type and ``func`` is not a
            dict, it will throw an error.

        See Also
        --------
        apply_nodes
        register_apply_edge_func
        """
        pass

    def register_apply_edge_func(self, func):
        """Register global edge apply function for each edge type provided.

        Once registered, ``func`` will be used as the default apply
        edge function in :func:`apply_edges`.

        Parameters
        ----------
        func : callable, dict[etype, callable]
            Apply function on the edge. The function should be
            an :mod:`Edge UDF <dgl.udf>`.

            If a dict is provided, the functions will be applied according to
            edge type.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.
            If the graph has more than one edge type and ``func`` is not a
            dict, it will throw an error.

        See Also
        --------
        apply_edges
        register_apply_node_func
        """
        pass

    def apply_nodes(self, func, v=ALL, inplace=False):
        """Apply the function on the nodes with the same type to update their
        features.

        If None is provided for ``func``, nothing will happen.

        Parameters
        ----------
        func : callable, dict[str, callable], or None
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.

            If a dict is provided, the functions will be applied according to
            node type.
            If the graph has more than one node type and ``func`` is not a
            dict, it will throw an error.
        v : int, iterable of int, tensor, dict, optional
            The (type-specific) node (ids) on which to apply ``func``.

            If ``func`` is not a dict, then ``v`` must not be a dict.
            If ``func`` is a dict, then ``v`` must either be
            * ALL: for computing on all nodes with the given types in ``func``.
            * a dict of int, iterable of int, or tensors, with the same keys
              as ``func``, indicating the nodes to be updated for each type.
        inplace : bool, optional
            If True, update will be done in place, but autograd will break.

        Examples
        --------
        >>> g['user'].ndata['h'] = torch.ones(3, 5)
        >>> g['user'].apply_nodes(lambda nodes: {'h': nodes.data['h'] * 2})
        >>> g['user'].ndata['h']
        tensor([[2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.]])
        """
        assert not utils.is_dict_like(func), \
            "multiple-type message passing is not implemented"

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

    def apply_edges(self, func, edges=ALL, inplace=False):
        """Apply the function on the edges with the same type to update their
        features.

        If None is provided for ``func``, nothing will happen.

        Parameters
        ----------
        func : callable, dict[etype, callable], or None
            Apply function on the edge. The function should be
            an :mod:`Edge UDF <dgl.udf>`.

            If a dict is provided, the functions will be applied according to
            edge type.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.
            If the graph has more than one edge type and ``func`` is not a
            dict, it will throw an error.
        edges : any valid edge specification, dict, optional
            Edges on which to apply ``func``. See :func:`send` for valid
            edge specification.

            If ``func`` is not a dict, then ``edges`` must not be a dict.
            If ``func`` is a dict, then ``edges`` must either be
            * ALL: for computing on all edges with the given types in ``func``.
            * a dict of int, iterable of int, or tensors, with the same keys
              as ``func``, indicating the edges to be updated for each type.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.

        Examples
        --------
        >>> g['plays'].edata['h'] = torch.ones(4, 5)
        >>> g['plays'].apply_edges(lambda edges: {'h': edges.data['h'] * 2})
        >>> g['plays'].edata['h']
        tensor([[2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.]])
        """
        assert not utils.is_dict_like(func), \
            "multiple-type message passing is not implemented"

        if is_all(edges):
            u, v, _ = self._graph.edges(self._current_etype_idx, 'eid')
            eid = utils.toindex(slice(0, self.number_of_edges()))
        elif isinstance(edges, tuple):
            u, v = edges
            u = utils.toindex(u)
            v = utils.toindex(v)
            # Rewrite u, v to handle edge broadcasting and multigraph.
            u, v, eid = self._graph.edge_ids(self._current_etype_idx, u, v)
        else:
            eid = utils.toindex(edges)
            u, v, _ = self._graph.find_edges(self._current_etype_idx, eid)

        with ir.prog() as prog:
            scheduler.schedule_apply_edges(graph=self,
                                           u=u,
                                           v=v,
                                           eid=eid,
                                           apply_func=func,
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
        func : callable, dict[etype, callable]
            Apply function on the edge.  The function should be
            an :mod:`Edge UDF <dgl.udf>`. The input of `Edge UDF` should
            be (bucket_size, degrees, *feature_shape), and
            return the dict with values of the same shapes.

            If a dict is provided, the functions will be applied according to
            edge type.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.
            If the graph has more than one edge type and ``func`` is not a
            dict, it will throw an error.
        edges : valid edges type, dict, optional
            Edges on which to group and apply ``func``. See :func:`send` for valid
            edges type. Default is all the edges.

            If ``func`` is not a dict, then ``edges`` must not be a dict.
            If ``func`` is a dict, then ``edges`` must either be
            * ALL: for computing on all edges with the given types in ``func``.
            * a dict of int, iterable of int, or tensors, with the same keys
              as ``func``, indicating the edges to be updated for each type.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.
        """
        assert not utils.is_dict_like(func), \
            "multiple-type message passing is not implemented"

        if group_by not in ('src', 'dst'):
            raise DGLError("Group_by should be either src or dst")

        if is_all(edges):
            u, v, _ = self._graph.edges(self._current_etype_idx)
            eid = utils.toindex(slice(0, self.number_of_edges()))
        elif isinstance(edges, tuple):
            u, v = edges
            u = utils.toindex(u)
            v = utils.toindex(v)
            # Rewrite u, v to handle edge broadcasting and multigraph.
            u, v, eid = self._graph.edge_ids(self._current_etype_idx, u, v)
        else:
            eid = utils.toindex(edges)
            u, v, _ = self._graph.find_edges(self._current_etype_idx, eid)

        with ir.prog() as prog:
            scheduler.schedule_group_apply_edge(graph=self,
                                                u=u,
                                                v=v,
                                                eid=eid,
                                                apply_func=func,
                                                group_by=group_by,
                                                inplace=inplace)
            Runtime.run(prog)

    # TODO: REVIEW
    def send(self, edges=ALL, message_func=None):
        """Send messages along the given edges with the same edge type.

        ``edges`` can be any of the following types:

        * ``int`` : Specify one edge using its edge id (of the given edge type).
        * ``pair of int`` : Specify one edge using its endpoints (of source node type
          and destination node type respectively).
        * ``int iterable`` / ``tensor`` : Specify multiple edges using their edge ids.
        * ``pair of int iterable`` / ``pair of tensors`` :
          Specify multiple edges using their endpoints.
        * a dict of all the above, if ``message_func`` is a dict.

        The UDF returns messages on the edges and can be later fetched in
        the destination node's ``mailbox``. Receiving will consume the messages.
        See :func:`recv` for example.

        If multiple ``send`` are triggered on the same edge without ``recv``. Messages
        generated by the later ``send`` will overwrite previous messages.

        Parameters
        ----------
        edges : valid edges type, dict, optional
            Edges on which to apply ``message_func``. Default is sending along all
            the edges.

            If ``message_func`` is not a dict, then ``edges`` must not be a dict.
            If ``message_func`` is a dict, then ``edges`` must either be
            * ALL: for computing on all edges with the given types in
              ``message_func``.
            * a dict of int, iterable of int, or tensors, with the same keys
              as ``message_func``, indicating the edges to be updated for each
              type.
        message_func : callable, dict[etype, callable]
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.

            If a dict is provided, the functions will be applied according to
            edge type.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.
            If the graph has more than one edge type and ``message_func`` is
            not a dict, it will throw an error.

        Notes
        -----
        On multigraphs, if :math:`u` and :math:`v` are specified, then the messages will be sent
        along all edges between :math:`u` and :math:`v`.
        """
        assert not utils.is_dict_like(message_func), \
            "multiple-type message passing is not implemented"
        assert message_func is not None

        if is_all(edges):
            eid = utils.toindex(slice(0, self.number_of_edges()))
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

        Parameters
        ----------
        v : int, container or tensor, dict, optional
            The node(s) to be updated. Default is receiving all the nodes.

            If ``apply_node_func`` is not a dict, then ``v`` must not be a
            dict.
            If ``apply_node_func`` is a dict, then ``v`` must either be
            * ALL: for computing on all nodes with the given types in
              ``apply_node_func``.
            * a dict of int, iterable of int, or tensors, indicating the nodes
              to be updated for each type.
        reduce_func : callable, dict[etype, callable], optional
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.

            If a dict is provided, the messages will be aggregated onto the
            nodes by the edge type of the message.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.
            If the graph has more than one edge type and ``reduce_func`` is not
            a dict, it will throw an error.
        apply_node_func : callable, dict[str, callable]
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.

            If a dict is provided, the functions will be applied according to
            node type.
            If the graph has more than one node type and ``apply_func`` is not
            a dict, it will throw an error.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.

        Notes
        -----
        If the graph is heterogeneous (i.e. having more than one node/edge
        type),
        * the node types in ``v``, the node types in ``apply_node_func``,
          and the destination types in ``reduce_func`` must be the same.
        """
        assert not utils.is_dict_like(reduce_func) and \
            not utils.is_dict_like(apply_node_func), \
            "multiple-type message passing is not implemented"
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

        Parameters
        ----------
        edges : valid edges type
            Edges on which to apply ``func``. See :func:`send` for valid
            edges type.

            If the functions are not dicts, then ``edges`` must not be a dict.
            If the functions are dicts, then ``edges`` must either be
            * ALL: for computing on all edges with the given types in the
              functions.
            * a dict of int, iterable of int, or tensors, indicating the edges
              to be updated for each type.
        message_func : callable, dict[etype, callable], optional
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.

            If a dict is provided, the functions will be applied according to
            edge type.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.
            If the graph has more than one edge type and ``message_func`` is
            not a dict, it will throw an error.
        reduce_func : callable, dict[etype, callable], optional
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.

            If a dict is provided, the messages will be aggregated onto the
            nodes by the edge type of the message.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.
            If the graph has more than one edge type and ``reduce_func`` is not
            a dict, it will throw an error.
        apply_node_func : callable, dict[str, callable], optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.

            If a dict is provided, the functions will be applied according to
            node type.
            If the graph has more than one node type and ``apply_func`` is not
            a dict, it will throw an error.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.

        Notes
        -----
        If the graph is heterogeneous (i.e. having more than one node/edge
        type),
        * the destination type of ``edges``, the node types in
          ``apply_node_func``, and the destination types in ``reduce_func``
          must be the same.
        * the edge type of ``edges``, ``message_func`` and ``reduce_func``
          must also be the same.
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
        v : int, container or tensor, dict, optional
            The node(s) to be updated. Default is receiving all the nodes.

            If the functions are not dicts, then ``v`` must not be a dict.
            If the functions are dicts, then ``v`` must either be
            * ALL: for computing on all nodes with the given types in the
              functions.
            * a dict of int, iterable of int, or tensors, indicating the nodes
              to be updated for each type.
        message_func : callable, dict[etype, callable], optional
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.

            If a dict is provided, the functions will be applied according to
            edge type.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.
            If the graph has more than one edge type and ``message_func`` is
            not a dict, it will throw an error.
        reduce_func : callable, dict[etype, callable], optional
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.

            If a dict is provided, the messages will be aggregated onto the
            nodes by the edge type of the message.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.
            If the graph has more than one edge type and ``reduce_func`` is not
            a dict, it will throw an error.
        apply_node_func : callable, dict[str, callable], optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.

            If a dict is provided, the functions will be applied according to
            node type.
            If the graph has more than one node type and ``apply_func`` is not
            a dict, it will throw an error.

        Notes
        -----
        If the graph is heterogeneous (i.e. having more than one node/edge
        type),
        * the node types of ``v``, the node types in ``apply_node_func``,
          and the destination types in ``reduce_func`` must be the same.
        * the edge type of ``message_func`` and ``reduce_func`` must also be
          the same.
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
             message_func="default",
             reduce_func="default",
             apply_node_func="default",
             inplace=False):
        """Send message from the node(s) to their successors and update them.

        Optionally, apply a function to update the node features after receive.

        Parameters
        ----------
        u : int, container or tensor, dict
            The node(s) to push messages out.

            If the functions are not dicts, then ``v`` must not be a dict.
            If the functions are dicts, then ``v`` must either be
            * ALL: for computing on all nodes with the given types in the
              functions.
            * a dict of int, iterable of int, or tensors, indicating the nodes
              to be updated for each type.
        message_func : callable, dict[etype, callable], optional
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.

            If a dict is provided, the functions will be applied according to
            edge type.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.
            If the graph has more than one edge type and ``message_func`` is
            not a dict, it will throw an error.
        reduce_func : callable, dict[etype, callable], optional
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.

            If a dict is provided, the messages will be aggregated onto the
            nodes by the edge type of the message.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.
            If the graph has more than one edge type and ``reduce_func`` is not
            a dict, it will throw an error.
        apply_node_func : callable, dict[str, callable], optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.

            If a dict is provided, the functions will be applied according to
            node type.
            If the graph has more than one node type and ``apply_func`` is not
            a dict, it will throw an error.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.

        Notes
        -----
        If the graph is heterogeneous (i.e. having more than one node/edge
        type),
        * the node types in ``apply_node_func`` and the destination types in
          ``reduce_func`` must be the same.
        * the source types of ``message_func`` and the node types of ``u`` must
          be the same.
        * the edge type of ``message_func`` and ``reduce_func`` must also be
          the same.
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

        Parameters
        ----------
        message_func : callable, dict[etype, callable], optional
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.

            If a dict is provided, the functions will be applied according to
            edge type.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.
            If the graph has more than one edge type and ``message_func`` is
            not a dict, it will throw an error.
        reduce_func : callable, dict[etype, callable], optional
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.

            If a dict is provided, the messages will be aggregated onto the
            nodes by the edge type of the message.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.
            If the graph has more than one edge type and ``reduce_func`` is not
            a dict, it will throw an error.
        apply_node_func : callable, dict[str, callable], optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.

            If a dict is provided, the functions will be applied according to
            node type.
            If the graph has more than one node type and ``apply_func`` is not
            a dict, it will throw an error.

        Notes
        -----
        If the graph is heterogeneous (i.e. having more than one node/edge
        type),
        * the node types in ``apply_node_func`` and the destination types in
          ``reduce_func`` must be the same.
        * the edge type of ``message_func`` and ``reduce_func`` must also be
          the same.
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

    # TODO should we support this?
    def prop_nodes(self,
                   nodes_generator,
                   message_func="default",
                   reduce_func="default",
                   apply_node_func="default"):
        """Node propagation in heterogeneous graph is not supported.
        """
        raise NotImplementedError('not supported')

    # TODO should we support this?
    def prop_edges(self,
                   edges_generator,
                   message_func="default",
                   reduce_func="default",
                   apply_node_func="default"):
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


class DGLGraph2(DGLHeteroGraph):
    def __init__(
            self,
            graph_data=None,
            node_frame=None,
            edge_frame=None,
            multigraph=None,
            readonly=True):
        if isinstance(graph_data, list):
            from .factory import graph_from_edge_list
            u, v = zip(*graph_data)
            bipartite = graph_from_edge_list(u, v)
            super(DGLGraph2, self).__init__(
                    [bipartite],
                    [node_frame] if node_frame is not None else None,
                    [edge_frame] if edge_frame is not None else None,
                    multigraph,
                    readonly)
