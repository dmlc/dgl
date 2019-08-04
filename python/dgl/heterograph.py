"""Classes for heterogeneous graphs."""
import networkx as nx
from . import heterograph_index, graph_index
from . import utils
from . import backend as F
from .base import ALL, is_all

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

        # Indicates which node/edge type (int) it is viewing (e.g. g[ntype])
        # The behavior of interfaces will change accordingly.
        self._view_ntype_idx = _view_ntype_idx
        self._view_etype_idx = _view_etype_idx

    def _create_node_type_view(self, ntype):
        return DGLBaseHeteroGraph(
            self._graph, self._ntypes, self._etypes,
            self._ntypes_invmap, self._etypes_invmap,
            ntype, None)

    def _create_edge_type_view(self, etype):
        return DGLBaseHeteroGraph(
            self._graph, self._ntypes, self._etypes,
            self._ntypes_invmap, self._etypes_invmap,
            None, etype)

    @property
    def is_node_type_view(self):
        return self._view_ntype_idx is not None

    @property
    def is_edge_type_view(self):
        return self._view_etype_idx is not None

    @property
    def is_view(self):
        return self.is_node_type_view or self.is_edge_type_view

    @property
    def node_types(self):
        return self._ntypes

    @property
    def edge_types(self):
        return self._etypes

    def __getitem__(self, key):
        """Returns a view on the heterogeneous graph with given node/edge
        type:

        If ``key`` is a str, it returns a heterogeneous subgraph induced
        from nodes or edges of type ``key``.

        The view would share the frames with the parent graph; any
        modifications on one's frames would reflect on the other.

        Note that the subgraph itself is not materialized until someone
        queries the subgraph structure.  This implies that calling computation
        methods such as

            g['user'].update_all(...)

        would not actually create a subgraph of users.

        Parameters
        ----------
        key : str
            See above

        Returns
        -------
        DGLBaseHeteroGraphView
            The induced subgraph view.
        """
        if self.is_view:
            raise RuntimeError('Cannot create a view from a view')
        else:
            if key in self._ntypes_invmap:
                return self._create_node_type_view(self._ntypes_invmap[key])
            elif key in self._etypes_invmap:
                return self._create_edge_type_view(self._etypes_invmap[key])
            else:
                raise KeyError('%s is neither a node type or an edge type' % key)

    @property
    def metagraph(self):
        """Return the metagraph as networkx.MultiDiGraph.

        The nodes are labeled with node type names.
        The edges have a field "type" holding the edge type names.
        """
        nx_graph = self._graph.metagraph.to_networkx()
        for i, uv in enumerate(nx_graph.edges):
            nx_graph.edges[uv]['type'] = self._etypes[nx_graph.edges[uv]['id']]
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

    def number_of_nodes(self):
        """Return the number of nodes in the graph.

        Returns
        -------
        int
            The number of nodes
        """
        assert self.is_node_type_view, 'only supported on node type views'
        return self._graph.number_of_nodes(self._view_ntype_idx)

    def __len__(self):
        """Return the number of nodes in the graph."""
        return self.number_of_nodes()

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
        assert self.is_edge_type_view, 'only supported on edge type views'
        return self._graph.number_of_edges(self._view_etype_idx)

    def has_node(self, vid):
        """Return True if the graph contains node `vid`.

        Only works if the graph has one node type.  For multiple types,
        query with

        .. code::

           g['vtype'].has_node(vid)

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
        >>> g['user'].has_node(0)
        True
        >>> g['user'].has_node(4)
        False

        Equivalently,

        >>> 0 in g['user']
        True

        See Also
        --------
        has_nodes
        """
        assert self.is_node_type_view, 'only supported on node type views'
        return self._graph.has_node(self._view_ntype_idx, vid)

    def __contains__(self, vid):
        """Return True if the graph contains node `vid`.

        Only works if the graph has one node type.  For multiple types,
        query with

        .. code::

           vid in g['vtype']

        Examples
        --------
        >>> 0 in g['user']
        True
        """
        return self.has_node(self._view_ntype_idx, vid)

    def has_nodes(self, vids):
        """Return a 0-1 array ``a`` given the node ID array ``vids``.

        ``a[i]`` is 1 if the graph contains node ``vids[i]``, 0 otherwise.

        Only works if the graph has one node type.  For multiple types,
        query with

        .. code::

           g['vtype'].has_nodes(vids)

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

        >>> g['user'].has_nodes([0, 1, 2, 3, 4])
        tensor([1, 1, 1, 0, 0])

        See Also
        --------
        has_node
        """
        assert self.is_node_type_view, 'only supported on node type views'
        vids = utils.toindex(vids)
        rst = self._graph.has_nodes(self._view_ntype_idx, vids)
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
        assert self.is_edge_type_view, 'only supported on edge type views'
        return self._graph.has_edge_between(self._view_etype_idx, u, v)

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
        assert self.is_edge_type_view, 'only supported on edge type views'
        u = utils.toindex(u)
        v = utils.toindex(v)
        rst = self._graph.has_edges_between(self._view_etype_idx, u, v)
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
        assert self.is_edge_type_view, 'only supported on edge type views'
        return self._graph.predecessors(self._view_etype_idx, v).tousertensor()

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
        assert self.is_edge_type_view, 'only supported on edge type views'
        return self._graph.successors(self._view_etype_idx, v).tousertensor()

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
        assert self.is_edge_type_view, 'only supported on edge type views'
        idx = self._graph.edge_id(self._view_etype_idx, u, v)
        return idx.tousertensor() if force_multi or self.is_multigraph else idx[0]

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
        assert self.is_edge_type_view, 'only supported on edge type views'
        u = utils.toindex(u)
        v = utils.toindex(v)
        src, dst, eid = self._graph.edge_ids(self._view_etype_idx, u, v)
        if force_multi or self.is_multigraph:
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
        assert self.is_edge_type_view, 'only supported on edge type views'
        eid = utils.toindex(eid)
        src, dst, _ = self._graph.find_edges(self._view_etype_idx, eid)
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
        assert self.is_edge_type_view, 'only supported on edge type views'

        v = utils.toindex(v)
        src, dst, eid = self._graph.in_edges(self._view_etype_idx, v)
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
        assert self.is_edge_type_view, 'only supported on edge type views'
        v = utils.toindex(v)
        src, dst, eid = self._graph.out_edges(self._view_etype_idx, v)
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
        assert self.is_edge_type_view, 'only supported on edge type views'
        src, dst, eid = self._graph.edges(self._view_etype_idx, order)
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
        assert self.is_edge_type_view, 'only supported on edge type views'
        return self._graph.in_degree(self._view_etype_idx, v)

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
        assert self.is_edge_type_view, 'only supported on edge type views'
        _, dsttype_idx = self._endpoint_types(self._view_etype_idx)
        if is_all(v):
            v = utils.toindex(slice(0, self._graph.number_of_nodes(dsttype_idx)))
        else:
            v = utils.toindex(v)
        return self._graph.in_degrees(self._view_etype_idx, v).tousertensor()

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
        assert self.is_edge_type_view, 'only supported on edge type views'
        return self._graph.out_degree(self._view_etype_idx, v)

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
        assert self.is_edge_type_view, 'only supported on edge type views'
        srctype_idx, _ = self._endpoint_types(self._view_etype_idx)
        if is_all(v):
            v = utils.toindex(slice(0, self._graph.number_of_nodes(srctype_idx)))
        else:
            v = utils.toindex(v)
        return self._graph.out_degrees(self._view_etype_idx, v).tousertensor()


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
    node_frame : dict[str, dict[str, Tensor]]
        The node frames for each node type
    edge_frame : dict[str, dict[str, Tensor]]
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
    """
    # pylint: disable=unused-argument
    def __init__(
            self,
            graph_data=None,
            node_frame=None,
            edge_frame=None,
            multigraph=None,
            readonly=False):
        if isinstance(graph_data, list):
            if not isinstance(graph_data[0], DGLBaseBipartite):
                raise TypeError('Only list of DGLBaseBipartite is supported')

            srctypes, dsttypes, etypes = [], [], []
            ntypes = []
            ntypes_invmap = {}
            etypes_invmap = {}
            for bipartite in graph_data:
                etype, = bipartite.edge_types
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
        self._graph.add_nodes(node_type, num)
        # TODO: frame

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
        u = utils.toindex(u)
        v = utils.toindex(v)
        self._graph.add_edge(u, v)
        # TODO: frame

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
        self._graph.add_edges(u, v)
        # TODO: frame

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
        pass

    def edge_attr_schemes(self, etype):
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
        pass

    def set_n_initializer(self, ntype, initializer, field=None):
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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

    def set_n_repr(self, data, ntype, u=ALL, inplace=False):
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
        ntype : str
            Node type.
        u : node, container or tensor
            The node(s).
        inplace : bool
            If True, update will be done in place, but autograd will break.
        """
        pass

    def get_n_repr(self, ntype, u=ALL):
        """Get node(s) representation of a single node type.

        The returned feature tensor batches multiple node features on the first dimension.

        Parameters
        ----------
        ntype : str
            Node type.
        u : node, container or tensor
            The node(s).

        Returns
        -------
        dict
            Representation dict from feature name to feature tensor.
        """
        pass

    def pop_n_repr(self, ntype, key):
        """Get and remove the specified node repr of a given node type.

        Parameters
        ----------
        ntype : str
            The node type.
        key : str
            The attribute name.

        Returns
        -------
        Tensor
            The popped representation
        """
        pass

    def set_e_repr(self, data, etype, edges=ALL, inplace=False):
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
        etype : tuple[str, str, str]
            The edge type, characterized by a triplet of source type name,
            destination type name, and edge type name.
        edges : edges
            Edges can be either

            * A pair of endpoint nodes (u, v), where u is the node ID of source
              node type and v is that of destination node type.
            * A tensor of edge ids of the given type.

            The default value is all the edges.
        inplace : bool
            If True, update will be done in place, but autograd will break.
        """
        pass

    def get_e_repr(self, etype, edges=ALL):
        """Get edge(s) representation.

        Parameters
        ----------
        etype : tuple[str, str, str]
            The edge type, characterized by a triplet of source type name,
            destination type name, and edge type name.
        edges : edges
            Edges can be a pair of endpoint nodes (u, v), or a
            tensor of edge ids. The default value is all the edges.

        Returns
        -------
        dict
            Representation dict
        """
        pass

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
        pass

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
        >>> g['user'].apply_nodes(lambda x: {'h': x * 2})
        >>> g['user'].ndata['h']
        tensor([[2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.]])
        >>> g.apply_nodes({'user': lambda x: {'h': x * 2}})
        >>> g['user'].ndata['h']
        tensor([[4., 4., 4., 4., 4.],
                [4., 4., 4., 4., 4.],
                [4., 4., 4., 4., 4.]])
        """
        pass

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
        >>> g['user', 'game', 'plays'].edata['h'] = torch.ones(3, 5)
        >>> g['user', 'game', 'plays'].apply_edges(lambda x: {'h': x * 2})
        >>> g['user', 'game', 'plays'].edata['h']
        tensor([[2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.]])
        >>> g.apply_edges({('user', 'game', 'plays'): lambda x: {'h': x * 2}})
        tensor([[4., 4., 4., 4., 4.],
                [4., 4., 4., 4., 4.],
                [4., 4., 4., 4., 4.]])
        """
        pass

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
        pass

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
        pass

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
        pass

    def send_and_recv(self,
                      edges,
                      message_func="default",
                      reduce_func="default",
                      apply_node_func="default",
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
        pass

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
        pass

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
        pass

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
        pass

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
