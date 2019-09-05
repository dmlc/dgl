"""Classes for heterogeneous graphs."""
from collections import defaultdict
from functools import partial
import networkx as nx
import scipy.sparse as ssp
from . import heterograph_index, graph_index
from . import utils
from . import backend as F
from . import init
from .runtime import ir, scheduler, Runtime, GraphAdapter
from .frame import Frame, FrameRef, frame_like
from .view import HeteroNodeView, HeteroNodeDataView, HeteroEdgeView, HeteroEdgeDataView
from .base import ALL, SLICE_FULL, is_all, DGLError

__all__ = ['DGLHeteroGraph']

class DGLHeteroGraph(object):
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

    Parameters
    ----------
    graph_data : graph data
        Data to initialize heterograph structure.
        Supported type: DGLGraph, HeteroGraphIndex
    ntypes : list of str
        Node type list. The i^th element stores the type name
        of node type i.
    etypes : list of str
        Edge type list. The i^th element stores the type name
        of edge type i.
    node_frames : list of FrameRef, optional
        Node feature storage. The i^th element stores the node features
        of node type i. If None, empty frame is created.  (default: None)
    edge_frames : list of FrameRef, optional
        Edge feature storage. The i^th element stores the edge features
        of edge type i. If None, empty frame is created.  (default: None)
    multigraph : bool, optional
        Whether the graph would be a multigraph. If none, the flag will be determined
        by scanning the whole graph. (default: None)
    readonly : bool, optional
        Whether the graph structure is read-only (default: True).

    Attributes
    ----------
    TBD

    Notes
    -----
    Currently, all heterogeneous graphs are readonly.
    """
    # pylint: disable=unused-argument
    def __init__(self,
                 graph_data,
                 ntypes,
                 etypes,
                 node_frames=None,
                 edge_frames=None,
                 multigraph=None,
                 readonly=True):
        assert readonly, "Only readonly heterogeneous graphs are supported"

        self._graph = heterograph_index.create_heterograph(graph_data)
        self._nx_metagraph = None
        self._ntypes = ntypes
        self._etypes = etypes
        self._canonical_etypes = make_canonical_etypes(etypes, ntypes, self._graph.metagraph)
        # An internal map from etype to canonical etype tuple.
        # If two etypes have the same name, an empty tuple is stored instead to indicte ambiguity.
        self._etype2canonical = {}
        for i, ety in enumerate(etypes):
            if ety in self._etype2canonical:
                self._etype2canonical[ety] = tuple()
            else:
                self._etype2canonical[ety] = self._canonical_etypes[i]
        self._ntypes_invmap = {t : i for i, t in enumerate(ntypes)}
        self._etypes_invmap = {t : i for i, t in enumerate(self._canonical_etypes)}

        # node and edge frame
        if node_frames is None:
            node_frames = [
                FrameRef(Frame(num_rows=self._graph.number_of_nodes(i)))
                for i in range(len(self._ntypes))]
        self._node_frames = node_frames

        if edge_frames is None:
            edge_frames = [
                FrameRef(Frame(num_rows=self._graph.number_of_edges(i)))
                for i in range(len(self._etypes))]
        self._edge_frames = edge_frames

        # message indicators
        self._msg_indices = [None] * len(self._etypes)
        self._msg_frames = []
        for i in range(len(self._etypes)):
            frame = FrameRef(Frame(num_rows=self._graph.number_of_edges(i)))
            frame.set_initializer(init.zero_initializer)
            self._msg_frames.append(frame)

    def _get_msg_index(self, etid):
        if self._msg_indices[etid] is None:
            self._msg_indices[etid] = utils.zero_index(
                size=self._graph.number_of_edges(etid))
        return self._msg_indices[etid]

    def _set_msg_index(self, etid, index):
        self._msg_indices[etid] = index

    #################################################################
    # Mutation operations
    #################################################################

    def add_nodes(self, num, data=None, ntype=None):
        """Add multiple new nodes of the same node type

        Parameters
        ----------
        ntype : str
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
        raise DGLError('Mutation is not supported in heterograph.')

    def add_edge(self, u, v, data=None, etype=None):
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
        raise DGLError('Mutation is not supported in heterograph.')

    def add_edges(self, u, v, data=None, etype=None):
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
        raise DGLError('Mutation is not supported in heterograph.')

    #################################################################
    # Metagraph query
    #################################################################

    @property
    def ntypes(self):
        """Return the list of node types of the entire heterograph."""
        return self._ntypes

    @property
    def etypes(self):
        """Return the list of edge types of the entire heterograph."""
        return self._etypes

    @property
    def canonical_etypes(self):
        return self._canonical_etypes

    @property
    def metagraph(self):
        """Return the metagraph as networkx.MultiDiGraph.

        The nodes are labeled with node type names.
        The edges have their keys holding the edge type names.

        Returns
        -------
        networkx.MultiDiGraph
        """
        if self._nx_metagraph is None:
            nx_graph = self._graph.metagraph.to_networkx()
            self._nx_metagraph = nx.MultiDiGraph()
            for u_v in nx_graph.edges:
                srctype, etype, dsttype = self.canonical_etypes[nx_graph.edges[u_v]['id']]
                self._nx_metagraph.add_edge(srctype, dsttype, etype)
        return self._nx_metagraph

    def to_canonical_etype(self, etype):
        """Convert edge type to canonical etype: (srctype, etype, dsttype).
        
        The input can already be a canonical tuple.

        Parameters
        ----------
        etype : str or tuple of str
            Edge type

        Returns
        -------
        tuple of str
        """
        if isinstance(etype, tuple):
            return etype
        else:
            ret = self._etype2canonical.get(etype, None)
            if ret is None:
                raise DGLError('Edge type "{}" does not exist.'.format(etype))
            if len(ret) == 0:
                raise DGLError('Edge type "%s" is ambiguous. Please use canonical etype '
                               'type in the form of (srctype, etype, dsttype)' % etype)
            return ret

    def get_ntype_id(self, ntype):
        """Return the id of the given node type.

        ntype can also be None. If so, there should be only one node type in the
        graph.

        Parameters
        ----------
        ntype : str
            Node type

        Returns
        -------
        int
        """
        if ntype is None:
            if self._graph.number_of_ntypes() != 1:
                raise DGLError('Node type name must be specified if there are more than one '
                               'node types.')
            return 0
        ntid = self._ntypes_invmap.get(ntype, None)
        if ntid is None:
            raise DGLError('Node type "{}" does not exist.'.format(ntype))
        return ntid

    def get_etype_id(self, etype):
        """Return the id of the given edge type.

        etype can also be None. If so, there should be only one edge type in the
        graph.

        Parameters
        ----------
        etype : str or tuple of str
            Edge type

        Returns
        -------
        int
        """
        if etype is None:
            if self._graph.number_of_etypes() != 1:
                raise DGLError('Edge type name must be specified if there are more than one '
                               'edge types.')
            return 0
        etid = self._etypes_invmap.get(self.to_canonical_etype(etype), None)
        if etid is None:
            raise DGLError('Edge type "{}" does not exist.'.format(etype))
        return etid

    #################################################################
    # View
    #################################################################

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
        return HeteroNodeDataView(self, None, ALL)

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
        return HeteroEdgeDataView(self, None, ALL)

    def _find_etypes(self, key):
        etypes = [
            i for i, (srctype, etype, dsttype) in enumerate(self._canonical_etypes) if
            (key[0] == SLICE_FULL or key[0] == srctype) and
            (key[1] == SLICE_FULL or key[1] == etype) and
            (key[2] == SLICE_FULL or key[2] == dsttype)]
        return etypes

    def __getitem__(self, key):
        """Return the relation view of this graph.
        """
        err_msg = "Invalid slice syntax. Use G['etype'] or G['srctype', 'etype', 'dsttype'] " +\
                  "to get view of one relation type. Use ... to slice multiple types (e.g. " +\
                  "G['srctype', :, 'dsttype'])."

        if not isinstance(key, tuple):
            key = (SLICE_FULL, key, SLICE_FULL)

        etypes = self._find_etypes(key)
        if len(etypes) == 1:
            # no ambiguity: return the unitgraph itself
            srctype, etype, dsttype = self._canonical_etypes[etypes[0]]
            stid = self.get_ntype_id(srctype)
            etid = self.get_etype_id((srctype, etype, dsttype))
            dtid = self.get_ntype_id(dsttype)
            new_g = self._graph.get_relation_graph(etid)

            if stid == dtid:
                new_ntypes = [srctype]
                new_nframes = [self._node_frames[stid]]
            else:
                new_ntypes = [srctype, dsttype]
                new_nframes = [self._node_frames[stid], self._node_frames[dtid]]
            new_etypes = [etype]
            new_eframes = [self._edge_frames[etid]]

            return DGLHeteroGraph(new_g, new_ntypes, new_etypes, new_nframes, new_eframes)
        else:
            fg = self._graph.flatten_relations(etypes)
            new_g = fg.graph

            # merge frames
            stids = fg.induced_srctype_set.asnumpy()
            dtids = fg.induced_dsttype_set.asnumpy()
            etids = fg.induced_etype_set.asnumpy()
            new_ntypes = ['src', 'dst']
            new_nframes = [
                combine_frames(self._node_frames, stids),
                combine_frames(self._node_frames, dtids)]
            new_etypes = ['edge']
            new_eframes = [combine_frames(self._edge_frames, etids)]

            # create new heterograph
            new_hg = FlattenedHeteroGraph(new_g, new_ntypes, new_etypes, new_nframes, new_eframes)

            # put the parent node/edge type and IDs
            new_hg.induced_srctype = F.zerocopy_from_dgl_ndarray(fg.induced_srctype)
            new_hg.induced_srcid = F.zerocopy_from_dgl_ndarray(fg.induced_srcid)
            new_hg.induced_dsttype = F.zerocopy_from_dgl_ndarray(fg.induced_dsttype)
            new_hg.induced_dstid = F.zerocopy_from_dgl_ndarray(fg.induced_dstid)
            new_hg.induced_etype = F.zerocopy_from_dgl_ndarray(fg.induced_etype)
            new_hg.induced_eid = F.zerocopy_from_dgl_ndarray(fg.induced_eid)

            return new_hg

    #################################################################
    # Graph query
    #################################################################

    def number_of_nodes(self, ntype=None):
        """Return the number of nodes of the given type in the heterograph.

        Parameters
        ----------
        ntype : str, optional
            The node type. Can be omitted if there is only one node type
            in the graph.

        Returns
        -------
        int
            The number of nodes

        Examples
        --------
        >>> g['user'].number_of_nodes()
        3
        """
        return self._graph.number_of_nodes(self.get_ntype_id(ntype))

    def number_of_edges(self, etype=None):
        """Return the number of edges of the given type in the heterograph.

        Parameters
        ----------
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph.

        Returns
        -------
        int
            The number of edges

        Examples
        --------
        >>> g.number_of_edges(('user', 'plays', 'game'))
        4
        """
        return self._graph.number_of_edges(self.get_etype_id(etype))

    @property
    def is_multigraph(self):
        """True if the graph is a multigraph, False otherwise."""
        return self._graph.is_multigraph()

    @property
    def is_readonly(self):
        """True if the graph is readonly, False otherwise."""
        return self._graph.is_readonly()
    
    def has_node(self, vid, ntype=None):
        """Return True if the graph contains node `vid` of type `ntype`.

        Parameters
        ----------
        vid : int
            The node ID.
        ntype : str, optional
            The node type. Can be omitted if there is only one node type
            in the graph.

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
        return self._graph.has_node(self.get_ntype_id(ntype), vid)

    def has_nodes(self, vids, ntype=None):
        """Return a 0-1 array ``a`` given the node ID array ``vids``.

        ``a[i]`` is 1 if the graph contains node ``vids[i]`` of type ``ntype``, 0 otherwise.

        Parameters
        ----------
        vid : list or tensor
            The array of node IDs.
        ntype : str, optional
            The node type. Can be omitted if there is only one node type
            in the graph.

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
        rst = self._graph.has_nodes(self.get_ntype_id(ntype), vids)
        return rst.tousertensor()

    def has_edge_between(self, u, v, etype=None):
        """Return True if the edge (u, v) of type ``etype`` is in the graph.

        Parameters
        ----------
        u : int
            The node ID of source type.
        v : int
            The node ID of destination type.
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph.

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
        return self._graph.has_edge_between(self.get_etype_id(etype), u, v)

    def has_edges_between(self, u, v, etype=None):
        """Return a 0-1 array ``a`` given the source node ID array ``u`` and
        destination node ID array ``v``.

        ``a[i]`` is 1 if the graph contains edge ``(u[i], v[i])`` of type ``etype``, 0 otherwise.

        Parameters
        ----------
        u : list, tensor
            The node ID array of source type.
        v : list, tensor
            The node ID array of destination type.
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph.

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
        rst = self._graph.has_edges_between(self.get_etype_id(etype), u, v)
        return rst.tousertensor()

    def predecessors(self, v, etype=None):
        """Return the predecessors of node `v` in the graph with the same
        edge type.

        Node `u` is a predecessor of `v` if an edge `(u, v)` exist in the
        graph.

        Parameters
        ----------
        v : int
            The node of destination type.
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph.

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
        return self._graph.predecessors(self.get_etype_id(etype), v).tousertensor()

    def successors(self, v, etype=None):
        """Return the successors of node `v` in the graph with the same edge
        type.

        Node `u` is a successor of `v` if an edge `(v, u)` exist in the
        graph.

        Parameters
        ----------
        v : int
            The node of source type.
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph.

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
        return self._graph.successors(self.get_etype_id(etype), v).tousertensor()

    def edge_id(self, u, v, etype=None, force_multi=False):
        """Return the edge ID, or an array of edge IDs, between source node
        `u` and destination node `v`.

        Only works if the graph has one edge type.  For multiple types,
        query with

        Parameters
        ----------
        u : int
            The node ID of source type.
        v : int
            The node ID of destination type.
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph.
        force_multi : bool, optional
            If False, will return a single edge ID if the graph is a simple graph.
            If True, will always return an array. (Default: False)

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
        idx = self._graph.edge_id(self.get_etype_id(etype), u, v)
        return idx.tousertensor() if force_multi or self._graph.is_multigraph() else idx[0]

    def edge_ids(self, u, v, etype=None, force_multi=False):
        """Return all edge IDs between source node array `u` and destination
        node array `v`.

        Only works if the graph has one edge type.  For multiple types,
        query with

        Parameters
        ----------
        u : list, tensor
            The node ID array of source type.
        v : list, tensor
            The node ID array of destination type.
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph.
        force_multi : bool, optional
            Whether to always treat the graph as a multigraph. (Default: False)

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
        src, dst, eid = self._graph.edge_ids(self.get_etype_id(etype), u, v)
        if force_multi or self._graph.is_multigraph():
            return src.tousertensor(), dst.tousertensor(), eid.tousertensor()
        else:
            return eid.tousertensor()

    def find_edges(self, eid, etype=None):
        """Given an edge ID array, return the source and destination node ID
        array `s` and `d`.  `s[i]` and `d[i]` are source and destination node
        ID for edge `eid[i]`.

        Parameters
        ----------
        eid : list, tensor
            The edge ID array.
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph.

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
        src, dst, _ = self._graph.find_edges(self.get_etype_id(etype), eid)
        return src.tousertensor(), dst.tousertensor()

    def in_edges(self, v, etype=None, form='uv'):
        """Return the inbound edges of the node(s).

        Parameters
        ----------
        v : int, list, tensor
            The node(s) of destination type.
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph.
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
        src, dst, eid = self._graph.in_edges(self.get_etype_id(etype), v)
        if form == 'all':
            return (src.tousertensor(), dst.tousertensor(), eid.tousertensor())
        elif form == 'uv':
            return (src.tousertensor(), dst.tousertensor())
        elif form == 'eid':
            return eid.tousertensor()
        else:
            raise DGLError('Invalid form:', form)

    def out_edges(self, v, etype=None, form='uv'):
        """Return the outbound edges of the node(s).

        Parameters
        ----------
        v : int, list, tensor
            The node(s) of source type.
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph.
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
        src, dst, eid = self._graph.out_edges(self.get_etype_id(etype), v)
        if form == 'all':
            return (src.tousertensor(), dst.tousertensor(), eid.tousertensor())
        elif form == 'uv':
            return (src.tousertensor(), dst.tousertensor())
        elif form == 'eid':
            return eid.tousertensor()
        else:
            raise DGLError('Invalid form:', form)

    def all_edges(self, etype=None, form='uv', order=None):
        """Return all the edges.

        Parameters
        ----------
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph.
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
        src, dst, eid = self._graph.edges(self.get_etype_id(etype), order)
        if form == 'all':
            return (src.tousertensor(), dst.tousertensor(), eid.tousertensor())
        elif form == 'uv':
            return (src.tousertensor(), dst.tousertensor())
        elif form == 'eid':
            return eid.tousertensor()
        else:
            raise DGLError('Invalid form:', form)

    def in_degree(self, v, etype=None):
        """Return the in-degree of node ``v``.

        Parameters
        ----------
        v : int
            The node ID of destination type.
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph.

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
        return self._graph.in_degree(self.get_etype_id(etype), v)

    def in_degrees(self, v=ALL, etype=None):
        """Return the array `d` of in-degrees of the node array `v`.

        `d[i]` is the in-degree of node `v[i]`.

        Parameters
        ----------
        v : list, tensor, optional.
            The node ID array of destination type. Default is to return the
            degrees of all the nodes.
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph.

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
        etid = self.get_etype_id(etype)
        stid, dtid = self._graph.metagraph.find_edge(etid)
        if is_all(v):
            v = utils.toindex(slice(0, self._graph.number_of_nodes(dtid)))
        else:
            v = utils.toindex(v)
        return self._graph.in_degrees(etid, v).tousertensor()

    def out_degree(self, v, etype=None):
        """Return the out-degree of node `v`.

        Parameters
        ----------
        v : int
            The node ID of source type.
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph.

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
        return self._graph.out_degree(self.get_etype_id(etype), v)

    def out_degrees(self, v=ALL, etype=None):
        """Return the array `d` of out-degrees of the node array `v`.

        `d[i]` is the out-degree of node `v[i]`.

        Parameters
        ----------
        v : list, tensor
            The node ID array of source type. Default is to return the degrees
            of all the nodes.
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph.

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
        etid = self.get_etype_id(etype)
        stid, dtid = self._graph.metagraph.find_edge(etid)
        if is_all(v):
            v = utils.toindex(slice(0, self._graph.number_of_nodes(stid)))
        else:
            v = utils.toindex(v)
        return self._graph.out_degrees(etid, v).tousertensor()

    #################################################################
    # Features
    #################################################################

    def node_attr_schemes(self, ntype=None):
        """Return the node feature schemes.

        Each feature scheme is a named tuple that stores the shape and data type
        of the node feature.

        Parameters
        ----------
        ntype : str, optional
            The node type. Could be omitted if there is only one node
            type in the graph. Error will be raised otherwise.
            (Default: None)

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
        return self._node_frames[self.get_ntype_id(ntype)].schemes

    def edge_attr_schemes(self, etype=None):
        """Return the edge feature schemes.

        Each feature scheme is a named tuple that stores the shape and data type
        of the edge feature.

        Parameters
        ----------
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph.

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
        return self._edge_frames[self.get_etype_id(etype)].schemes

    def _set_n_repr(self, ntype, u, data, inplace=False):
        """Internal API to set node features.

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
        u : node, container or tensor
            The node(s).
        data : dict of tensor
            Node representation.
        inplace : bool, optional
            If True, update will be done in place, but autograd will break.
            (Default: False)
        """
        ntid = self.get_ntype_id(ntype)
        if is_all(u):
            num_nodes = self._graph.number_of_nodes(ntid)
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
                self._node_frames[ntid][key] = val
        else:
            self._node_frames[ntid].update_rows(u, data, inplace=inplace)

    def _get_n_repr(self, ntype, u):
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
        ntid = self.get_ntype_id(ntype)
        if is_all(u):
            return dict(self._node_frames[ntid])
        else:
            u = utils.toindex(u)
            return self._node_frames[ntid].select_rows(u)

    def _pop_n_repr(self, ntype, key):
        """Internal API to get and remove the specified node feature.

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
        ntid = self.get_ntype_id(ntype)
        return self._node_frames[ntid].pop(key)

    def _set_e_repr(self, etype, edges, data, inplace=False):
        """Internal API to set edge(s) features.

        `data` is a dictionary from the feature name to feature tensor. Each tensor
        is of shape (B, D1, D2, ...), where B is the number of edges to be updated,
        and (D1, D2, ...) be the shape of the edge representation tensor.

        All update will be done out of place to work with autograd unless the
        inplace flag is true.

        Parameters
        ----------
        etype : (str, str, str)
            The source-edge-destination type triplet
        edges : edges
            Edges can be either

            * A pair of endpoint nodes (u, v), where u is the node ID of source
              node type and v is that of destination node type.
            * A tensor of edge ids of the given type.

            The default value is all the edges.
        data : tensor or dict of tensor
            Edge representation.
        inplace : bool, optional
            If True, update will be done in place, but autograd will break.
            (Default: False)
        """
        etid = self.get_etype_id(etype)
        # parse argument
        if is_all(edges):
            eid = ALL
        elif isinstance(edges, tuple):
            u, v = edges
            u = utils.toindex(u)
            v = utils.toindex(v)
            # Rewrite u, v to handle edge broadcasting and multigraph.
            _, _, eid = self._graph.edge_ids(etid, u, v)
        else:
            eid = utils.toindex(edges)

        # sanity check
        if not utils.is_dict_like(data):
            raise DGLError('Expect dictionary type for feature data.'
                           ' Got "%s" instead.' % type(data))

        if is_all(eid):
            num_edges = self._graph.number_of_edges(etid)
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
                self._edge_frames[etid][key] = val
        else:
            # update row
            self._edge_frames[etid].update_rows(eid, data, inplace=inplace)

    def _get_e_repr(self, etype, edges):
        """Internal API to get edge features.

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
        etid = self.get_etype_id(etype)
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
            _, _, eid = self._graph.edge_ids(etid, u, v)
        else:
            eid = utils.toindex(edges)

        if is_all(eid):
            return dict(self._edge_frames[etid])
        else:
            eid = utils.toindex(eid)
            return self._edge_frames[etid].select_rows(eid)

    def _pop_e_repr(self, etype, key):
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
        etid = self.get_etype_id(etype)
        self._edge_frames[etid].pop(key)

    #################################################################
    # Message passing
    #################################################################
    
    def apply_nodes(self, func, v=ALL, ntype=None, inplace=False):
        """Apply the function on the nodes with the same type to update their
        features.

        If None is provided for ``func``, nothing will happen.

        Parameters
        ----------
        func : callable
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        v : int or iterable of int or tensor, optional
            The (type-specific) node (ids) on which to apply ``func``.
        ntype : str, optional
            The node type. Can be omitted if there is only one node type
            in the graph.
        inplace : bool, optional
            If True, update will be done in place, but autograd will break.

        Examples
        --------
        >>> g.ndata['h', 'user'] = torch.ones(3, 5)
        >>> g.apply_nodes(lambda nodes: {'h': nodes.data['h'] * 2}, ntype='user')
        >>> g.ndata['h', 'user']
        tensor([[2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.]])
        """
        ntid = self.get_ntype_id(ntype)
        if is_all(v):
            v_ntype = utils.toindex(slice(0, self.number_of_nodes(ntype)))
        else:
            v_ntype = utils.toindex(v)
        with ir.prog() as prog:
            scheduler.schedule_apply_nodes(v_ntype, func, self._node_frames[ntid],
                                           inplace=inplace)
            Runtime.run(prog)

    def apply_edges(self, func, edges=ALL, etype=None, inplace=False):
        """Apply the function on the edges with the same type to update their
        features.

        If None is provided for ``func``, nothing will happen.

        Parameters
        ----------
        func : callable or None
            Apply function on the edge. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        edges : edges data, optional
            Edges on which to apply ``func``. See :func:`send` for valid
            edge specification.
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.

        Examples
        --------
        >>> g.edata['h', ('user', 'plays', 'game')] = torch.ones(4, 5)
        >>> g.apply_edges(lambda edges: {'h': edges.data['h'] * 2})
        >>> g.edata['h', ('user', 'plays', 'game')]
        tensor([[2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.]])
        """
        etid = self.get_etype_id(etype)
        stid, dtid = self._graph.metagraph.find_edge(etid)
        if is_all(edges):
            u, v, _ = self._graph.edges(etid, 'eid')
            eid = utils.toindex(slice(0, self.number_of_edges(etype)))
        elif isinstance(edges, tuple):
            u, v = edges
            u = utils.toindex(u)
            v = utils.toindex(v)
            # Rewrite u, v to handle edge broadcasting and multigraph.
            u, v, eid = self._graph.edge_ids(etid, u, v)
        else:
            eid = utils.toindex(edges)
            u, v, _ = self._graph.find_edges(etid, eid)

        with ir.prog() as prog:
            scheduler.schedule_apply_edges(
                AdaptedHeteroGraph(self, stid, dtid, etid),
                u, v, eid, func, inplace=inplace)
            Runtime.run(prog)

    def group_apply_edges(self, group_by, func, edges=ALL, etype=None, inplace=False):
        """Group the edges by nodes and apply the function of the grouped
        edges to update their features.  The edges are of the same edge type
        (hence having the same source and destination node type).

        Parameters
        ----------
        group_by : str
            Specify how to group edges. Expected to be either 'src' or 'dst'
        func : callable
            Apply function on the edge.  The function should be
            an :mod:`Edge UDF <dgl.udf>`. The input of `Edge UDF` should
            be (bucket_size, degrees, *feature_shape), and
            return the dict with values of the same shapes.
        edges : edges data, optional
            Edges on which to group and apply ``func``. See :func:`send` for valid
            edges type. Default is all the edges.
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.
        """
        if group_by not in ('src', 'dst'):
            raise DGLError("Group_by should be either src or dst")

        etid = self.get_etype_id(etype)
        stid, dtid = self._graph.metagraph.find_edge(etid)
        if is_all(edges):
            u, v, _ = self._graph.edges(etid)
            eid = utils.toindex(slice(0, self.number_of_edges(etype)))
        elif isinstance(edges, tuple):
            u, v = edges
            u = utils.toindex(u)
            v = utils.toindex(v)
            # Rewrite u, v to handle edge broadcasting and multigraph.
            u, v, eid = self._graph.edge_ids(etid, u, v)
        else:
            eid = utils.toindex(edges)
            u, v, _ = self._graph.find_edges(etid, eid)

        with ir.prog() as prog:
            scheduler.schedule_group_apply_edge(
                AdaptedHeteroGraph(self, stid, dtid, etid),
                u, v, eid,
                efunc, group_by,
                inplace=inplace)
            Runtime.run(prog)

    def send(self, edges, message_func, etype=None):
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
        assert message_func is not None
        etid = self.get_etype_id(etype)
        stid, dtid = self._graph.metagraph.find_edge(etid)

        if is_all(edges):
            eid = utils.toindex(slice(0, self._graph.number_of_edges(etid)))
            u, v, _ = self._graph.edges(etid)
        elif isinstance(edges, tuple):
            u, v = edges
            u = utils.toindex(u)
            v = utils.toindex(v)
            # Rewrite u, v to handle edge broadcasting and multigraph.
            u, v, eid = self._graph.edge_ids(etid, u, v)
        else:
            eid = utils.toindex(edges)
            u, v, _ = self._graph.find_edges(etid, eid)

        if len(eid) == 0:
            # no edge to be triggered
            return

        with ir.prog() as prog:
            scheduler.schedule_send(
                AdaptedHeteroGraph(self, stid, dtid, etid),
                u, v, eid,
                message_func)
            Runtime.run(prog)

    def recv(self,
             v,
             reduce_func,
             apply_node_func=None,
             etype=None,
             inplace=False):
        """Receive and reduce incoming messages and update the features of node(s) :math:`v`.

        Two kinds of computation are supported:

        If there is only one edge type in the graph, ``G.recv(v, reduce_func, apply_func)``
        calculates:

        .. math::
            h_v^{new} = \sigma(\sum_{u\in\mathcal{N}(v)}m_{uv})

        where :math:`\mathcal{N}(v)` defines the predecessors of node(s) ``v``, and
        :math:`m_{uv}` is the message on edge (u,v). 

        * ``reduce_func`` specifies :math:`\sum`.
        * ``apply_func`` specifies :math:`\sigma`.
        
        If there are more than one edge types, ``G.recv(v, per_type_reducer,
        cross_type_reducer, apply_func)`` calculates:

        .. math::
            h_v^{new} = \sigma(\prod_{t\inT_e}\sum_{u\in\mathcal{N_t}(v)}m_{uv})

        * ``per_type_reducer`` is a dictionary from edge type to reduce functions
          :math:`\sum_{u\in\mathcal{N_t}(v)}` of each type.
        * ``cross_type_reducer`` specifies :math:`\prod_{t\inT_e}`
        * ``apply_func`` specifies :math:`\sigma`.

        Other notes:

        * `reduce_func` will be skipped for nodes with no incoming message.
        * If all ``v`` have no incoming message, this will downgrade to an :func:`apply_nodes`.
        * If some ``v`` have no incoming message, their new feature value will be calculated
          by the column initializer (see :func:`set_n_initializer`). The feature shapes and
          dtypes will be inferred.
        * The node features will be updated by the result of the ``reduce_func``.
        * Messages are consumed once received.
        * The provided UDF maybe called multiple times so it is recommended to provide
          function with no side effect.
        * The cross-type reducer will check the output field of each per-type reducer
          and aggregate those who write to the **same** fields. If None is provided,
          the default behavior is overwrite.

        Examples
        --------
        Only one type of nodes in the graph:

        >>> import dgl.function as fn
        >>> G.recv(v, fn.sum('m', 'h'))

        Specify reducer for each type and use cross-type reducer to accum results.

        >>> import dgl.function as fn
        >>> G.recv(v,
        >>> ...    {'plays' : fn.sum('m', 'h'), 'develops' : fn.max('m', 'h')},
        >>> ...    'sum')

        Error will be thrown if per-type reducers cannot determine the node type of v.

        >>> import dgl.function as fn
        >>> # ambiguous, v is of both 'user' and 'game' types
        >>> G.recv(v,
        >>> ...    {('user', 'follows', 'user') : fn.sum('m', 'h'),
        >>> ...     ('user', 'plays', 'game') : fn.max('m', 'h')},
        >>> ...    'sum')

        Parameters
        ----------
        v : int, container or tensor
            The node(s) to be updated. Default is receiving all the nodes.
        reduce_func : callable
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        apply_node_func : callable
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        etype : str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.
        """
        etid = self.get_etype_id(etype)
        stid, dtid = self._graph.metagraph.find_edge(etid)
        if is_all(v):
            v = F.arange(0, self.number_of_nodes(dtid))
        elif isinstance(v, int):
            v = [v]
        v = utils.toindex(v)
        if len(v) == 0:
            # no vertex to be triggered.
            return
        with ir.prog() as prog:
            scheduler.schedule_recv(AdaptedHeteroGraph(self, stid, dtid, etid),
                                    v, reduce_func, apply_node_func,
                                    inplace=inplace)
            Runtime.run(prog)

    def multi_recv(self, v, reducer_dict, cross_reducer, apply_func=None, inplace=False):
        # infer receive node type
        ntype = infer_ntype_from_dict(self, reducer_dict)
        ntid = self.get_ntype_id(ntype)
        if is_all(v):
            v = F.arange(0, self.number_of_nodes(ntid))
        elif isinstance(v, int):
            v = [v]
        v = utils.toindex(v)
        if len(v) == 0:
            return
        # TODO(minjie): currently loop over each edge type and reuse the old schedule.
        #   Should replace it with fused kernel.
        all_out = []
        with ir.prog() as prog:
            for ety, args in reducer_dict.items():
                outframe = FrameRef(frame_like(self._node_frames[ntid]._frame))
                args = pad_tuple(args, 2)  # (rfunc, afunc)
                if len(args) == 0:
                    raise DGLError('Invalid per-type arguments. Should be either '
                                   '(1) reduce_func or (2) (reduce_func, apply_func)')
                etid = self.get_etype_id(ety)
                stid, dtid = self._graph.metagraph.find_edge(etid)
                scheduler.schedule_recv(AdaptedHeteroGraph(self, stid, dtid, etid),
                                        v, *args,
                                        inplace=inplace, outframe=outframe)
                all_out.append(outframe)
            Runtime.run(prog)
        # merge by cross_reducer
        self._node_frames[ntid].update(merge_frames(all_out, cross_reducer))
        # apply
        if apply_func is not None:
            self.apply_nodes(apply_func, v, ntype, inplace)

    def send_and_recv(self,
                      edges,
                      message_func,
                      reduce_func,
                      apply_node_func=None,
                      etype=None,
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
        message_func : callable
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        reduce_func : callable
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        apply_node_func : callable, optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        etype : str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.
        """
        etid = self.get_etype_id(etype)
        stid, dtid = self._graph.metagraph.find_edge(etid)

        if isinstance(edges, tuple):
            u, v = edges
            u = utils.toindex(u)
            v = utils.toindex(v)
            # Rewrite u, v to handle edge broadcasting and multigraph.
            u, v, eid = self._graph.edge_ids(etid, u, v)
        else:
            eid = utils.toindex(edges)
            u, v, _ = self._graph.find_edges(etid, eid)

        if len(u) == 0:
            # no edges to be triggered
            return

        with ir.prog() as prog:
            scheduler.schedule_snr(AdaptedHeteroGraph(self, stid, dtid, etid),
                                   (u, v, eid),
                                   message_func, reduce_func, apply_node_func,
                                   inplace=inplace)
            Runtime.run(prog)

    def multi_send_and_recv(self, etype_dict, cross_reducer, apply_func=None, inplace=False):
        # infer receive node type
        ntype = infer_ntype_from_dict(self, etype_dict)
        dtid = self.get_ntype_id(ntype)

        # TODO(minjie): currently loop over each edge type and reuse the old schedule.
        #   Should replace it with fused kernel.
        all_out = []
        all_vs = []
        with ir.prog() as prog:
            for etype, args in etype_dict.items():
                etid = self.get_etype_id(etype)
                stid, _ = self._graph.metagraph.find_edge(etid)
                outframe = FrameRef(frame_like(self._node_frames[dtid]._frame))
                edges, mfunc, rfunc, afunc = pad_tuple(args, 4)
                if len(args) == 0:
                    raise DGLError('Invalid per-type arguments. Should be '
                                   '(edges, msg_func, reduce_func, [apply_func])')
                if isinstance(edges, tuple):
                    u, v = edges
                    u = utils.toindex(u)
                    v = utils.toindex(v)
                    # Rewrite u, v to handle edge broadcasting and multigraph.
                    u, v, eid = self._graph.edge_ids(etid, u, v)
                else:
                    eid = utils.toindex(edges)
                    u, v, _ = self._graph.find_edges(etid, eid)
                all_vs.append(v)
                if len(u) == 0:
                    # no edges to be triggered
                    continue
                scheduler.schedule_snr(AdaptedHeteroGraph(self, stid, dtid, etid),
                                       (u, v, eid),
                                       mfunc, rfunc, afunc,
                                       inplace=inplace, outframe=outframe)
                all_out.append(outframe)
            Runtime.run(prog)
        # merge by cross_reducer
        self._node_frames[dtid].update(merge_frames(all_out, cross_reducer))
        # apply
        if apply_func is not None:
            dstnodes = F.unique(F.cat([x.tousertensor() for x in all_vs], 0))
            self.apply_nodes(apply_func, dstnodes, ntype, inplace)

    def pull(self,
             v,
             message_func,
             reduce_func,
             apply_node_func=None,
             etype=None,
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
        message_func : callable
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        reduce_func : callable
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        apply_node_func : callable, optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        etype : str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.
        """
        # only one type of edges
        etid = self.get_etype_id(etype)
        stid, dtid = self._graph.metagraph.find_edge(etid)

        v = utils.toindex(v)
        if len(v) == 0:
            return
        with ir.prog() as prog:
            scheduler.schedule_pull(AdaptedHeteroGraph(self, stid, dtid, etid),
                                    v,
                                    message_func, reduce_func, apply_node_func,
                                    inplace=inplace)
            Runtime.run(prog)

    def multi_pull(self, v, etype_dict, cross_reducer, apply_func=None, inplace=False):
        v = utils.toindex(v)
        if len(v) == 0:
            return
        # infer receive node type
        ntype = infer_ntype_from_dict(self, etype_dict)
        dtid = self.get_ntype_id(ntype)
        # TODO(minjie): currently loop over each edge type and reuse the old schedule.
        #   Should replace it with fused kernel.
        all_out = []
        with ir.prog() as prog:
            for etype, args in etype_dict.items():
                etid = self.get_etype_id(etype)
                stid, _ = self._graph.metagraph.find_edge(etid)
                outframe = FrameRef(frame_like(self._node_frames[dtid]._frame))
                mfunc, rfunc, afunc = pad_tuple(args, 3)
                if len(args) == 0:
                    raise DGLError('Invalid per-type arguments. Should be '
                                   '(msg_func, reduce_func, [apply_func])')
                scheduler.schedule_pull(AdaptedHeteroGraph(self, stid, dtid, etid),
                                        v,
                                        mfunc, rfunc, afunc,
                                        inplace=inplace, outframe=outframe)
                all_out.append(outframe)
            Runtime.run(prog)
        # merge by cross_reducer
        self._node_frames[dtid].update(merge_frames(all_out, cross_reducer))
        # apply
        if apply_func is not None:
            self.apply_nodes(apply_func, v, ntype, inplace)

    def push(self,
             u,
             message_func,
             reduce_func,
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
        message_func : callable
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        reduce_func : callable
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        apply_node_func : callable, optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.
        """
        # only one type of edges
        etid = self.get_etype_id(None)  # must be 0
        stid, dtid = self._graph.metagraph.find_edge(etid)

        u = utils.toindex(u)
        if len(u) == 0:
            return
        with ir.prog() as prog:
            scheduler.schedule_push(AdaptedHeteroGraph(self, stid, dtid, etid),
                                    u,
                                    message_func, reduce_func, apply_node_func,
                                    inplace=inplace)
            Runtime.run(prog)

    def update_all(self,
                   message_func,
                   reduce_func,
                   apply_node_func=None,
                   etype=None):
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
        message_func : callable
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        reduce_func : callable
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        apply_node_func : callable, optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        etype : str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph.
        """
        # only one type of edges
        etid = self.get_etype_id(etype)
        stid, dtid = self._graph.metagraph.find_edge(etid)

        with ir.prog() as prog:
            scheduler.schedule_update_all(AdaptedHeteroGraph(self, stid, dtid, etid),
                                          message_func, reduce_func,
                                          apply_node_func)
            Runtime.run(prog)

    def multi_update_all(self, etype_dict, cross_reducer, apply_func=None):
        # TODO(minjie): currently loop over each edge type and reuse the old schedule.
        #   Should replace it with fused kernel.
        all_out = defaultdict(list)
        with ir.prog() as prog:
            for etype, args in etype_dict.items():
                etid = self.get_etype_id(etype)
                stid, dtid = self._graph.metagraph.find_edge(etid)
                outframe = FrameRef(frame_like(self._node_frames[dtid]._frame))
                mfunc, rfunc, afunc = pad_tuple(args, 3)
                if len(args) == 0:
                    raise DGLError('Invalid per-type arguments. Should be '
                                   '(msg_func, reduce_func, [apply_func])')
                scheduler.schedule_update_all(AdaptedHeteroGraph(self, stid, dtid, etid),
                                              mfunc, rfunc, afunc,
                                              outframe=outframe)
                all_out[dtid].append(outframe)
            Runtime.run(prog)
        for dtid, frames in all_out.items():
            # merge by cross_reducer
            self._node_frames[dtid].update(merge_frames(frames, cross_reducer))
            # apply
            if apply_func is not None:
                self.apply_nodes(apply_func, ALL, self.ntypes[dtid], inplace=False)

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

############################################################
# Internal APIs
############################################################

def make_canonical_etypes(etypes, ntypes, metagraph):
    """Internal function to convert etype name to (srctype, etype, dsttype)

    Parameters
    ----------
    etypes : list of str
        Edge type list
    ntypes : list of str
        Node type list
    metagraph : GraphIndex
        Meta graph.

    Returns
    -------
    list of tuples (srctype, etype, dsttype)
    """
    # sanity check
    if len(etypes) != metagraph.number_of_edges():
        raise DGLError('Length of edge type list must match the number of '
                       'edges in the metagraph. {} vs {}'.format(
                            len(etypes), metagraph.number_of_edges()))
    if len(ntypes) != metagraph.number_of_nodes():
        raise DGLError('Length of nodes type list must match the number of '
                       'nodes in the metagraph. {} vs {}'.format(
                            len(ntypes), metagraph.number_of_nodes()))
    rst = []
    src, dst, eid = metagraph.edges()
    for s, d, e in zip(src, dst, eid):
        rst.append((ntypes[s], etypes[e], ntypes[d]))
    return rst

def infer_ntype_from_dict(graph, etype_dict):
    """Infer node type from dictionary of edge type to values.

    All the edge types in the dict must share the same destination node type
    and the node type will be returned. Otherwise, throw error.

    Parameters
    ----------
    graph : DGLHeteroGraph
        Graph
    etype_dict : dict
        Dictionary whose key is edge type

    Returns
    -------
    str
        Node type
    """
    ntype = None
    for ety in etype_dict:
        _, _, dty = graph.to_canonical_etype(ety)
        if ntype is None:
            ntype = dty
        if ntype != dty:
            raise DGLError("Cannot infer destination node type from the dictionary. "
                           "A valid specification must make sure that all the edge "
                           "type keys share the same destination node type.")
    return ntype

def pad_tuple(tup, length, pad_val=None):
    """Pad the given tuple to the given length.

    If the input is not a tuple, convert it to a tuple of length one.
    Return an empty tuple if pad fails.
    """
    if not isinstance(tup, tuple):
        tup = (tup, )
    if len(tup) > length:
        return ()
    elif len(tup) == length:
        return tup
    else:
        return tup + (None,) * (length - len(tup))

def merge_frames(frames, reducer):
    """Merge input frames into one. Resolve conflict fields using reducer.

    Parameters
    ----------
    frames : list of FrameRef
        Input frames
    reducer : str
        One of "sum", "max", "min", "mean", "stack"

    Returns
    -------
    FrameRef
        Merged frame
    """
    if len(frames) == 1:
        return frames[0]
    if reducer == 'stack':
        def merger(flist):
            flist = [F.unsqueeze(f, 1) for f in flist]
            return F.stack(flist, 1)
    else:
        redfn = getattr(F, reducer, None)
        if redfn is None:
            raise DGLError('Invalid cross type reducer. Must be one of '
                           '"sum", "max", "min", "mean" or "stack".')
        def merger(flist):
            return redfn(F.stack(flist, 0), 0)
    ret = FrameRef(frame_like(frames[0]._frame))
    keys = set()
    for f in frames:
        keys.update(f.keys())
    for k in keys:
        flist = []
        for f in frames:
            if k in f:
                flist.append(f[k])
        if len(flist) > 1:
            ret[k] = merger(flist)
        else:
            ret[k] = flist[0]
    return ret

def combine_frames(frames, ids):
    """Merge the frames into one frame, taking the common columns.

    Parameters
    ----------
    frames : List[FrameRef]
        List of frames
    ids : List[int]
        List of frame IDs

    Returns
    -------
    FrameRef
        The resulting frame
    """
    # find common columns and check if their schemes match
    schemes = {key: scheme for key, scheme in frames[ids[0]].schemes.items()}
    for frame_id in ids:
        frame = frames[frame_id]
        for key, scheme in schemes.items():
            if key in frame.schemes:
                if frame.schemes[key] != scheme:
                    raise DGLError('Cannot concatenate column %s with shape %s and shape %s' %
                            (key, frame.schemes[key], scheme))
            else:
                del schemes[key]

    # concatenate the columns
    cols = {key: F.cat([
            frames[frame_id][key] for frame_id in ids if frames[frame_id].num_rows > 0],
            dim=0)}
    return FrameRef(Frame(cols))

class FlattenedHeteroGraph(DGLHeteroGraph):
    @property
    def induced_srctype(self):
        """Return the parent node type of source nodes in the induced unitgraph
        """
        return self._induced_srctype

    @induced_srctype.setter
    def induced_srctype(self, value):
        self._induced_srctype = value

    @property
    def induced_srcid(self):
        """Return the parent node ID of source nodes in the induced unitgraph
        """
        return self._induced_srcid

    @induced_srcid.setter
    def induced_srcid(self, value):
        self._induced_srcid = value

    @property
    def induced_srctype(self):
        """Return the parent node type of source nodes in the induced unitgraph
        """
        return self._induced_srctype

    @induced_srctype.setter
    def induced_srctype(self, value):
        self._induced_srctype = value

    @property
    def induced_srcid(self):
        """Return the parent node ID of source nodes in the induced unitgraph
        """
        return self._induced_srcid

    @induced_srcid.setter
    def induced_srcid(self, value):
        self._induced_srcid = value

    @property
    def induced_etype(self):
        """Return the parent edge type of edges in the induced unitgraph
        """
        return self._induced_etype

    @induced_etype.setter
    def induced_etype(self, value):
        self._induced_etype = value

    @property
    def induced_eid(self):
        """Return the parent edge ID of edges in the induced unitgraph
        """
        return self._induced_eid

    @induced_eid.setter
    def induced_eid(self, value):
        self._induced_eid = value

    @property
    def induced_dsttype(self):
        """Return the parent node type of destination nodes in the induced unitgraph
        """
        return self._induced_dsttype

    @induced_dsttype.setter
    def induced_dsttype(self, value):
        self._induced_dsttype = value

    @property
    def induced_dstid(self):
        """Return the parent node ID of destination nodes in the induced unitgraph
        """
        return self._induced_dstid

    @induced_dstid.setter
    def induced_dstid(self, value):
        self._induced_dstid = value

class AdaptedHeteroGraph(GraphAdapter):
    """Adapt DGLGraph to interface required by scheduler.

    Parameters
    ----------
    graph : DGLHeteroGraph
        Graph
    stid : int
        Source node type id
    dtid : int
        Destination node type id
    etid : int
        Edge type id
    """
    def __init__(self, graph, stid, dtid, etid):
        self.graph = graph
        self.stid = stid
        self.dtid = dtid
        self.etid = etid

    @property
    def gidx(self):
        return self.graph._graph

    def num_src(self):
        """Number of source nodes."""
        return self.graph._graph.number_of_nodes(self.stid)

    def num_dst(self):
        """Number of destination nodes."""
        return self.graph._graph.number_of_nodes(self.dtid)

    def num_edges(self):
        """Number of edges."""
        return self.graph._graph.number_of_edges(self.etid)

    @property
    def srcframe(self):
        """Frame to store source node features."""
        return self.graph._node_frames[self.stid]

    @property
    def dstframe(self):
        """Frame to store source node features."""
        return self.graph._node_frames[self.dtid]

    @property
    def edgeframe(self):
        """Frame to store edge features."""
        return self.graph._edge_frames[self.etid]

    @property
    def msgframe(self):
        """Frame to store messages."""
        return self.graph._msg_frames[self.etid]

    @property
    def msgindicator(self):
        """Message indicator tensor."""
        return self.graph._get_msg_index(self.etid)

    @msgindicator.setter
    def msgindicator(self, val):
        """Set new message indicator tensor."""
        self.graph._set_msg_index(self.etid, val)

    def in_edges(self, nodes):
        return self.graph._graph.in_edges(self.etid, nodes)

    def out_edges(self, nodes):
        return self.graph._graph.out_edges(self.etid, nodes)

    def edges(self, form):
        return self.graph._graph.edges(self.etid, form)

    def get_immutable_gidx(self, ctx):
        return self.graph._graph.get_unitgraph(self.etid, ctx)

    def bits_needed(self):
        return self.graph._graph.bits_needed(self.etid)
