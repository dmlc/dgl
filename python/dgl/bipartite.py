"""Bipartite graph class specialized for neural networks on graphs."""
from .graph_index import create_bigraph_index
from .heterograph import DGLHeteroGraph
from .base import ALL, is_all
from . import backend as F
from .frame import FrameRef, Frame
from .view import HeteroEdgeView, HeteroNodeView
from . import utils
from .runtime import ir, scheduler, Runtime
from .udf import NodeBatch, EdgeBatch

__all__ = ['DGLBipartiteGraph']

class DGLBipartiteGraph(DGLHeteroGraph):
    """Bipartite graph class.

    Bipartite graphs have two types of nodes and one type of edges to connect nodes.
    The graph stores nodes, edges and also their (type-specific) features.

    Parameters
    ----------
    metagraph, number_of_nodes_by_type, edge_connections_by_type :
        See DGLBaseHeteroGraph
    node_frame : dict[str, FrameRef], optional
        Node feature storage per type
    edge_frame : dict[str, FrameRef], optional
        Edge feature storage per type
    readonly : bool, optional
        Whether the graph structure is read-only (default: False)
    """
    def __init__(
            self,
            metagraph,
            number_of_nodes_by_type,
            edge_connections_by_type,
            node_frame=None,
            edge_frame=None,
            readonly=False):
        super(DGLBipartiteGraph, self).__init__(
            metagraph, number_of_nodes_by_type, edge_connections_by_type)
        if not readonly:
            raise Exception("Bipartite graph only readonly graphs for now.")

        if len(number_of_nodes_by_type) != 2:
            raise Exception("Bipartite graph should have only two node types")
        assert metagraph.number_of_edges() == 1
        self._metagraph = metagraph

        self._ntypes = list(number_of_nodes_by_type.keys())
        assert self._ntypes[0] in number_of_nodes_by_type.keys()
        assert self._ntypes[1] in number_of_nodes_by_type.keys()
        self._num_nodes_by_type = number_of_nodes_by_type
        self._num_nodes = [number_of_nodes_by_type[self._ntypes[0]],
                           number_of_nodes_by_type[self._ntypes[1]]]

        self._one_dir_graphs = {}
        self._graph = None
        self._etypes = list(edge_connections_by_type.keys())
        if len(self._etypes) > 2:
            raise Exception("Bipartite graph has at most two types of relations")
        for etype in self._etypes:
            edges = edge_connections_by_type[etype]
            if edges is not None:
                num_nodes = [number_of_nodes_by_type[etype[0]],
                             number_of_nodes_by_type[etype[1]]]
                self._one_dir_graphs[etype] = create_bigraph_index(edges, num_nodes,
                                                                   False, readonly)
            else:
                self._one_dir_graphs[etype] = None
        if len(self._etypes) == 1:
            self._graph = self._one_dir_graphs[self._etypes[0]]

        if node_frame is not None:
            assert self._ntypes[0] in node_frame
            assert self._ntypes[1] in node_frame
            assert len(node_frame) == 2
            self._node_frames = node_frame
        else:
            num_nodes = [number_of_nodes_by_type[self._ntypes[0]],
                         number_of_nodes_by_type[self._ntypes[1]]]
            self._node_frames = {self._ntypes[0]: FrameRef(Frame(num_rows=num_nodes[0])),
                                 self._ntypes[1]: FrameRef(Frame(num_rows=num_nodes[1]))}
        self._node_frame_vec = [self._node_frames[ntype] for ntype in self._ntypes]

        if edge_frame is not None:
            assert len(edge_frame) == len(self._etypes)
            for etype in self._etypes:
                assert etype in edge_frame
            self._edge_frames = edge_frame
        else:
            self._edge_frames = {}
            for etype in self._etypes:
                num_edges = self._one_dir_graphs[etype].number_of_edges()
                self._edge_frames[etype] = FrameRef(Frame(num_rows=num_edges))
        self._edge_frame_vec = [self._edge_frames[etype] for etype in self._etypes]

        # registered functions
        self._message_funcs = {}
        self._reduce_funcs = {}
        self._apply_node_funcs = {}
        self._apply_edge_funcs = {}

    def _get_node_frame(self, idx):
        return self._node_frame_vec[idx]

    def _get_edge_frame(self, idx):
        return self._edge_frame_vec[idx]

    def _number_of_nodes(self, ntype):
        if isinstance(ntype, str):
            return self._num_nodes_by_type[ntype]
        else:
            return self._num_nodes[ntype]

    def _get_graph(self, etype):
        return self._one_dir_graphs[etype]

    def _number_of_edges(self, etype):
        return self._get_graph(etype).number_of_edges()

    def number_of_nodes(self):
        """Return the number of nodes in the graph.

        Notes
        -----
        An error is raised if the graph contains multiple node types.  Use

            g[ntype].number_of_nodes()

        to get the number of nodes with type ``ntype``.

        Returns
        -------
        int
            The number of nodes
        """
        raise Exception("Bipartite graph doesn't support number_of_nodes."
                        "Please use g['ntype'].number_of_nodes() to get number of nodes.")

    def number_of_edges(self):
        """Return the number of edges in the graph.

        Returns
        -------
        int
            The number of edges
        """
        if len(self._etypes) > 1:
            raise Exception("There are more than one relation in the bipartite graph. "
                            "Please use g['srctype', 'dsttype', 'edgetype'].number_of_edges().")
        return self._graph.number_of_edges()

    def has_node(self, vid):
        """Return True if the graph contains node `vid`.
        """
        raise Exception("Bipartite graph doesn't support has_node")

    def has_nodes(self, vids):
        """Return a 0-1 array ``a`` given the node ID array ``vids``.
        """
        raise Exception("Bipartite graph doesn't support has_nodes")

    def __contains__(self, vid):
        """Return True if the graph contains node `vid`.
        """
        raise Exception("Bipartite graph doesn't support __contains__")

    def _has_node(self, ntype, vid):
        if isinstance(ntype, str):
            return 0 <= vid < self._number_of_nodes(ntype)
        else:
            raise Exception('invalid node type')

    def _has_nodes(self, ntype, vids):
        if isinstance(ntype, str):
            if isinstance(vids, list):
                vids = F.tensor(vids, F.int64)
            return (vids < self._number_of_nodes(ntype)) + (vids >= 0) > 1
        else:
            raise Exception('invalid node type')

    def has_edge_between(self, u, v):
        """Return True if the edge (u, v) is in the graph.

        Only works if the graph has one edge type.  For multiple types,
        query with

        .. code::

           g['srctype', 'dsttype', 'edgetype'].has_edge_between(u, v)

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
        """
        if len(self._etypes) > 1:
            raise Exception("There are more than one relation in the bipartite graph. "
                            "Please use g['srctype', 'dsttype', 'edgetype'].has_edge_between(u, v).")
        return self._graph.has_edge_between(u, v)

    def has_edges_between(self, u, v):
        """Return a 0-1 array `a` given the source node ID array `u` and
        destination node ID array `v`.

        `a[i]` is 1 if the graph contains edge `(u[i], v[i])`, 0 otherwise.

        Only works if the graph has one edge type.  For multiple types,
        query with

        .. code::

           g['srctype', 'dsttype', 'edgetype'].has_edges_between(u, v)

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
        """
        if len(self._etypes) > 1:
            raise Exception("There are more than one relation in the bipartite graph. "
                            "Please use g['srctype', 'dsttype', 'edgetype'].has_edges_between(u, v).")
        u = utils.toindex(u)
        v = utils.toindex(v)
        return self._graph.has_edges_between(u, v).tousertensor()

    @property
    def metagraph(self):
        """Return the metagraph as networkx.MultiDiGraph."""
        return self._metagraph

    @property
    def is_multigraph(self):
        """True if the graph is a multigraph, False otherwise.
        """
        return self._get_graph(self._etypes[0]).is_multigraph()

    @property
    def is_readonly(self):
        """True if the graph is readonly, False otherwise.
        """
        return self._get_graph(self._etypes[0]).is_readonly()

    def predecessors(self, v):
        """Return the predecessors of node `v` in the graph with the same
        edge type.

        Node `u` is a predecessor of `v` if an edge `(u, v)` exist in the
        graph.

        Parameters
        ----------
        v : int
            The node of destination type.

        Returns
        -------
        tensor
            Array of predecessor node IDs of source node type.
        """
        if len(self._etypes) > 1:
            raise Exception("There are more than one relation in the bipartite graph. "
                            "Please use g['srctype', 'dsttype', 'edgetype'].predecessors(v).")
        return self._graph.predecessors(v).tousertensor()

    def successors(self, v):
        """Return the successors of node `v` in the graph with the same edge
        type.

        Node `u` is a successor of `v` if an edge `(v, u)` exist in the
        graph.

        Parameters
        ----------
        v : int
            The node of source type.

        Returns
        -------
        tensor
            Array of successor node IDs if destination node type.
        """
        if len(self._etypes) > 1:
            raise Exception("There are more than one relation in the bipartite graph. "
                            "Please use g['srctype', 'dsttype', 'edgetype'].successors(v).")
        return self._graph.successors(v).tousertensor()

    def edge_id(self, u, v, force_multi=False):
        """Return the edge ID, or an array of edge IDs, between source node
        `u` and destination node `v`.

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
        """
        if len(self._etypes) > 1:
            raise Exception("There are more than one relation in the bipartite graph. "
                            "Please use g['srctype', 'dsttype', 'edgetype'].edge_id(u, v).")
        idx = self._graph.edge_id(u, v)
        return idx.tousertensor() if force_multi or self.is_multigraph else idx[0]

    def edge_ids(self, u, v, force_multi=False):
        """Return all edge IDs between source node array `u` and destination
        node array `v`.

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
        """
        if len(self._etypes) > 1:
            raise Exception("There are more than one relation in the bipartite graph. "
                            "Please use g['srctype', 'dsttype', 'edgetype'].edge_ids(u, v).")
        u = utils.toindex(u)
        v = utils.toindex(v)
        src, dst, eid = self._graph.edge_ids(u, v)
        if force_multi or self.is_multigraph:
            return src.tousertensor(), dst, eid.tousertensor()
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
        """
        if len(self._etypes) > 1:
            raise Exception("There are more than one relation in the bipartite graph. "
                            "Please use g['srctype', 'dsttype', 'edgetype'].find_edges(eid).")
        eid = utils.toindex(eid)
        src, dst, _ = self._graph.find_edges(eid)
        return src.tousertensor(), dst.tousertensor()

    def in_edges(self, v, form='uv'):
        """Return the inbound edges of the node(s).

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
        """
        if len(self._etypes) > 1:
            raise Exception("There are more than one relation in the bipartite graph. "
                            "Please use g['srctype', 'dsttype', 'edgetype'].in_edges(v).")
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
        """
        if len(self._etypes) > 1:
            raise Exception("There are more than one relation in the bipartite graph. "
                            "Please use g['srctype', 'dsttype', 'edgetype'].out_edges(v).")
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

    def _all_edges(self, etype, form='uv', order=None):
        src, dst, eid = self._get_graph(etype).edges(order)
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

           g['srctype', 'dsttype', 'edgetype'].all_edges()

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
        """
        if len(self._etypes) > 1:
            raise Exception("There are more than one relation in the bipartite graph. "
                            "Please use g['srctype', 'dsttype', 'edgetype'].all_edges().")
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

        Only works if the graph has one edge type.  For multiple types,
        query with

        .. code::

           g['srctype', 'dsttype', 'edgetype'].in_degree(v)

        Parameters
        ----------
        v : int
            The node ID of destination type.

        Returns
        -------
        int
            The in-degree.
        """
        if len(self._etypes) > 1:
            raise Exception("There are more than one relation in the bipartite graph. "
                            "Please use g['srctype', 'dsttype', 'edgetype'].in_degree(v).")
        return self._graph.in_degree(v)

    def in_degrees(self, v=ALL):
        """Return the array `d` of in-degrees of the node array `v`.

        `d[i]` is the in-degree of node `v[i]`.

        Only works if the graph has one edge type.  For multiple types,
        query with

        .. code::

           g['srctype', 'dsttype', 'edgetype'].in_degrees(v)

        Parameters
        ----------
        v : list, tensor, optional.
            The node ID array of destination type. Default is to return the
            degrees of all the nodes.

        Returns
        -------
        d : tensor
            The in-degree array.
        """
        if len(self._etypes) > 1:
            raise Exception("There are more than one relation in the bipartite graph. "
                            "Please use g['srctype', 'dsttype', 'edgetype'].in_degrees(v).")
        if is_all(v):
            v = utils.toindex(slice(0, self._graph.number_of_nodes(1)))
        else:
            v = utils.toindex(v)
        return self._graph.in_degrees(v).tousertensor()

    def out_degree(self, v):
        """Return the out-degree of node `v`.

        Only works if the graph has one edge type.  For multiple types,
        query with

        .. code::

           g['srctype', 'dsttype', 'edgetype'].out_degree(v)

        Parameters
        ----------
        v : int
            The node ID of source type.

        Returns
        -------
        int
            The out-degree.
        """
        if len(self._etypes) > 1:
            raise Exception("There are more than one relation in the bipartite graph. "
                            "Please use g['srctype', 'dsttype', 'edgetype'].out_degree(v).")
        return self._graph.out_degree(v)

    def out_degrees(self, v=ALL):
        """Return the array `d` of out-degrees of the node array `v`.

        `d[i]` is the out-degree of node `v[i]`.

        Only works if the graph has one edge type.  For multiple types,
        query with

        .. code::

           g['srctype', 'dsttype', 'edgetype'].out_degrees(v)

        Parameters
        ----------
        v : list, tensor
            The node ID array of source type. Default is to return the degrees
            of all the nodes.

        Returns
        -------
        d : tensor
            The out-degree array.
        """
        if len(self._etypes) > 1:
            raise Exception("There are more than one relation in the bipartite graph. "
                            "Please use g['srctype', 'dsttype', 'edgetype'].out_degrees(v).")
        if is_all(v):
            v = utils.toindex(slice(0, self._graph.number_of_nodes(0)))
        else:
            v = utils.toindex(v)
        return self._graph.out_degrees(v).tousertensor()

    def add_nodes(self, num, node_type, data=None):
        """Add multiple new nodes of the same node type

        Parameters
        ----------
        num : int
            Number of nodes to be added.
        node_type : str
            Type of the added nodes.  Must appear in the metagraph.
        data : dict, optional
            Feature data of the added nodes.
        """
        raise Exception("Bipartite graph doesn't support adding nodes")

    def add_edge(self, u, v, utype, vtype, etype, data=None):
        """Add an edge of ``etype`` between u of type ``utype`` and v of type
        ``vtype``.

        Parameters
        ----------
        u : int
            The source node ID of type ``utype``.  Must exist in the graph.
        v : int
            The destination node ID of type ``vtype``.  Must exist in the
            graph.
        utype : str
            The source node type name.  Must exist in the metagraph.
        vtype : str
            The destination node type name.  Must exist in the metagraph.
        etype : str
            The edge type name.  Must exist in the metagraph.
        data : dict, optional
            Feature data of the added edge.
        """
        raise Exception("Bipartite graph doesn't support adding edges")

    def add_edges(self, u, v, utype, vtype, etype, data=None):
        """Add multiple edges of ``etype`` between list of source nodes ``u``
        of type ``utype`` and list of destination nodes ``v`` of type
        ``vtype``.  A single edge is added between every pair of ``u[i]`` and
        ``v[i]``.

        Parameters
        ----------
        u : list, tensor
            The source node IDs of type ``utype``.  Must exist in the graph.
        v : list, tensor
            The destination node IDs of type ``vtype``.  Must exist in the
            graph.
        utype : str
            The source node type name.  Must exist in the metagraph.
        vtype : str
            The destination node type name.  Must exist in the metagraph.
        etype : str
            The edge type name.  Must exist in the metagraph.
        data : dict, optional
            Feature data of the added edge.
        """
        raise Exception("Bipartite graph doesn't support adding edges")

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
        return self._node_frames[ntype].schemes

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
        return self._edge_frames[etype].schemes

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
        self._node_frames[ntype].set_initializer(initializer, field)

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
        self._edge_frames[etype].set_initializer(initializer, field)

    @property
    def nodes(self):
        """Return a node view that can used to set/get feature data of a
        single node type.

        Notes
        -----
        An error is raised if the graph contains multiple node types.  Use

            g[ntype]

        to select nodes with type ``ntype``.
        """
        raise Exception("Bipartite graph doesn't support accessing nodes directly."
                        "Please use g[ntype].nodes.")

    @property
    def ndata(self):
        """Return the data view of all the nodes of a single node type.

        Notes
        -----
        An error is raised if the graph contains multiple node types.  Use

            g[ntype]

        to select nodes with type ``ntype``.
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
        if len(self._etypes) == 1:
            return HeteroEdgeView(self, self._etypes[0])
        else:
            raise Exception("Bipartite graph has two relations. "
                            "Please use g[src_type, dst_type, edge_type].edges.")

    @property
    def edata(self):
        """Return the data view of all the edges of a single edge type.
        """
        return self.edges[:].data

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
        if not utils.is_dict_like(data):
            raise DGLError('Expect dictionary type for feature data.'
                           ' Got "%s" instead.' % type(data))
        if is_all(u):
            num_nodes = self._number_of_nodes(ntype)
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
                self._node_frames[ntype][key] = val
        else:
            self._node_frames[ntype].update_rows(u, data, inplace=inplace)

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
        if len(self.node_attr_schemes(ntype)) == 0:
            return dict()
        if is_all(u):
            return dict(self._node_frames[ntype])
        else:
            u = utils.toindex(u)
            return self._node_frames[ntype].select_rows(u)

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
        return self._node_frames[ntype].pop(key)

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
            _, _, eid = self._get_graph(etype).edge_ids(u, v)
        else:
            eid = utils.toindex(edges)

        # sanity check
        if not utils.is_dict_like(data):
            raise DGLError('Expect dictionary type for feature data.'
                           ' Got "%s" instead.' % type(data))

        if is_all(eid):
            num_edges = self._number_of_edges(etype)
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
                self._edge_frames[etype][key] = val
        else:
            # update row
            self._edge_frames[etype].update_rows(eid, data, inplace=inplace)

    def get_e_repr(self, etype, edges=ALL):
        """Get edge(s) representation.

        Parameters
        ----------
        etype : tuple[str, str, str], optional
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
            _, _, eid = self._get_graph(etype).edge_ids(u, v)
        else:
            eid = utils.toindex(edges)

        if is_all(eid):
            return dict(self._edge_frames[etype])
        else:
            eid = utils.toindex(eid)
            return self._edge_frames[etype].select_rows(eid)

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
        return self._edge_frames[etype].pop(key)

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
        """
        assert isinstance(func, dict)
        for key in func:
            assert key in self._etypes
            self._message_funcs[key] = func[key]

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
        """
        assert isinstance(func, dict)
        for key in func:
            assert key in self._ntypes
            self._reduce_funcs[key] = func[key]

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
        """
        assert isinstance(func, dict)
        for key in func:
            assert key in self._ntypes
            self._apply_node_funcs[key] = func[key]

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
        """
        assert isinstance(func, dict)
        for key in func:
            assert key in self._etypes
            self._apply_edge_funcs[key] = func[key]

    def apply_nodes(self, func="default", v=ALL, inplace=False):
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
        if func == "default":
            func = self._apply_node_funcs
        if len(func) == 0:
            return

        assert isinstance(v, dict)
        assert isinstance(func, dict)

        for key in func:
            self[key].apply_nodes(func[key], v[key], inplace)

    def apply_edges(self, func="default", edges=ALL, inplace=False):
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
        if func == "default":
            func = self._apply_edge_funcs
        if not isinstance(func, dict):
            func = {self._etypes[0]: func}
        if len(func) == 0:
            return

        for etype in self._etypes:
            if is_all(edges):
                u, v, _ = self._get_graph(etype).edges('eid')
                eid = utils.toindex(slice(0, self._number_of_edges(etype)))
            elif isinstance(edges[etype], tuple):
                u, v = edges[etype]
                u = utils.toindex(u)
                v = utils.toindex(v)
                # Rewrite u, v to handle edge broadcasting and multigraph.
                u, v, eid = self._get_graph(etype).edge_ids(u, v)
                v = utils.toindex(v.tousertensor())
            else:
                eid = utils.toindex(edges[etype])
                u, v, _ = self._get_graph(etype).find_edges(eid)
                v = utils.toindex(v.tousertensor())

            with ir.prog() as prog:
                scheduler.schedule_bipartite_apply_edges(graph=self[etype],
                                                         u=u,
                                                         v=v,
                                                         eid=eid,
                                                         apply_func=func[etype],
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
        raise Exception("bipartite graph doesn't support group_apply_edges for now")

    def send(self, edges=ALL, message_func=None):
        """Send messages along the given edges with the same edge type.

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
        raise Exception("bipartite graph doesn't support send for now")

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
        raise Exception("bipartite graph doesn't support recv for now")

    def _get_func_iter(self, message_func, reduce_func, apply_node_func):
        if message_func == "default":
            message_func = self._message_funcs
        if reduce_func == "default":
            reduce_func = self._reduce_funcs
        if apply_node_func == "default":
            apply_node_func = self._apply_node_funcs
        if not isinstance(message_func, dict):
            assert not isinstance(reduce_func, dict)
            assert not isinstance(apply_node_func, dict)
            etype = self._etypes[0]
            message_func = {etype: message_func}
            reduce_func = {etype[1]: reduce_func}
            apply_node_func = {etype[1]: apply_node_func}

        class FuncIter(object):
            def __init__(self):
                self._idx = 0
                self._relations = list(message_func.keys())

            def __iter__(self):
                return self

            def __next__(self):
                if len(self._relations) == self._idx:
                    raise StopIteration
                relation = self._relations[self._idx]
                self._idx += 1
                mfunc = message_func[relation]
                rfunc = reduce_func[relation[1]]
                if apply_node_func is not None:
                    nfunc = apply_node_func[relation[1]]
                return mfunc, rfunc, nfunc, relation

        return FuncIter()

    def _get_edges(self, edges, etype):
        if isinstance(edges, dict):
            return edges[etype]
        else:
            return edges

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
        for mfunc, rfunc, nfunc, etype in self._get_func_iter(message_func, reduce_func,
                                                              apply_node_func):
            assert mfunc is not None
            assert rfunc is not None

            edges1 = self._get_edges(edges, etype)
            if isinstance(edges1, tuple):
                u, v = edges1
                u = utils.toindex(u)
                v = utils.toindex(v)
                # Rewrite u, v to handle edge broadcasting and multigraph.
                u, v, eid = self._get_graph(etype).edge_ids(u, v)
            else:
                eid = utils.toindex(edges1)
                u, v, _ = self._get_graph(etype).find_edges(eid)

            if len(u) == 0:
                # no edges to be triggered
                continue

            with ir.prog() as prog:
                scheduler.schedule_bipartite_snr(graph=self[etype],
                                                 edge_tuples=(u, v, eid),
                                                 message_func=mfunc,
                                                 reduce_func=rfunc,
                                                 apply_func=nfunc,
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
        for mfunc, rfunc, nfunc, etype in self._get_func_iter(message_func, reduce_func,
                                                              apply_node_func):
            assert mfunc is not None
            assert rfunc is not None

            v = utils.toindex(v)
            if len(v) == 0:
                continue
            with ir.prog() as prog:
                scheduler.schedule_bipartite_pull(graph=self[etype],
                                                  pull_nodes=v,
                                                  message_func=mfunc,
                                                  reduce_func=rfunc,
                                                  apply_func=nfunc,
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
        for mfunc, rfunc, nfunc, etype in self._get_func_iter(message_func, reduce_func,
                                                              apply_node_func):
            assert mfunc is not None
            assert rfunc is not None

            u = utils.toindex(u)
            if len(u) == 0:
                continue
            with ir.prog() as prog:
                scheduler.schedule_bipartite_push(graph=self[etype],
                                                  u=u,
                                                  message_func=mfunc,
                                                  reduce_func=rfunc,
                                                  apply_func=nfunc,
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
        for mfunc, rfunc, nfunc, etype in self._get_func_iter(message_func, reduce_func,
                                                              apply_node_func):
            assert mfunc is not None
            assert rfunc is not None

            with ir.prog() as prog:
                scheduler.schedule_bipartite_update_all(graph=self[etype],
                                                        message_func=mfunc,
                                                        reduce_func=rfunc,
                                                        apply_func=nfunc)
                Runtime.run(prog)

    # TODO should we support this?
    def prop_nodes(self,
                   nodes_generator,
                   message_func="default",
                   reduce_func="default",
                   apply_node_func="default"):
        """Node propagation in bipartite graph is not supported.
        """
        raise NotImplementedError('not supported')

    # TODO should we support this?
    def prop_edges(self,
                   edges_generator,
                   message_func="default",
                   reduce_func="default",
                   apply_node_func="default"):
        """Edge propagation in bipartite graph is not supported.
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
        raise NotImplementedError('not supported')

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
        raise NotImplementedError('not supported')

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
        raise NotImplementedError('not supported')

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
        return self._get_graph(etype).adjacency_matrix_scipy(transpose, fmt)

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
        return self._get_graph(etype).adjacency_matrix(transpose, ctx)[0]

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
        return self._get_graph(etype).incidence_matrix(typestr, ctx)[0]

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
        if is_all(nodes):
            v = utils.toindex(slice(0, self._number_of_nodes(ntype)))
        else:
            v = utils.toindex(nodes)

        n_repr = self.get_n_repr(ntype, v)
        nbatch = NodeBatch(self, v, n_repr)
        n_mask = predicate(nbatch)

        if is_all(nodes):
            return F.nonzero_1d(n_mask)
        else:
            nodes = F.tensor(nodes)
            return F.boolean_mask(nodes, n_mask)

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
        if is_all(edges):
            u, v, _ = self._get_graph(etype).edges('eid')
            eid = utils.toindex(slice(0, self._number_of_edges(etype)))
        elif isinstance(edges, tuple):
            u, v = edges
            u = utils.toindex(u)
            v = utils.toindex(v)
            # Rewrite u, v to handle edge broadcasting and multigraph.
            u, v, eid = self._get_graph(etype).edge_ids(u, v)
        else:
            eid = utils.toindex(edges)
            u, v, _ = self._get_graph(etype).find_edges(eid)

        src_data = self.get_n_repr(etype[0], u)
        edge_data = self.get_e_repr(etype, eid)
        dst_data = self.get_n_repr(etype[1], v)
        ebatch = EdgeBatch(self, (u, v, eid), src_data, edge_data, dst_data)
        e_mask = predicate(ebatch)

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
        """
        assert self.is_readonly
        assert readonly_state, "Bipartite graph doesn't support mutable graph for now."

    def __repr__(self):
        if len(self._etypes) == 1:
            ret = ('DGLBipartiteGraph(num_src_nodes={src_node},\n'
                   '                  num_dst_nodes={dst_node},\n'
                   '                  num_edges={edge},\n'
                   '                  src_ndata_schemes={src_ndata}\n'
                   '                  dst_ndata_schemes={dst_ndata}\n'
                   '                  edata_schemes={edata})')
            return ret.format(src_node=self._graph.number_of_nodes(0),
                              dst_node=self._graph.number_of_nodes(1),
                              edge=self.number_of_edges(),
                              src_ndata=str(self.node_attr_schemes(self._ntypes[0])),
                              dst_ndata=str(self.node_attr_schemes(self._ntypes[1])),
                              edata=str(self.edge_attr_schemes(self._etype)))
        else:
            ret = ('DGLBipartiteGraph(num_src_nodes={src_node},\n'
                   '                  num_dst_nodes={dst_node},\n'
                   '                  num_edges={edge},\n'
                   '                  src_ndata_schemes={src_ndata}\n'
                   '                  dst_ndata_schemes={dst_ndata}\n'
                   '                  edata_schemes1={edata1})\n'
                   '                  edata_schemes2={edata2})')
            return ret.format(src_node=self._one_dir_graphs[self._etypes[0]].number_of_nodes(0),
                              dst_node=self._one_dir_graphs[self._etypes[1]].number_of_nodes(1),
                              edge=self.number_of_edges(),
                              src_ndata=str(self.node_attr_schemes(self._ntypes[0])),
                              dst_ndata=str(self.node_attr_schemes(self._ntypes[1])),
                              edata1=str(self.edge_attr_schemes(self._etypes[0])),
                              edata2=str(self.edge_attr_schemes(self._etypes[1])))

    def _get_edge_view(self, key):
        gidx = self._get_graph(key)
        edge_frame = {key: self._edge_frames[key]}
        g = DGLBipartiteGraph(self.metagraph, self._num_nodes_by_type, {key: None},
                              self._node_frames, edge_frame, readonly=self.is_readonly)
        g._one_dir_graphs = {key: gidx}
        g._graph = gidx
        g._etypes = [key]
        assert g._ntypes[0] in g._etypes[0]
        assert g._ntypes[1] in g._etypes[0]
        return g

    def __getitem__(self, key):
        """Returns a view on the bipartite graph with given node/edge type:

        ``key`` has to be a string. It returns a subgraph induced
          from nodes of type ``key``. The induced subgraph doesn't have edges.

        If ``key`` is a pair of str (type_A, type_B) or a triplet of str
        (src_type_name, dst_type_name, edge_type_name), it returns itself if
        the key matches the metagraph of this bipartite graph or throw an exception
        if the view doesn't exist.

        The view would share the frames with the parent graph; any
        modifications on one's frames would reflect on the other.

        Parameters
        ----------
        key : str or tuple
            See above

        Returns
        -------
        DGLBaseHeteroGraphView
            The induced subgraph view.
        """
        if isinstance(key, str):
            assert key in self._num_nodes_by_type
            return DGLBipartiteGraphNodeView(self, key)
        else:
            key = tuple(key)
            if len(key) == 2:
                if key[0] == self._ntypes[0] and key[1] == self._ntypes[1]:
                    return self
                else:
                    raise Exception("The subgraph view doesn't exist")
            elif len(key) == 3:
                if key == self._etypes[0] and len(self._etypes) == 1:
                    return self
                else:
                    return self._get_edge_view(key)
            else:
                raise Exception('invalid key')

class DGLBipartiteGraphNodeView(object):
    """View on a bipartite graph, constructed from
    DGLBipartiteGraph.__getitem__().

    It is semantically the same as a subgraph.

    Parameters
    ----------
    parent : DGLBipartiteGraph
        The bipartite graph where the view is created.
    subtype : str
        The node type or edge type.
    """
    def __init__(self, parent, subtype):
        self._subtype = subtype
        self._parent = parent

    def __len__(self):
        """Return the number of nodes in the graph."""
        return self._parent._number_of_nodes(self._subtype)

    def number_of_nodes(self):
        """Return the number of nodes in the graph.
        """
        return self._parent._number_of_nodes(self._subtype)

    def number_of_edges(self):
        """Return the number of edges in the graph.
        """
        return 0

    def has_node(self, vid):
        """Return True if the graph contains node `vid`.
        """
        return self._parent._has_node(self._subtype, vid)

    def __contains__(self, vid):
        """Return True if the graph contains node `vid`.
        """
        return self._parent._has_node(self._subtype, vid)

    def has_nodes(self, vids):
        """Return a 0-1 array ``a`` given the node ID array ``vids``.
        """
        return self._parent._has_nodes(self._subtype, vids)

    @property
    def _node_frame(self):
        assert isinstance(self._subtype, str)
        return self._parent._node_frames[self._subtype]

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
    def nodes(self):
        """Return a node view that can used to set/get feature data of a
        single node type.
        """
        assert isinstance(self._subtype, str)
        return HeteroNodeView(self, self._subtype)

    def set_n_repr(self, data, ntype, u=ALL, inplace=False):
        """Set node(s) representation of a single node type.
        """
        assert ntype == self._subtype
        self._parent.set_n_repr(data, ntype, u, inplace)

    def get_n_repr(self, ntype, u=ALL):
        """Get node(s) representation of a single node type.
        """
        assert ntype == self._subtype
        return self._parent.get_n_repr(ntype, u)

    def pop_n_repr(self, ntype, key):
        """Get and remove the specified node repr of a given node type.
        """
        assert ntype == self._subtype
        self._parent.pop_n_repr(ntype, key)

    def register_reduce_func(self, func):
        """Register global message reduce function for each edge type provided.
        """
        assert isinstance(self._subtype, str)
        self._parent.register_reduce_func({self._subtype: func})

    def register_apply_node_func(self, func):
        """Register global node apply function for each node type provided.
        """
        assert isinstance(self._subtype, str)
        self._parent.register_apply_node_func({self._subtype: func})

    def register_message_func(self, func):
        """Register global message function for each edge type provided.
        """
        assert isinstance(self._subtype, tuple)
        self._parent.register_apply_message_func(func)

    def register_apply_edge_func(self, func):
        """Register global edge apply function for each edge type provided.
        """
        assert isinstance(self._subtype, tuple)
        self._parent.register_apply_edge_func(func)

    def apply_nodes(self, func="default", v=ALL, inplace=False):
        """Apply the function on the nodes with the same type to update their
        features.
        """
        if func == "default":
            assert isinstance(self._subtype, str)
            func = self._parent._apply_node_funcs[self._subtype]

        if is_all(v):
            v = utils.toindex(slice(0, self._parent._number_of_nodes(self._subtype)))
        else:
            v = utils.toindex(v)
        with ir.prog() as prog:
            scheduler.schedule_apply_nodes(graph=self,
                                           v=v,
                                           apply_func=func,
                                           inplace=inplace)
            Runtime.run(prog)
