"""Bipartite graph class specialized for neural networks on graphs."""
import numpy as np
from scipy import sparse as spsp

from .graph_index import create_bigraph_index
from .heterograph import DGLHeteroGraph
from .base import ALL, is_all
from . import backend as F
from .frame import FrameRef, Frame, Scheme
from .view import HeteroEdgeView, HeteroNodeView
from . import utils
from .runtime import ir, scheduler, Runtime

__all__ = ['DGLGraph']

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
    # pylint: disable=unused-argument
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
        self._etype = list(metagraph.edges)[0]
        self._ntypes = [self._etype[0], self._etype[1]]
        assert self._ntypes[0] in number_of_nodes_by_type.keys()
        assert self._ntypes[1] in number_of_nodes_by_type.keys()
        self._num_nodes = [number_of_nodes_by_type[self._ntypes[0]],
                           number_of_nodes_by_type[self._ntypes[1]]]
        self._num_nodes_by_type = number_of_nodes_by_type

        # TODO(zhengda) this is a hack way of constructing a bipartite graph.
        if len(edge_connections_by_type) > 1:
            raise Exception("Bipartite graph only support one type of edges")
        assert self._etype in edge_connections_by_type.keys()
        edges = edge_connections_by_type[self._etype]

        self._graph = create_bigraph_index(edges, self._num_nodes, False, readonly)

        if node_frame is not None:
            assert self._ntypes[0] in node_frame
            assert self._ntypes[1] in node_frame
            assert len(node_frame) == 2
            self._node_frames = node_frame
        else:
            self._node_frames = {self._ntypes[0]: FrameRef(Frame(num_rows=self._num_nodes[0])),
                                 self._ntypes[1]: FrameRef(Frame(num_rows=self._num_nodes[1]))}

        if edge_frame is not None:
            assert self._etype in edge_frame
            assert len(edge_frame) == 1
            self._edge_frames = edge_frame
        else:
            num_edges = self._graph.number_of_edges()
            self._edge_frames = {self._etype: FrameRef(Frame(num_rows=num_edges))}

        # registered functions
        self._message_funcs = {}
        self._reduce_funcs = {}
        self._apply_node_funcs = {}
        self._apply_edge_funcs = {}

    def _get_node_frame(self, idx):
        return self._node_frames[self._ntypes[idx]]

    def _get_edge_frame(self, idx):
        return self._edge_frames[self._etype]

    def _number_of_nodes(self, ntype):
        if isinstance(ntype, str):
            return self._num_nodes_by_type[ntype]
        elif isinstance(ntype, int):
            return self._num_nodes[ntype]
        elif isinstance(ntype, tuple):
            raise Exception("we can't get the number of nodes of different types")
        else:
            raise Exception('Wrong input type')

    def number_of_nodes(self):
        raise Exception("Bipartite graph doesn't support number_of_nodes."
                        "Please use g['ntype'].number_of_nodes() to get number of nodes.")

    def _number_of_edges(self, etype):
        assert etype == self._etype
        return self._graph.number_of_edges()

    def number_of_edges(self):
        return self._graph.number_of_edges()

    def _has_node(self, ntype, vid):
        if isinstance(ntype, str):
            return vid < self._number_of_nodes(ntype) and vid >= 0
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
        return self._graph.has_edge_between(u, v + self._number_of_nodes(0))

    def has_edges_between(self, u, v):
        u = utils.toindex(u)
        v = F.tensor(v, F.int64) + self._number_of_nodes(0)
        v = utils.toindex(v)
        return self._graph.has_edges_between(u, v).tousertensor()

    @property
    def is_multigraph(self):
        """True if the graph is a multigraph, False otherwise.
        """
        return self._graph.is_multigraph()

    def predecessors(self, v):
        return self._graph.predecessors(v + self._number_of_nodes(0)).tousertensor()

    def successors(self, v):
        return self._graph.successors(v).tousertensor() - self._number_of_nodes(0)

    def edge_id(self, u, v, force_multi=False):
        idx = self._graph.edge_id(u, v + self._number_of_nodes(0))
        return idx.tousertensor() if force_multi or self.is_multigraph else idx[0]

    def _to_dst_index(self, v):
        if isinstance(v, int):
            v = v + self._number_of_nodes(0)
        elif isinstance(v, list):
            v = F.tensor(v, F.int64) + self._number_of_nodes(0)
        elif isinstance(v, utils.Index):
            v = v.tousertensor() + self._number_of_nodes(0)
        else:
            v = v + self._number_of_nodes(0)
        return utils.toindex(v)

    def edge_ids(self, u, v, force_multi=False):
        u = utils.toindex(u)
        v = self._to_dst_index(v)
        src, dst, eid = self._graph.edge_ids(u, v)
        if force_multi or self.is_multigraph:
            dst = dst.tousertensor() - self._number_of_nodes(0)
            return src.tousertensor(), dst, eid.tousertensor()
        else:
            return eid.tousertensor()

    def find_edges(self, eid):
        eid = utils.toindex(eid)
        src, dst, _ = self._graph.find_edges(eid)
        return src.tousertensor(), dst.tousertensor() - self._number_of_nodes(0)

    def in_edges(self, v, form='uv'):
        v = self._to_dst_index(v)
        src, dst, eid = self._graph.in_edges(v)
        if form == 'all':
            dst = dst.tousertensor() - self._number_of_nodes(0)
            return (src.tousertensor(), dst, eid.tousertensor())
        elif form == 'uv':
            return (src.tousertensor(), dst.tousertensor() - self._number_of_nodes(0))
        elif form == 'eid':
            return eid.tousertensor()
        else:
            raise DGLError('Invalid form:', form)

    def out_edges(self, v, form='uv'):
        v = utils.toindex(v)
        src, dst, eid = self._graph.out_edges(v)
        if form == 'all':
            dst = dst.tousertensor() - self._number_of_nodes(0)
            return (src.tousertensor(), dst, eid.tousertensor())
        elif form == 'uv':
            return (src.tousertensor(), dst.tousertensor() - self._number_of_nodes(0))
        elif form == 'eid':
            return eid.tousertensor()
        else:
            raise DGLError('Invalid form:', form)

    def _all_edges(self, etype, form='uv', order=None):
        assert etype == self._etype
        return self.all_edges(form, order)

    def all_edges(self, form='uv', order=None):
        src, dst, eid = self._graph.edges(order)
        if form == 'all':
            dst = dst.tousertensor() - self._number_of_nodes(0)
            return (src.tousertensor(), dst, eid.tousertensor())
        elif form == 'uv':
            return (src.tousertensor(), dst.tousertensor() - self._number_of_nodes(0))
        elif form == 'eid':
            return eid.tousertensor()
        else:
            raise DGLError('Invalid form:', form)

    def in_degree(self, v):
        v = v + self._number_of_nodes(0)
        return self._graph.in_degree(v)

    def in_degrees(self, v=ALL):
        if is_all(v):
            v = utils.toindex(slice(self._number_of_nodes(0),
                                    self._number_of_nodes(0) + self._number_of_nodes(1)))
        else:
            v = utils.toindex(self._to_dst_index(v))
        return self._graph.in_degrees(v).tousertensor()

    def out_degree(self, v):
        return self._graph.out_degree(v)

    def out_degrees(self, v=ALL):
        if is_all(v):
            v = utils.toindex(slice(0, self._number_of_nodes(0)))
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

        Examples
        --------
        The variable ``g`` is constructed from the example in
        DGLBaseHeteroGraph.

        >>> g['user', 'game', 'plays'].number_of_edges()
        4
        >>> g.add_edge(2, 0, 'user', 'game', 'plays')
        >>> g['user', 'game', 'plays'].number_of_edges()
        5
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

        Examples
        --------
        The variable ``g`` is constructed from the example in
        DGLBaseHeteroGraph.

        >>> g['user', 'game', 'plays'].number_of_edges()
        4
        >>> g.add_edges([0, 2], [1, 0], 'user', 'game', 'plays')
        >>> g['user', 'game', 'plays'].number_of_edges()
        6
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
        return HeteroNodeView(self)

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
        """
        return HeteroEdgeView(self, self._etype)

    @property
    def edata(self):
        """Return the data view of all the edges of a single edge type.

        Notes
        -----
        An error is raised if the graph contains multiple edge types.  Use

            g[src_type, dst_type, edge_type]

        to select edges with type ``(src_type, dst_type, edge_type)``.
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
            v = v + self.number_of_nodes(0)
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
                self._edge_frames[etype][key] = val
        else:
            # update row
            self._edge_frames[etype].update_rows(eid, data, inplace=inplace)

    def get_e_repr(self, etype=ALL, edges=ALL):
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
        if is_all(etype):
            etype = self._etype

        if len(self.edge_attr_schemes(self._etype)) == 0:
            return dict()
        # parse argument
        if is_all(edges):
            eid = ALL
        elif isinstance(edges, tuple):
            u, v = edges
            v = v + self.number_of_nodes(0)
            u = utils.toindex(u)
            v = utils.toindex(v)
            # Rewrite u, v to handle edge broadcasting and multigraph.
            _, _, eid = self._graph.edge_ids(u, v)
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
        if isinstance(func, dict):
            assert len(func) == 1
            key = list(func.keys())[0]
            assert key == self._etype
            self._message_funcs[key] = func[key]
        else:
            self._message_funcs[self._etype] = func

    def register_reduce_func(self, func):
        assert isinstance(func, dict)
        for key in func:
            assert key in self._ntypes
            self._reduce_funcs[key] = func[key]

    def register_apply_node_func(self, func):
        assert isinstance(func, dict)
        for key in func:
            assert key in self._ntypes
            self._apply_node_funcs[key] = func[key]

    def register_apply_edge_func(self, func):
        if isinstance(func, dict):
            assert len(func) == 1
            key = list(func.keys())[0]
            assert key == self._etype
            self._apply_edge_funcs[key] = func[key]
        else:
            self._apply_edge_funcs[self._etype] = func

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
        assert isinstance(v, dict)
        assert isinstance(func, dict)

        for ntype, varr in v:
            if is_all(varr):
                varr = utils.toindex(slice(0, self.number_of_nodes(ntype)))
            else:
                varr = utils.toindex(varr)
            assert ntype in func
            with ir.prog() as prog:
                scheduler.schedule_heterograph_apply_nodes(graph=self,
                                                           ntype=ntype,
                                                           v=varr,
                                                           apply_func=func[ntype],
                                                           inplace=inplace)
                Runtime.run(prog)

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
            func = self._apply_edge_funcs[self._etype]
        assert func is not None

        # Bipartite graphs have only one edge type.
        if is_all(edges):
            u, v, _ = self._graph.edges('eid')
            v = utils.toindex(v.tousertensor() - self._number_of_nodes(0))
            eid = utils.toindex(slice(0, self.number_of_edges()))
        elif isinstance(edges, tuple):
            u, v = edges
            u = utils.toindex(u)
            v = utils.toindex(v + self._number_of_nodes(0))
            # Rewrite u, v to handle edge broadcasting and multigraph.
            u, v, eid = self._graph.edge_ids(u, v)
            v = utils.toindex(v.tousertensor() - self._number_of_nodes(0))
        else:
            eid = utils.toindex(edges)
            u, v, _ = self._graph.find_edges(eid)
            v = utils.toindex(v.tousertensor() - self._number_of_nodes(0))

        with ir.prog() as prog:
            scheduler.schedule_bipartite_apply_edges(graph=self,
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

    def _get_func(self, message_func, reduce_func, apply_node_func):
        if message_func == "default":
            if self._etype in self._message_funcs:
                message_func = self._message_funcs[self._etype]
            else:
                message_func = None
        if reduce_func == "default":
            if self._ntypes[1] in self._reduce_funcs:
                reduce_func = self._reduce_funcs[self._ntypes[1]]
            else:
                reduce_func = None
        if apply_node_func == "default":
            if self._ntypes[1] in self._apply_node_funcs:
                apply_node_func = self._apply_node_funcs[self._ntypes[1]]
            else:
                apply_node_func = None
        return message_func, reduce_func, apply_node_func

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
        message_func, reduce_func, apply_node_func = self._get_func(message_func, reduce_func,
                                                                    apply_node_func)
        assert message_func is not None
        assert reduce_func is not None

        if isinstance(edges, tuple):
            u, v = edges
            u = utils.toindex(u)
            gv = self._to_dst_index(v)
            # Rewrite u, v to handle edge broadcasting and multigraph.
            u, gv, eid = self._graph.edge_ids(u, gv)
        else:
            eid = utils.toindex(edges)
            u, gv, _ = self._graph.find_edges(eid)

        if len(u) == 0:
            # no edges to be triggered
            return

        v = utils.toindex(gv.tousertensor() - self._number_of_nodes(0))
        with ir.prog() as prog:
            scheduler.schedule_bipartite_snr(graph=self,
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
        message_func, reduce_func, apply_node_func = self._get_func(message_func, reduce_func,
                                                                    apply_node_func)
        assert message_func is not None
        assert reduce_func is not None

        v = utils.toindex(v)
        if len(v) == 0:
            return
        with ir.prog() as prog:
            scheduler.schedule_bipartite_pull(graph=self,
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
        message_func, reduce_func, apply_node_func = self._get_func(message_func, reduce_func,
                                                                    apply_node_func)
        assert message_func is not None
        assert reduce_func is not None

        with ir.prog() as prog:
            scheduler.schedule_bipartite_update_all(graph=self,
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
        assert etype == self._etype
        return self._graph.adjacency_matrix_scipy(transpose, fmt)

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
        assert etype == self._etype
        return self._graph.adjacency_matrix(transpose, ctx)[0]

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
        assert etype == self._etype
        return self._graph.incidence_matrix(typestr, ctx)[0]

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
        pass

    def __getitem__(self, key):
        """Returns a view on the heterogeneous graph with given node/edge
        type:

        * If ``key`` is a str, it returns a heterogeneous subgraph induced
          from nodes of type ``key``.
        * If ``key`` is a pair of str (type_A, type_B), it returns a
          heterogeneous subgraph induced from the union of both node types.
        * If ``key`` is a triplet of str

              (src_type_name, dst_type_name, edge_type_name)

          It returns a heterogeneous subgraph induced from the edges with
          source type name ``src_type_name``, destination type name
          ``dst_type_name``, and edge type name ``edge_type_name``.

        The view would share the frames with the parent graph; any
        modifications on one's frames would reflect on the other.

        Note that the subgraph itself is not materialized until someone
        queries the subgraph structure.  This implies that calling computation
        methods such as

            g['user'].update_all(...)

        would not create a subgraph of users.

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
            return DGLBipartiteGraphView(self, key)
        else:
            key = tuple(key)
            if len(key) == 2:
                if key[0] == self._ntypes[0] and key[1] == self._ntypes[1]:
                    return self
                else:
                    raise Exception("The subgraph view doesn't exist")
            elif len(key) == 3:
                if key[0] == self._ntypes[0] and key[1] == self._ntypes[1] \
                   and key[2] == self._etype:
                    return self
                else:
                    raise Exception("The subgraph view doesn't exist")
            else:
                raise Exception('invalid key')

class DGLBipartiteGraphView(object):
    def __init__(self, parent, subtype):
        self._subtype = subtype
        self._parent = parent

    def __len__(self):
        """Return the number of nodes in the graph."""
        return self._parent._number_of_nodes(self._subtype)

    def number_of_nodes(self):
        return self._parent._number_of_nodes(self._subtype)

    def number_of_edges(self):
        return 0

    def has_node(self, vid):
        return self._parent._has_node(self._subtype, vid)

    def __contains__(self, vid):
        return self._parent._has_node(self._subtype, vid)

    def has_nodes(self, vids):
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
        assert isinstance(self._subtype, str)
        return HeteroNodeView(self, self._subtype)

    def set_n_repr(self, data, ntype, u=ALL, inplace=False):
        assert ntype == self._subtype
        self._parent.set_n_repr(data, ntype, u, inplace)

    def get_n_repr(self, ntype, u=ALL):
        assert ntype == self._subtype
        return self._parent.get_n_repr(ntype, u)

    def pop_n_repr(self, ntype, key):
        assert ntype == self._subtype
        self._parent.pop_n_repr(ntype, key)

    def register_reduce_func(self, func):
        assert isinstance(self._subtype, str)
        self._parent.register_reduce_func({self._subtype: func})

    def register_apply_node_func(self, func):
        assert isinstance(self._subtype, str)
        self._parent.register_apply_node_func({self._subtype: func})

    def register_message_func(self, func):
        assert isinstance(self._subtype, tuple)
        self._parent.register_apply_message_func(func)

    def register_apply_edge_func(self, func):
        assert isinstance(self._subtype, tuple)
        self._parent.register_apply_edge_func(func)

    def apply_nodes(self, func="default", v=ALL, inplace=False):
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
