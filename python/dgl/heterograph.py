"""Classes for heterogeneous graphs."""
#pylint: disable= too-many-lines
from collections import defaultdict
from collections.abc import Mapping
from contextlib import contextmanager
import copy
import numbers
import networkx as nx
import numpy as np

from ._ffi.function import _init_api
from .base import ALL, SLICE_FULL, NTYPE, NID, ETYPE, EID, is_all, DGLError, dgl_warning
from . import core
from . import graph_index
from . import heterograph_index
from . import utils
from . import backend as F
from .frame import Frame
from .view import HeteroNodeView, HeteroNodeDataView, HeteroEdgeView, HeteroEdgeDataView

__all__ = ['DGLHeteroGraph', 'combine_names']

class DGLHeteroGraph(object):
    """Base heterogeneous graph class.

    **Do NOT instantiate from this class directly; use** :mod:`conversion methods
    <dgl.convert>` **instead.**

    A Heterogeneous graph is defined as a graph with node types and edge
    types.

    If two edges share the same edge type, then their source nodes, as well
    as their destination nodes, also have the same type (the source node
    types don't have to be the same as the destination node types).

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

    And suppose that one maps the users, games and developers to the following
    IDs:

    =========  =====  ===  =====
    User name  Alice  Bob  Carol
    =========  =====  ===  =====
    User ID    0      1    2
    =========  =====  ===  =====

    =========  ======  =========
    Game name  Tetris  Minecraft
    =========  ======  =========
    Game ID    0       1
    =========  ======  =========

    ==============  ========  ======
    Developer name  Nintendo  Mojang
    ==============  ========  ======
    Developer ID    0         1
    ==============  ========  ======

    One can construct the graph as follows:

    >>> follows_g = dgl.graph(([0, 1], [1, 2]), 'user', 'follows')
    >>> plays_g = dgl.bipartite(([0, 1, 1, 2], [0, 0, 1, 1]), 'user', 'plays', 'game')
    >>> devs_g = dgl.bipartite(([0, 1], [0, 1]), 'developer', 'develops', 'game')
    >>> g = dgl.hetero_from_relations([follows_g, plays_g, devs_g])

    Or equivalently

    >>> g = dgl.heterograph({
    ...     ('user', 'follows', 'user'): ([0, 1], [1, 2]),
    ...     ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 1, 1]),
    ...     ('developer', 'develops', 'game'): ([0, 1], [0, 1]),
    ...     })

    :func:`dgl.graph` and :func:`dgl.bipartite` can create a graph from a variety of
    data types including:

    * edge list
    * edge tuples
    * networkx graph
    * scipy sparse matrix

    Click the function names for more details.

    Then one can query the graph structure by specifying the ``ntype`` or ``etype`` arguments:

    >>> g.number_of_nodes('user')
    3
    >>> g.number_of_edges('plays')
    4
    >>> g.out_degrees(etype='develops')  # out-degrees of source nodes of 'develops' relation
    tensor([1, 1])
    >>> g.in_edges(0, etype='develops')  # in-edges of destination node 0 of 'develops' relation
    (tensor([0]), tensor([0]))

    Or on the sliced graph for an edge type:

    >>> g['plays'].number_of_edges()
    4
    >>> g['develops'].out_degrees()
    tensor([1, 1])
    >>> g['develops'].in_edges(0)
    (tensor([0]), tensor([0]))

    Node type names must be distinct (no two types have the same name). Edge types could
    have the same name but they must be distinguishable by the ``(src_type, edge_type, dst_type)``
    triplet (called *canonical edge type*).

    For example, suppose a graph that has two types of relation "user-watches-movie"
    and "user-watches-TV" as follows:

    >>> g0 = dgl.bipartite(([0, 1, 1], [1, 0, 1]), 'user', 'watches', 'movie')
    >>> g1 = dgl.bipartite(([0, 1], [0, 1]), 'user', 'watches', 'TV')
    >>> GG = dgl.hetero_from_relations([g0, g1]) # Merge the two graphs

    To distinguish between the two "watches" edge type, one must specify a full triplet:

    >>> GG.number_of_edges(('user', 'watches', 'movie'))
    3
    >>> GG.number_of_edges(('user', 'watches', 'TV'))
    2
    >>> GG['user', 'watches', 'movie'].out_degrees()
    tensor([1, 2])

    Using only one single edge type string "watches" is ambiguous and will cause error:

    >>> GG.number_of_edges('watches')  # AMBIGUOUS!!

    In many cases, there is only one type of nodes or one type of edges, and the ``ntype``
    and ``etype`` argument could be omitted. This is very common when using the sliced
    graph, which usually contains only one edge type, and sometimes only one node type:

    >>> g['follows'].number_of_nodes()  # OK!! because g['follows'] only has one node type 'user'
    3
    >>> g['plays'].number_of_nodes()  # ERROR!! There are two types 'user' and 'game'.
    >>> g['plays'].number_of_edges()  # OK!! because there is only one edge type 'plays'

    TODO(minjie): docstring about uni-directional bipartite graph

    Metagraph
    ---------
    For each heterogeneous graph, one can often infer the *metagraph*, the template of
    edge connections showing how many types of nodes and edges exist in the graph, and
    how each edge type could connect between node types.

    One can analyze the example gameplay graph above and figure out the metagraph as
    follows:

    .. graphviz::

       digraph G {
           User -> User [label=follows]
           User -> Game [label=plays]
           Developer -> Game [label=develops]
       }


    Parameters
    ----------
    gidx : HeteroGraphIndex
        Graph index object.
    ntypes : list of str, pair of list of str
        Node type list. ``ntypes[i]`` stores the name of node type i.
        If a pair is given, the graph created is a uni-directional bipartite graph,
        and its SRC node types and DST node types are given as in the pair.
    etypes : list of str
        Edge type list. ``etypes[i]`` stores the name of edge type i.
    node_frames : list[Frame], optional
        Node feature storage. If None, empty frame is created.
        Otherwise, ``node_frames[i]`` stores the node features
        of node type i. (default: None)
    edge_frames : list[Frame], optional
        Edge feature storage. If None, empty frame is created.
        Otherwise, ``edge_frames[i]`` stores the edge features
        of edge type i. (default: None)
    """
    is_block = False

    # pylint: disable=unused-argument, dangerous-default-value
    def __init__(self,
                 gidx=[],
                 ntypes=['_U'],
                 etypes=['_V'],
                 node_frames=None,
                 edge_frames=None,
                 **deprecate_kwargs):
        if isinstance(gidx, DGLHeteroGraph):
            raise DGLError('The input is already a DGLGraph. No need to create it again.')
        if not isinstance(gidx, heterograph_index.HeteroGraphIndex):
            dgl_warning('Recommend creating graphs by `dgl.graph(data)`'
                        ' instead of `dgl.DGLGraph(data)`.')
            u, v, num_src, num_dst = utils.graphdata2tensors(gidx)
            gidx = heterograph_index.create_unitgraph_from_coo(
                1, num_src, num_dst, u, v, ['coo', 'csr', 'csc'])
        if len(deprecate_kwargs) != 0:
            dgl_warning('Keyword arguments {} are deprecated in v0.5, and can be safely'
                        ' removed in all cases.'.format(list(deprecate_kwargs.keys())))
        self._init(gidx, ntypes, etypes, node_frames, edge_frames)

    def _init(self, gidx, ntypes, etypes, node_frames, edge_frames):
        """Init internal states."""
        self._graph = gidx
        self._canonical_etypes = None
        self._batch_num_nodes = None
        self._batch_num_edges = None

        # Handle node types
        if isinstance(ntypes, tuple):
            if len(ntypes) != 2:
                errmsg = 'Invalid input. Expect a pair (srctypes, dsttypes) but got {}'.format(
                    ntypes)
                raise TypeError(errmsg)
            if not is_unibipartite(self._graph.metagraph):
                raise ValueError('Invalid input. The metagraph must be a uni-directional'
                                 ' bipartite graph.')
            self._ntypes = ntypes[0] + ntypes[1]
            self._srctypes_invmap = {t : i for i, t in enumerate(ntypes[0])}
            self._dsttypes_invmap = {t : i + len(ntypes[0]) for i, t in enumerate(ntypes[1])}
            self._is_unibipartite = True
            if len(ntypes[0]) == 1 and len(ntypes[1]) == 1 and len(etypes) == 1:
                self._canonical_etypes = [(ntypes[0][0], etypes[0], ntypes[1][0])]
        else:
            self._ntypes = ntypes
            if len(ntypes) == 1:
                src_dst_map = None
            else:
                src_dst_map = find_src_dst_ntypes(self._ntypes, self._graph.metagraph)
            self._is_unibipartite = (src_dst_map is not None)
            if self._is_unibipartite:
                self._srctypes_invmap, self._dsttypes_invmap = src_dst_map
            else:
                self._srctypes_invmap = {t : i for i, t in enumerate(self._ntypes)}
                self._dsttypes_invmap = self._srctypes_invmap

        # Handle edge types
        self._etypes = etypes
        if self._canonical_etypes is None:
            if (len(etypes) == 1 and len(ntypes) == 1):
                self._canonical_etypes = [(ntypes[0], etypes[0], ntypes[0])]
            else:
                self._canonical_etypes = make_canonical_etypes(
                    self._etypes, self._ntypes, self._graph.metagraph)

        # An internal map from etype to canonical etype tuple.
        # If two etypes have the same name, an empty tuple is stored instead to indicate
        # ambiguity.
        self._etype2canonical = {}
        for i, ety in enumerate(self._etypes):
            if ety in self._etype2canonical:
                self._etype2canonical[ety] = tuple()
            else:
                self._etype2canonical[ety] = self._canonical_etypes[i]
        self._etypes_invmap = {t : i for i, t in enumerate(self._canonical_etypes)}

        # node and edge frame
        if node_frames is None:
            node_frames = [None] * len(self._ntypes)
        node_frames = [Frame(num_rows=self._graph.number_of_nodes(i))
                       if frame is None else frame
                       for i, frame in enumerate(node_frames)]
        self._node_frames = node_frames

        if edge_frames is None:
            edge_frames = [None] * len(self._etypes)
        edge_frames = [Frame(num_rows=self._graph.number_of_edges(i))
                       if frame is None else frame
                       for i, frame in enumerate(edge_frames)]
        self._edge_frames = edge_frames

    def __setstate__(self, state):
        # Compatibility check
        # TODO: version the storage
        if isinstance(state, dict):
            # Since 0.5 we use the default __dict__ method
            self.__dict__.update(state)
        elif isinstance(state, tuple) and len(state) == 5:
            # DGL == 0.4.3
            dgl_warning("The object is pickled with DGL == 0.4.3.  "
                        "Some of the original attributes are ignored.")
            self._init(*state)
        elif isinstance(state, dict):
            # DGL <= 0.4.2
            dgl_warning("The object is pickled with DGL <= 0.4.2.  "
                        "Some of the original attributes are ignored.")
            self._init(state['_graph'], state['_ntypes'], state['_etypes'], state['_node_frames'],
                       state['_edge_frames'])
        else:
            raise IOError("Unrecognized pickle format.")

    def __repr__(self):
        if len(self.ntypes) == 1 and len(self.etypes) == 1:
            ret = ('Graph(num_nodes={node}, num_edges={edge},\n'
                   '      ndata_schemes={ndata}\n'
                   '      edata_schemes={edata})')
            return ret.format(node=self.number_of_nodes(), edge=self.number_of_edges(),
                              ndata=str(self.node_attr_schemes()),
                              edata=str(self.edge_attr_schemes()))
        else:
            ret = ('Graph(num_nodes={node},\n'
                   '      num_edges={edge},\n'
                   '      metagraph={meta})')
            nnode_dict = {self.ntypes[i] : self._graph.number_of_nodes(i)
                          for i in range(len(self.ntypes))}
            nedge_dict = {self.canonical_etypes[i] : self._graph.number_of_edges(i)
                          for i in range(len(self.etypes))}
            meta = str(self.metagraph().edges(keys=True))
            return ret.format(node=nnode_dict, edge=nedge_dict, meta=meta)

    def __copy__(self):
        """Shallow copy implementation."""
        #TODO(minjie): too many states in python; should clean up and lower to C
        cls = type(self)
        obj = cls.__new__(cls)
        obj.__dict__.update(self.__dict__)
        return obj

    #################################################################
    # Mutation operations
    #################################################################

    def add_nodes(self, num, data=None, ntype=None):
        r"""Add new nodes of the same node type

        Parameters
        ----------
        num : int
            Number of nodes to add.
        data : dict, optional
            Feature data of the added nodes.
        ntype : str, optional
            The type of the new nodes. Can be omitted if there is
            only one node type in the graph.

        Notes
        -----

        * Inplace update is applied to the current graph.
        * If the key of ``data`` does not contain some existing feature fields,
        those features for the new nodes will be created by initializers
        defined with :func:`set_n_initializer` (default initializer fills zeros).
        * If the key of ``data`` contains new feature fields, those features for
        the old nodes will be created by initializers defined with
        :func:`set_n_initializer` (default initializer fills zeros).

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        **Homogeneous Graphs or Heterogeneous Graphs with A Single Node Type**

        >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
        >>> g.num_nodes()
        3
        >>> g.add_nodes(2)
        >>> g.num_nodes()
        5

        If the graph has some node features and new nodes are added without
        features, their features will be created by initializers defined
        with :func:`set_n_initializer`.

        >>> g.ndata['h'] = torch.ones(5, 1)
        >>> g.add_nodes(1)
        >>> g.ndata['h']
        tensor([[1.], [1.], [1.], [1.], [1.], [0.]])

        We can also assign features for the new nodes in adding new nodes.

        >>> g.add_nodes(1, {'h': torch.ones(1, 1), 'w': torch.ones(1, 1)})
        >>> g.ndata['h']
        tensor([[1.], [1.], [1.], [1.], [1.], [0.], [1.]])

        Since ``data`` contains new feature fields, the features for old nodes
        will be created by initializers defined with :func:`set_n_initializer`.

        >>> g.ndata['w']
        tensor([[0.], [0.], [0.], [0.], [0.], [0.], [1.]])


        **Heterogeneous Graphs with Multiple Node Types**

        >>> g = dgl.heterograph({
        >>>     ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
        >>>                                 torch.tensor([0, 0, 1, 1])),
        >>>     ('developer', 'develops', 'game'): (torch.tensor([0, 1]),
        >>>                                         torch.tensor([0, 1]))
        >>>     })
        >>> g.add_nodes(2)
        DGLError: Node type name must be specified
        if there are more than one node types.
        >>> g.num_nodes('user')
        3
        >>> g.add_nodes(2, ntype='user')
        >>> g.num_nodes('user')
        5

        See Also
        --------
        remove_nodes
        add_edges
        remove_edges
        """
        # TODO(xiangsx): block do not support add_nodes
        if ntype is None:
            if self._graph.number_of_ntypes() != 1:
                raise DGLError('Node type name must be specified if there are more than one '
                               'node types.')

        # nothing happen
        if num == 0:
            return

        assert num > 0, 'Number of new nodes should be larger than one.'
        ntid = self.get_ntype_id(ntype)
        # update graph idx
        metagraph = self._graph.metagraph
        num_nodes_per_type = []
        for c_ntype in self.ntypes:
            if self.get_ntype_id(c_ntype) == ntid:
                num_nodes_per_type.append(self.number_of_nodes(c_ntype) + num)
            else:
                num_nodes_per_type.append(self.number_of_nodes(c_ntype))

        relation_graphs = []
        for c_etype in self.canonical_etypes:
            # src or dst == ntype, update the relation graph
            if self.get_ntype_id(c_etype[0]) == ntid or self.get_ntype_id(c_etype[2]) == ntid:
                u, v = self.edges(form='uv', order='eid', etype=c_etype)
                hgidx = heterograph_index.create_unitgraph_from_coo(
                    1 if c_etype[0] == c_etype[2] else 2,
                    self.number_of_nodes(c_etype[0]) + \
                        (num if self.get_ntype_id(c_etype[0]) == ntid else 0),
                    self.number_of_nodes(c_etype[2]) + \
                        (num if self.get_ntype_id(c_etype[2]) == ntid else 0),
                    u,
                    v,
                    ['coo', 'csr', 'csc'])
                relation_graphs.append(hgidx)
            else:
                # do nothing
                relation_graphs.append(self._graph.get_relation_graph(self.get_etype_id(c_etype)))
        hgidx = heterograph_index.create_heterograph_from_relations(
            metagraph, relation_graphs, utils.toindex(num_nodes_per_type, "int64"))
        self._graph = hgidx

        # update data frames
        if data is None:
            # Initialize feature with :func:`set_n_initializer`
            self._node_frames[ntid].add_rows(num)
        else:
            self._node_frames[ntid].append(data)
        self._reset_cached_info()

    def add_edge(self, u, v, data=None, etype=None):
        """Add one edge to the graph.

        DEPRECATED: please use ``add_edges``.
        """
        dgl_warning("DGLGraph.add_edge is deprecated. Please use DGLGraph.add_edges")
        self.add_edges(u, v, data, etype)

    def add_edges(self, u, v, data=None, etype=None):
        r"""Add multiple new edges for the specified edge type

        The i-th new edge will be from ``u[i]`` to ``v[i]``.

        Parameters
        ----------
        u : int, tensor, numpy.ndarray, list
            Source node IDs, ``u[i]`` gives the source node for the i-th new edge.
        v : int, tensor, numpy.ndarray, list
            Destination node IDs, ``v[i]`` gives the destination node for the i-th new edge.
        data : dict, optional
            Feature data of the added edges. The i-th row of the feature data
            corresponds to the i-th new edge.
        etype : str or tuple of str, optional
            The type of the new edges. Can be omitted if there is
            only one edge type in the graph.

        Notes
        -----

        * Inplace update is applied to the current graph.
        * If end nodes of adding edges does not exists, add_nodes is invoked
        to add new nodes. The node features of the new nodes will be created
        by initializers defined with :func:`set_n_initializer` (default
        initializer fills zeros). In certain cases, it is recommanded to
        add_nodes first and then add_edges.
        * If the key of ``data`` does not contain some existing feature fields,
        those features for the new edges will be created by initializers
        defined with :func:`set_n_initializer` (default initializer fills zeros).
        * If the key of ``data`` contains new feature fields, those features for
        the old edges will be created by initializers defined with
        :func:`set_n_initializer` (default initializer fills zeros).

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        **Homogeneous Graphs or Heterogeneous Graphs with A Single Edge Type**

        >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
        >>> g.num_edges()
        2
        >>> g.add_edges(torch.tensor([1, 3]), torch.tensor([0, 1]))
        >>> g.num_edges()
        4

        Since ``u`` or ``v`` contains a non-existing node ID, the nodes are
        added implicitly.
        >>> g.num_nodes()
        4

        If the graph has some edge features and new edges are added without
        features, their features will be created by initializers defined
        with :func:`set_n_initializer`.

        >>> g.edata['h'] = torch.ones(4, 1)
        >>> g.add_edges(torch.tensor([1]), torch.tensor([1]))
        >>> g.edata['h']
        tensor([[1.], [1.], [1.], [1.], [0.]])

        We can also assign features for the new edges in adding new edges.

        >>> g.add_edges(torch.tensor([0, 0]), torch.tensor([2, 2]),
        >>>             {'h': torch.tensor([[1.], [2.]]), 'w': torch.ones(2, 1)})
        >>> g.edata['h']
        tensor([[1.], [1.], [1.], [1.], [0.], [1.], [2.]])

        Since ``data`` contains new feature fields, the features for old edges
        will be created by initializers defined with :func:`set_n_initializer`.

        >>> g.edata['w']
        tensor([[0.], [0.], [0.], [0.], [0.], [1.], [1.]])

        **Heterogeneous Graphs with Multiple Edge Types**

        >>> g = dgl.heterograph({
        >>>     ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
        >>>                                 torch.tensor([0, 0, 1, 1])),
        >>>     ('developer', 'develops', 'game'): (torch.tensor([0, 1]),
        >>>                                         torch.tensor([0, 1]))
        >>>     })
        >>> g.add_edges(torch.tensor([3]), torch.tensor([3]))
        DGLError: Edge type name must be specified
        if there are more than one edge types.
        >>> g.number_of_edges('plays')
        4
        >>>  g.add_edges(torch.tensor([3]), torch.tensor([3]), etype='plays')
        >>> g.number_of_edges('plays')
        5

        See Also
        --------
        add_nodes
        remove_nodes
        remove_edges
        """
        # TODO(xiangsx): block do not support add_edges
        u = utils.prepare_tensor(self, u, 'u')
        v = utils.prepare_tensor(self, v, 'v')

        if etype is None:
            if self._graph.number_of_etypes() != 1:
                raise DGLError('Edge type name must be specified if there are more than one '
                               'edge types.')

        # nothing changed
        if len(u) == 0 or len(v) == 0:
            return

        assert len(u) == len(v) or len(u) == 1 or len(v) == 1, \
            'The number of source nodes and the number of destination nodes should be same, ' \
            'or either the number of source nodes or the number of destination nodes is 1.'

        if len(u) == 1 and len(v) > 1:
            u = F.full_1d(len(v), F.as_scalar(u), dtype=F.dtype(u), ctx=F.context(u))
        if len(v) == 1 and len(u) > 1:
            v = F.full_1d(len(u), F.as_scalar(v), dtype=F.dtype(v), ctx=F.context(v))

        u_type, e_type, v_type = self.to_canonical_etype(etype)
        # if end nodes of adding edges does not exists
        # use add_nodes to add new nodes first.
        num_of_u = self.number_of_nodes(u_type)
        num_of_v = self.number_of_nodes(v_type)
        u_max = F.as_scalar(F.max(u, dim=0)) + 1
        v_max = F.as_scalar(F.max(v, dim=0)) + 1

        if u_type == v_type:
            num_nodes = max(u_max, v_max)
            if num_nodes > num_of_u:
                self.add_nodes(num_nodes - num_of_u, ntype=u_type)
        else:
            if u_max > num_of_u:
                self.add_nodes(u_max - num_of_u, ntype=u_type)
            if v_max > num_of_v:
                self.add_nodes(v_max - num_of_v, ntype=v_type)

        # metagraph is not changed
        metagraph = self._graph.metagraph
        num_nodes_per_type = []
        for ntype in self.ntypes:
            num_nodes_per_type.append(self.number_of_nodes(ntype))
        # update graph idx
        relation_graphs = []
        for c_etype in self.canonical_etypes:
            # the target edge type
            if c_etype == (u_type, e_type, v_type):
                old_u, old_v = self.edges(form='uv', order='eid', etype=c_etype)
                hgidx = heterograph_index.create_unitgraph_from_coo(
                    1 if u_type == v_type else 2,
                    self.number_of_nodes(u_type),
                    self.number_of_nodes(v_type),
                    F.cat([old_u, u], dim=0),
                    F.cat([old_v, v], dim=0),
                    ['coo', 'csr', 'csc'])
                relation_graphs.append(hgidx)
            else:
                # do nothing
                # Note: node range change has been handled in add_nodes()
                relation_graphs.append(self._graph.get_relation_graph(self.get_etype_id(c_etype)))

        hgidx = heterograph_index.create_heterograph_from_relations(
            metagraph, relation_graphs, utils.toindex(num_nodes_per_type, "int64"))
        self._graph = hgidx

        # handle data
        etid = self.get_etype_id(etype)
        if data is None:
            self._edge_frames[etid].add_rows(len(u))
        else:
            self._edge_frames[etid].append(data)
        self._reset_cached_info()

    def remove_edges(self, eids, etype=None):
        r"""Remove multiple edges with the specified edge type

        Nodes will not be removed. After removing edges, the rest
        edges will be re-indexed using consecutive integers from 0,
        with their relative order preserved.

        The features for the removed edges will be removed accordingly.

        Parameters
        ----------
        eids : int, tensor, numpy.ndarray, list
            IDs for the edges to remove.
        etype : str or tuple of str, optional
            The type of the edges to remove. Can be omitted if there is
            only one edge type in the graph.

        Examples
        --------

        >>> import dgl
        >>> import torch

        **Homogeneous Graphs or Heterogeneous Graphs with A Single Edge Type**

        >>> g = dgl.graph((torch.tensor([0, 0, 2]), torch.tensor([0, 1, 2])))
        >>> g.edata['he'] = torch.arange(3).float().reshape(-1, 1)
        >>> g.remove_edges(torch.tensor([0, 1]))
        >>> g
        Graph(num_nodes=3, num_edges=1,
            ndata_schemes={}
            edata_schemes={'he': Scheme(shape=(1,), dtype=torch.float32)})
        >>> g.edges('all')
        (tensor([2]), tensor([2]), tensor([0]))
        >>> g.edata['he']
        tensor([[2.]])

        **Heterogeneous Graphs with Multiple Edge Types**

        >>> g = dgl.heterograph({
        >>>     ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
        >>>                                 torch.tensor([0, 0, 1, 1])),
        >>>     ('developer', 'develops', 'game'): (torch.tensor([0, 1]),
        >>>                                         torch.tensor([0, 1]))
        >>>     })
        >>> g.remove_edges(torch.tensor([0, 1]))
        DGLError: Edge type name must be specified
        if there are more than one edge types.
        >>> g.remove_edges(torch.tensor([0, 1]), 'plays')
        >>> g.edges('all', etype='plays')
        (tensor([0, 1]), tensor([0, 0]), tensor([0, 1]))

        See Also
        --------
        add_nodes
        add_edges
        remove_nodes
        """
        # TODO(xiangsx): block do not support remove_edges
        if etype is None:
            if self._graph.number_of_etypes() != 1:
                raise DGLError('Edge type name must be specified if there are more than one ' \
                               'edge types.')
        eids = utils.prepare_tensor(self, eids, 'u')
        if len(eids) == 0:
            # no edge to delete
            return
        assert self.number_of_edges(etype) > F.as_scalar(F.max(eids, dim=0)), \
            'The input eid {} is out of the range [0:{})'.format(
                F.as_scalar(F.max(eids, dim=0)), self.number_of_edges(etype))

        # edge_subgraph
        edges = {}
        u_type, e_type, v_type = self.to_canonical_etype(etype)
        for c_etype in self.canonical_etypes:
            # the target edge type
            if c_etype == (u_type, e_type, v_type):
                origin_eids = self.edges(form='eid', order='eid', etype=c_etype)
                edges[c_etype] = utils.compensate(eids, origin_eids)
            else:
                edges[c_etype] = self.edges(form='eid', order='eid', etype=c_etype)

        sub_g = self.edge_subgraph(edges, preserve_nodes=True)
        self._graph = sub_g._graph
        self._node_frames = sub_g._node_frames
        self._edge_frames = sub_g._edge_frames

    def remove_nodes(self, nids, ntype=None):
        r"""Remove multiple nodes with the specified node type

        Edges that connect to the nodes will be removed as well. After removing
        nodes and edges, the rest nodes and edges will be re-indexed using
        consecutive integers from 0, with their relative order preserved.

        The features for the removed nodes/edges will be removed accordingly.

        Parameters
        ----------
        nids : int, tensor, numpy.ndarray, list
            Nodes to remove.
        ntype : str, optional
            The type of the nodes to remove. Can be omitted if there is
            only one node type in the graph.

        Examples
        --------

        >>> import dgl
        >>> import torch

        **Homogeneous Graphs or Heterogeneous Graphs with A Single Node Type**

        >>> g = dgl.graph((torch.tensor([0, 0, 2]), torch.tensor([0, 1, 2])))
        >>> g.ndata['hv'] = torch.arange(3).float().reshape(-1, 1)
        >>> g.edata['he'] = torch.arange(3).float().reshape(-1, 1)
        >>> g.remove_nodes(torch.tensor([0, 1]))
        >>> g
        Graph(num_nodes=1, num_edges=1,
            ndata_schemes={'hv': Scheme(shape=(1,), dtype=torch.float32)}
            edata_schemes={'he': Scheme(shape=(1,), dtype=torch.float32)})
        >>> g.ndata['hv']
        tensor([[2.]])
        >>> g.edata['he']
        tensor([[2.]])

        **Heterogeneous Graphs with Multiple Node Types**

        >>> g = dgl.heterograph({
        >>>     ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
        >>>                                 torch.tensor([0, 0, 1, 1])),
        >>>     ('developer', 'develops', 'game'): (torch.tensor([0, 1]),
        >>>                                         torch.tensor([0, 1]))
        >>>     })
        >>> g.remove_nodes(torch.tensor([0, 1]))
        DGLError: Node type name must be specified
        if there are more than one node types.
        >>> g.remove_nodes(torch.tensor([0, 1]), ntype='game')
        >>> g.num_nodes('user')
        3
        >>> g.num_nodes('game')
        0
        >>> g.num_edges('plays')
        0

        See Also
        --------
        add_nodes
        add_edges
        remove_edges
        """
        # TODO(xiangsx): block do not support remove_nodes
        if ntype is None:
            if self._graph.number_of_ntypes() != 1:
                raise DGLError('Node type name must be specified if there are more than one ' \
                               'node types.')

        nids = utils.prepare_tensor(self, nids, 'u')
        if len(nids) == 0:
            # no node to delete
            return
        assert self.number_of_nodes(ntype) > F.as_scalar(F.max(nids, dim=0)), \
            'The input nids {} is out of the range [0:{})'.format(
                F.as_scalar(F.max(nids, dim=0)), self.number_of_nodes(ntype))

        ntid = self.get_ntype_id(ntype)
        nodes = {}
        for c_ntype in self.ntypes:
            if self.get_ntype_id(c_ntype) == ntid:
                original_nids = self.nodes(c_ntype)
                nodes[c_ntype] = utils.compensate(nids, original_nids)
            else:
                nodes[c_ntype] = self.nodes(c_ntype)

        # node_subgraph
        sub_g = self.subgraph(nodes)
        self._graph = sub_g._graph
        self._node_frames = sub_g._node_frames
        self._edge_frames = sub_g._edge_frames

    def _reset_cached_info(self):
        """Some info like batch_num_nodes may be stale after mutation
        Clean these cached info
        """
        self._batch_num_nodes = None
        self._batch_num_edges = None


    #################################################################
    # Metagraph query
    #################################################################

    @property
    def is_unibipartite(self):
        """Return whether the graph is a uni-bipartite graph.

        A uni-bipartite heterograph can further divide its node types into two sets:
        SRC and DST. All edges are from nodes in SRC to nodes in DST. The following APIs
        can be used to get the nodes and types that belong to SRC and DST sets:

        * :func:`srctype` and :func:`dsttype`
        * :func:`srcdata` and :func:`dstdata`
        * :func:`srcnodes` and :func:`dstnodes`

        Note that we allow two node types to have the same name as long as one
        belongs to SRC while the other belongs to DST. To distinguish them, prepend
        the name with ``"SRC/"`` or ``"DST/"`` when specifying a node type.
        """
        return self._is_unibipartite

    @property
    def ntypes(self):
        """Return the list of node types of this graph.

        Returns
        -------
        list of str

        Examples
        --------

        >>> follows_g = dgl.graph(([0, 1], [1, 2]), 'user', 'follows')
        >>> plays_g = dgl.bipartite(([0, 1, 1, 2], [0, 0, 1, 1]), 'user', 'plays', 'game')
        >>> g = dgl.hetero_from_relations([follows_g, plays_g])
        >>> g.ntypes
        ['user', 'game']
        """
        return self._ntypes

    @property
    def etypes(self):
        """Return the list of edge types of this graph.

        Returns
        -------
        list of str

        Examples
        --------

        >>> follows_g = dgl.graph(([0, 1], [1, 2]), 'user', 'follows')
        >>> plays_g = dgl.bipartite(([0, 1, 1, 2], [0, 0, 1, 1]), 'user', 'plays', 'game')
        >>> g = dgl.hetero_from_relations([follows_g, plays_g])
        >>> g.etypes
        ['follows', 'plays']
        """
        return self._etypes

    @property
    def canonical_etypes(self):
        """Return the list of canonical edge types of this graph.

        A canonical edge type is a tuple of string (src_type, edge_type, dst_type).

        Returns
        -------
        list of 3-tuples

        Examples
        --------

        >>> follows_g = dgl.graph(([0, 1], [1, 2]), 'user', 'follows')
        >>> plays_g = dgl.bipartite(([0, 1, 1, 2], [0, 0, 1, 1]), 'user', 'plays', 'game')
        >>> g = dgl.hetero_from_relations([follows_g, plays_g])
        >>> g.canonical_etypes
        [('user', 'follows', 'user'), ('user', 'plays', 'game')]
        """
        return self._canonical_etypes

    @property
    def srctypes(self):
        """Return the node types in the SRC category. Return :attr:``ntypes`` if
        the graph is not a uni-bipartite graph.
        """
        if self.is_unibipartite:
            return sorted(list(self._srctypes_invmap.keys()))
        else:
            return self.ntypes

    @property
    def dsttypes(self):
        """Return the node types in the DST category. Return :attr:``ntypes`` if
        the graph is not a uni-bipartite graph.
        """
        if self.is_unibipartite:
            return sorted(list(self._dsttypes_invmap.keys()))
        else:
            return self.ntypes

    def metagraph(self):
        """Return the metagraph as networkx.MultiDiGraph.

        The nodes are labeled with node type names.
        The edges have their keys holding the edge type names.

        Returns
        -------
        networkx.MultiDiGraph

        Examples
        --------

        >>> follows_g = dgl.graph(([0, 1], [1, 2]), 'user', 'follows')
        >>> plays_g = dgl.bipartite(([0, 1, 1, 2], [0, 0, 1, 1]), 'user', 'plays', 'game')
        >>> g = dgl.hetero_from_relations([follows_g, plays_g])
        >>> meta_g = g.metagraph()

        The metagraph then has two nodes and two edges.

        >>> meta_g.nodes()
        NodeView(('user', 'game'))
        >>> meta_g.number_of_nodes()
        2
        >>> meta_g.edges()
        OutMultiEdgeDataView([('user', 'user'), ('user', 'game')])
        >>> meta_g.number_of_edges()
        2
        """
        nx_graph = self._graph.metagraph.to_networkx()
        nx_metagraph = nx.MultiDiGraph()
        for u_v in nx_graph.edges:
            srctype, etype, dsttype = self.canonical_etypes[nx_graph.edges[u_v]['id']]
            nx_metagraph.add_edge(srctype, dsttype, etype)
        return nx_metagraph

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

        Examples
        --------

        Instantiate a heterograph.

        >>> g1 = dgl.graph(([0, 1], [1, 2]), 'user', 'follows')
        >>> g2 = dgl.bipartite(([0, 1, 1, 2], [0, 0, 1, 1]), 'user', 'plays', 'game')
        >>> g3 = dgl.bipartite(([0, 1], [0, 1]), 'developer', 'follows', 'game')
        >>> g = dgl.hetero_from_relations([g1, g2, g3])

        Get canonical edge types.

        >>> g.to_canonical_etype('plays')
        ('user', 'plays', 'game')
        >>> g.to_canonical_etype(('user', 'plays', 'game'))
        ('user', 'plays', 'game')
        >>> g.to_canonical_etype('follows')
        DGLError: Edge type "follows" is ambiguous.
        Please use canonical etype type in the form of (srctype, etype, dsttype)
        """
        if etype is None:
            if len(self.etypes) != 1:
                raise DGLError('Edge type name must be specified if there are more than one '
                               'edge types.')
            etype = self.etypes[0]
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
        if self.is_unibipartite and ntype is not None:
            # Only check 'SRC/' and 'DST/' prefix when is_unibipartite graph is True.
            if ntype.startswith('SRC/'):
                return self.get_ntype_id_from_src(ntype[4:])
            elif ntype.startswith('DST/'):
                return self.get_ntype_id_from_dst(ntype[4:])
            # If there is no prefix, fallback to normal lookup.

        # Lookup both SRC and DST
        if ntype is None:
            if self.is_unibipartite or len(self._srctypes_invmap) != 1:
                raise DGLError('Node type name must be specified if there are more than one '
                               'node types.')
            return 0
        ntid = self._srctypes_invmap.get(ntype, self._dsttypes_invmap.get(ntype, None))
        if ntid is None:
            raise DGLError('Node type "{}" does not exist.'.format(ntype))
        return ntid

    def get_ntype_id_from_src(self, ntype):
        """Return the id of the given SRC node type.

        ntype can also be None. If so, there should be only one node type in the
        SRC category. Callable even when the self graph is not uni-bipartite.

        Parameters
        ----------
        ntype : str
            Node type

        Returns
        -------
        int
        """
        if ntype is None:
            if len(self._srctypes_invmap) != 1:
                raise DGLError('SRC node type name must be specified if there are more than one '
                               'SRC node types.')
            return next(iter(self._srctypes_invmap.values()))
        ntid = self._srctypes_invmap.get(ntype, None)
        if ntid is None:
            raise DGLError('SRC node type "{}" does not exist.'.format(ntype))
        return ntid

    def get_ntype_id_from_dst(self, ntype):
        """Return the id of the given DST node type.

        ntype can also be None. If so, there should be only one node type in the
        DST category. Callable even when the self graph is not uni-bipartite.

        Parameters
        ----------
        ntype : str
            Node type

        Returns
        -------
        int
        """
        if ntype is None:
            if len(self._dsttypes_invmap) != 1:
                raise DGLError('DST node type name must be specified if there are more than one '
                               'DST node types.')
            return next(iter(self._dsttypes_invmap.values()))
        ntid = self._dsttypes_invmap.get(ntype, None)
        if ntid is None:
            raise DGLError('DST node type "{}" does not exist.'.format(ntype))
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
    # Batching
    #################################################################
    @property
    def batch_size(self):
        """TBD"""
        return len(self.batch_num_nodes(self.ntypes[0]))

    def batch_num_nodes(self, ntype=None):
        """TBD"""
        if self._batch_num_nodes is None:
            self._batch_num_nodes = {}
            for ty in self.ntypes:
                bnn = F.copy_to(F.tensor([self.number_of_nodes(ty)], F.int64), self.device)
                self._batch_num_nodes[ty] = bnn
        if ntype is None:
            if len(self.ntypes) != 1:
                raise DGLError('Node type name must be specified if there are more than one '
                               'node types.')
            ntype = self.ntypes[0]
        return self._batch_num_nodes[ntype]

    def set_batch_num_nodes(self, val):
        """TBD"""
        if not isinstance(val, Mapping):
            if len(self.ntypes) != 1:
                raise DGLError('Must provide a dictionary when there are multiple node types.')
            val = {self.ntypes[0] : val}
        self._batch_num_nodes = val

    def batch_num_edges(self, etype=None):
        """TBD"""
        if self._batch_num_edges is None:
            self._batch_num_edges = {}
            for ty in self.canonical_etypes:
                bne = F.copy_to(F.tensor([self.number_of_edges(ty)], F.int64), self.device)
                self._batch_num_edges[ty] = bne
        if etype is None:
            if len(self.etypes) != 1:
                raise DGLError('Edge type name must be specified if there are more than one '
                               'edge types.')
            etype = self.canonical_etypes[0]
        return self._batch_num_edges[etype]

    def set_batch_num_edges(self, val):
        """TBD"""
        if not isinstance(val, Mapping):
            if len(self.etypes) != 1:
                raise DGLError('Must provide a dictionary when there are multiple edge types.')
            val = {self.canonical_etypes[0] : val}
        self._batch_num_edges = val

    #################################################################
    # View
    #################################################################

    @property
    def nodes(self):
        """Return a node view that can be used to set/get feature
        data of a single node type.

        Examples
        --------
        The following example uses PyTorch backend.

        To set features of all users

        >>> g = dgl.graph(([0, 1], [1, 2]), 'user', 'follows')
        >>> g.nodes['user'].data['h'] = torch.zeros(3, 5)

        See Also
        --------
        ndata
        """
        return HeteroNodeView(self, self.get_ntype_id)

    @property
    def srcnodes(self):
        """Return a SRC node view that can be used to set/get feature
        data of a single node type.

        Examples
        --------
        The following example uses PyTorch backend.

        To set features of all users

        >>> g = dgl.bipartite(([0, 1], [1, 2]), 'user', 'plays', 'game')
        >>> g.srcnodes['user'].data['h'] = torch.zeros(2, 5)

        See Also
        --------
        srcdata
        """
        return HeteroNodeView(self, self.get_ntype_id_from_src)

    @property
    def dstnodes(self):
        """Return a DST node view that can be used to set/get feature
        data of a single node type.

        Examples
        --------
        The following example uses PyTorch backend.

        To set features of all games

        >>> g = dgl.bipartite(([0, 1], [1, 2]), 'user', 'plays', 'game')
        >>> g.dstnodes['game'].data['h'] = torch.zeros(3, 5)

        See Also
        --------
        dstdata
        """
        return HeteroNodeView(self, self.get_ntype_id_from_dst)

    @property
    def ndata(self):
        """Return the data view of all the nodes.

        If the graph has only one node type, ``g.ndata['feat']`` gives
        the node feature data under name ``'feat'``.
        If the graph has multiple node types, then ``g.ndata['feat']``
        returns a dictionary where the key is the node type and the
        value is the node feature tensor. If the node type does not
        have feature `'feat'`, it is not included in the dictionary.

        Examples
        --------
        The following example uses PyTorch backend.

        To set features of all nodes in a heterogeneous graph
        with only one node type:

        >>> g = dgl.graph(([0, 1], [1, 2]), 'user', 'follows')
        >>> g.ndata['h'] = torch.zeros(3, 5)

        To set features of all nodes in a heterogeneous graph
        with multiple node types:

        >>> g = dgl.heterograph({('user', 'like', 'movie') : ([0, 1, 1], [1, 2, 0])})
        >>> g.ndata['h'] = {'user': torch.zeros(2, 5),
        ...                 'movie': torch.zeros(3, 5)}
        >>> g.ndata['h']
        ... {'user': tensor([[0., 0., 0., 0., 0.],
        ...                 [0., 0., 0., 0., 0.]]),
        ...  'movie': tensor([[0., 0., 0., 0., 0.],
        ...                   [0., 0., 0., 0., 0.],
        ...                   [0., 0., 0., 0., 0.]])}

        To set features of part of nodes in a heterogeneous graph
        with multiple node types:

        >>> g = dgl.heterograph({('user', 'like', 'movie') : ([0, 1, 1], [1, 2, 0])})
        >>> g.ndata['h'] = {'user': torch.zeros(2, 5)}
        >>> g.ndata['h']
        ... {'user': tensor([[0., 0., 0., 0., 0.],
        ...                  [0., 0., 0., 0., 0.]])}
        >>> # clean the feature 'h' and no node type contains 'h'
        >>> g.ndata.pop('h')
        >>> g.ndata['h']
        ... {}

        See Also
        --------
        nodes
        """
        if len(self.ntypes) == 1:
            ntid = self.get_ntype_id(None)
            ntype = self.ntypes[0]
            return HeteroNodeDataView(self, ntype, ntid, ALL)
        else:
            ntids = [self.get_ntype_id(ntype) for ntype in self.ntypes]
            ntypes = self.ntypes
            return HeteroNodeDataView(self, ntypes, ntids, ALL)


    @property
    def srcdata(self):
        """Return the data view of all nodes in the SRC category.

        If the source nodes have only one node type, ``g.srcdata['feat']``
        gives the node feature data under name ``'feat'``.
        If the source nodes have multiple node types, then
        ``g.srcdata['feat']`` returns a dictionary where the key is
        the source node type and the value is the node feature
        tensor. If the source node type does not have feature
        `'feat'`, it is not included in the dictionary.

        Examples
        --------
        The following example uses PyTorch backend.

        To set features of all source nodes in a graph with only one edge type:

        >>> g = dgl.bipartite(([0, 1], [1, 2]), 'user', 'plays', 'game')
        >>> g.srcdata['h'] = torch.zeros(2, 5)

        This is equivalent to

        >>> g.nodes['user'].data['h'] = torch.zeros(2, 5)

        Also work on more complex uni-bipartite graph

        >>> g = dgl.heterograph({
        ...     ('user', 'plays', 'game') : ([0, 1], [1, 2]),
        ...     ('user', 'reads', 'book') : ([0, 1], [1, 0]),
        ...     })
        >>> print(g.is_unibipartite)
        True
        >>> g.srcdata['h'] = torch.zeros(2, 5)

        To set features of all source nodes in a uni-bipartite graph
        with multiple source node types:

        >>> g = dgl.heterograph({
        ...     ('game', 'liked-by', 'user') : ([1, 2], [0, 1]),
        ...     ('book', 'liked-by', 'user') : ([0, 1], [1, 0]),
        ...     })
        >>> print(g.is_unibipartite)
        True
        >>> g.srcdata['h'] = {'game' : torch.zeros(3, 5),
        ...                   'book' : torch.zeros(2, 5)}
        >>> g.srcdata['h']
        ... {'game': tensor([[0., 0., 0., 0., 0.],
        ...                  [0., 0., 0., 0., 0.],
        ...                  [0., 0., 0., 0., 0.]]),
        ...  'book': tensor([[0., 0., 0., 0., 0.],
        ...                  [0., 0., 0., 0., 0.]])}

        To set features of part of source nodes in a uni-bipartite graph
        with multiple source node types:
        >>> g = dgl.heterograph({
        ...     ('game', 'liked-by', 'user') : ([1, 2], [0, 1]),
        ...     ('book', 'liked-by', 'user') : ([0, 1], [1, 0]),
        ...     })
        >>> g.srcdata['h'] = {'game' : torch.zeros(3, 5)}
        >>> g.srcdata['h']
        >>> {'game': tensor([[0., 0., 0., 0., 0.],
        ...                  [0., 0., 0., 0., 0.],
        ...                  [0., 0., 0., 0., 0.]])}
        >>> # clean the feature 'h' and no source node type contains 'h'
        >>> g.srcdata.pop('h')
        >>> g.srcdata['h']
        ... {}


        Notes
        -----
        This is identical to :any:`DGLHeteroGraph.ndata` if the graph is homogeneous.

        See Also
        --------
        nodes
        """
        if len(self.srctypes) == 1:
            ntype = self.srctypes[0]
            ntid = self.get_ntype_id_from_src(ntype)
            return HeteroNodeDataView(self, ntype, ntid, ALL)
        else:
            ntypes = self.srctypes
            ntids = [self.get_ntype_id_from_src(ntype) for ntype in ntypes]
            return HeteroNodeDataView(self, ntypes, ntids, ALL)

    @property
    def dstdata(self):
        """Return the data view of all destination nodes.

        If the destination nodes have only one node type,
        ``g.dstdata['feat']`` gives the node feature data under name
        ``'feat'``.
        If the destination nodes have multiple node types, then
        ``g.dstdata['feat']`` returns a dictionary where the key is
        the destination node type and the value is the node feature
        tensor. If the destination node type does not have feature
        `'feat'`, it is not included in the dictionary.

        Examples
        --------
        The following example uses PyTorch backend.

        To set features of all source nodes in a graph with only one edge type:

        >>> g = dgl.bipartite(([0, 1], [1, 2]), 'user', 'plays', 'game')
        >>> g.dstdata['h'] = torch.zeros(3, 5)

        This is equivalent to

        >>> g.nodes['game'].data['h'] = torch.zeros(3, 5)

        Also work on more complex uni-bipartite graph

        >>> g = dgl.heterograph({
        ...     ('user', 'plays', 'game') : ([0, 1], [1, 2]),
        ...     ('store', 'sells', 'game') : ([0, 1], [1, 0]),
        ...     })
        >>> print(g.is_unibipartite)
        True
        >>> g.dstdata['h'] = torch.zeros(3, 5)

        To set features of all destination nodes in a uni-bipartite graph
        with multiple destination node types::

        >>> g = dgl.heterograph({
        ...     ('user', 'plays', 'game') : ([0, 1], [1, 2]),
        ...     ('user', 'reads', 'book') : ([0, 1], [1, 0]),
        ...     })
        >>> print(g.is_unibipartite)
        True
        >>> g.dstdata['h'] = {'game' : torch.zeros(3, 5),
        ...                   'book' : torch.zeros(2, 5)}
        >>> g.dstdata['h']
        ... {'game': tensor([[0., 0., 0., 0., 0.],
        ...                  [0., 0., 0., 0., 0.],
        ...                  [0., 0., 0., 0., 0.]]),
        ...  'book': tensor([[0., 0., 0., 0., 0.],
        ...                  [0., 0., 0., 0., 0.]])}

        To set features of part of destination nodes in a uni-bipartite graph
        with multiple destination node types:
        >>> g = dgl.heterograph({
        ...     ('user', 'plays', 'game') : ([0, 1], [1, 2]),
        ...     ('user', 'reads', 'book') : ([0, 1], [1, 0]),
        ...     })
        >>> g.dstdata['h'] = {'game' : torch.zeros(3, 5)}
        >>> g.dstdata['h']
        ... {'game': tensor([[0., 0., 0., 0., 0.],
        ...                  [0., 0., 0., 0., 0.],
        ...                  [0., 0., 0., 0., 0.]])}
        >>> # clean the feature 'h' and no destination node type contains 'h'
        >>> g.dstdata.pop('h')
        >>> g.dstdata['h']
        ... {}

        Notes
        -----
        This is identical to :any:`DGLHeteroGraph.ndata` if the graph is homogeneous.

        See Also
        --------
        nodes
        """
        if len(self.dsttypes) == 1:
            ntype = self.dsttypes[0]
            ntid = self.get_ntype_id_from_dst(ntype)
            return HeteroNodeDataView(self, ntype, ntid, ALL)
        else:
            ntypes = self.dsttypes
            ntids = [self.get_ntype_id_from_dst(ntype) for ntype in ntypes]
            return HeteroNodeDataView(self, ntypes, ntids, ALL)

    @property
    def edges(self):
        """Return an edge view that can be used to set/get feature
        data of a single edge type.

        Examples
        --------
        The following example uses PyTorch backend.

        To set features of all "play" relationships:

        >>> g = dgl.bipartite(([0, 1, 1], [0, 0, 2]), 'user', 'plays', 'game')
        >>> g.edges['plays'].data['h'] = torch.zeros(3, 4)

        See Also
        --------
        edata
        """
        return HeteroEdgeView(self)

    @property
    def edata(self):
        """Return the data view of all the edges.

        If the graph has only one edge type, ``g.edata['feat']`` gives the
        edge feature data under name ``'feat'``.
        If the graph has multiple edge types, then ``g.edata['feat']``
        returns a dictionary where the key is the edge type and the value
        is the edge feature tensor. If the edge type does not have feature
        ``'feat'``, it is not included in the dictionary.

        Note: When the graph has multiple edge type, The key used in
        ``g.edata['feat']`` should be the canonical_etypes, i.e.
        (h_ntype, r_type, t_ntype).

        Examples
        --------
        The following example uses PyTorch backend.

        To set features of all edges in a heterogeneous graph
        with only one edge type:

        >>> g = dgl.graph(([0, 1], [1, 2]), 'user', 'follows')
        >>> g.edata['h'] = torch.zeros(2, 5)

        To set features of all edges in a heterogeneous graph
        with multiple edge types:

        >>> g0 = dgl.bipartite(([0, 1, 1], [1, 0, 1]), 'user', 'watches', 'movie')
        >>> g1 = dgl.bipartite(([0, 1], [0, 1]), 'user', 'watches', 'TV')
        >>> g = dgl.hetero_from_relations([g0, g1])
        >>> g.edata['h'] = {('user', 'watches', 'movie') : torch.zeros(3, 5),
                            ('user', 'watches', 'TV') : torch.zeros(2, 5)}
        >>> g.edata['h']
        ... {('user', 'watches', 'movie'): tensor([[0., 0., 0., 0., 0.],
        ...                                        [0., 0., 0., 0., 0.],
        ...                                        [0., 0., 0., 0., 0.]]),
        ...  ('user', 'watches', 'TV'): tensor([[0., 0., 0., 0., 0.],
        ...                                     [0., 0., 0., 0., 0.]])}

        To set features of part of edges in a heterogeneous graph
        with multiple edge types:
        >>> g0 = dgl.bipartite(([0, 1, 1], [1, 0, 1]), 'user', 'watches', 'movie')
        >>> g1 = dgl.bipartite(([0, 1], [0, 1]), 'user', 'watches', 'TV')
        >>> g = dgl.hetero_from_relations([g0, g1])
        >>> g.edata['h'] = {('user', 'watches', 'movie') : torch.zeros(3, 5)}
        >>> g.edata['h']
        ... {('user', 'watches', 'movie'): tensor([[0., 0., 0., 0., 0.],
        ...                                        [0., 0., 0., 0., 0.],
        ...                                        [0., 0., 0., 0., 0.]])}
        >>> # clean the feature 'h' and no edge type contains 'h'
        >>> g.edata.pop('h')
        >>> g.edata['h']
        ... {}

        See Also
        --------
        edges
        """
        if len(self.canonical_etypes) == 1:
            return HeteroEdgeDataView(self, None, ALL)
        else:
            return HeteroEdgeDataView(self, self.canonical_etypes, ALL)

    def _find_etypes(self, key):
        etypes = [
            i for i, (srctype, etype, dsttype) in enumerate(self._canonical_etypes) if
            (key[0] == SLICE_FULL or key[0] == srctype) and
            (key[1] == SLICE_FULL or key[1] == etype) and
            (key[2] == SLICE_FULL or key[2] == dsttype)]
        return etypes

    def __getitem__(self, key):
        """Return the relation slice of this graph.

        A relation slice is accessed with ``self[srctype, etype, dsttype]``, where
        ``srctype``, ``etype``, and ``dsttype`` can be either a string or a full
        slice (``:``) representing wildcard (i.e. any source/edge/destination type).

        A relation slice is a homogeneous (with one node type and one edge type) or
        bipartite (with two node types and one edge type) graph, transformed from
        the original heterogeneous graph.

        If there is only one canonical edge type found, then the returned relation
        slice would be a subgraph induced from the original graph.  That is, it is
        equivalent to ``self.edge_type_subgraph(etype)``.  The node and edge features
        of the returned graph would be shared with thew original graph.

        If there are multiple canonical edge type found, then the source/edge/destination
        node types would be a *concatenation* of original node/edge types.  The
        new source/destination node type would have the concatenation determined by
        :func:`dgl.combine_names() <dgl.combine_names>` called on original source/destination
        types as its name.  The source/destination node would be formed by concatenating the
        common features of the original source/destination types, therefore they are not
        shared with the original graph.  Edge type is similar.
        """
        err_msg = "Invalid slice syntax. Use G['etype'] or G['srctype', 'etype', 'dsttype'] " +\
                  "to get view of one relation type. Use : to slice multiple types (e.g. " +\
                  "G['srctype', :, 'dsttype'])."

        orig_key = key
        if not isinstance(key, tuple):
            key = (SLICE_FULL, key, SLICE_FULL)

        if len(key) != 3:
            raise DGLError(err_msg)

        etypes = self._find_etypes(key)

        if len(etypes) == 0:
            raise DGLError('Invalid key "{}". Must be one of the edge types.'.format(orig_key))

        if len(etypes) == 1:
            # no ambiguity: return the unitgraph itself
            srctype, etype, dsttype = self._canonical_etypes[etypes[0]]
            stid = self.get_ntype_id_from_src(srctype)
            etid = self.get_etype_id((srctype, etype, dsttype))
            dtid = self.get_ntype_id_from_dst(dsttype)
            new_g = self._graph.get_relation_graph(etid)

            if stid == dtid:
                new_ntypes = [srctype]
                new_nframes = [self._node_frames[stid]]
            else:
                new_ntypes = ([srctype], [dsttype])
                new_nframes = [self._node_frames[stid], self._node_frames[dtid]]
            new_etypes = [etype]
            new_eframes = [self._edge_frames[etid]]

            return self.__class__(new_g, new_ntypes, new_etypes, new_nframes, new_eframes)
        else:
            flat = self._graph.flatten_relations(etypes)
            new_g = flat.graph

            # merge frames
            stids = flat.induced_srctype_set.asnumpy()
            dtids = flat.induced_dsttype_set.asnumpy()
            etids = flat.induced_etype_set.asnumpy()
            new_ntypes = [combine_names(self.ntypes, stids)]
            if new_g.number_of_ntypes() == 2:
                new_ntypes.append(combine_names(self.ntypes, dtids))
                new_nframes = [
                    combine_frames(self._node_frames, stids),
                    combine_frames(self._node_frames, dtids)]
            else:
                assert np.array_equal(stids, dtids)
                new_nframes = [combine_frames(self._node_frames, stids)]
            new_etypes = [combine_names(self.etypes, etids)]
            new_eframes = [combine_frames(self._edge_frames, etids)]

            # create new heterograph
            new_hg = self.__class__(new_g, new_ntypes, new_etypes, new_nframes, new_eframes)

            src = new_ntypes[0]
            dst = new_ntypes[1] if new_g.number_of_ntypes() == 2 else src
            # put the parent node/edge type and IDs
            new_hg.nodes[src].data[NTYPE] = F.zerocopy_from_dgl_ndarray(flat.induced_srctype)
            new_hg.nodes[src].data[NID] = F.zerocopy_from_dgl_ndarray(flat.induced_srcid)
            new_hg.nodes[dst].data[NTYPE] = F.zerocopy_from_dgl_ndarray(flat.induced_dsttype)
            new_hg.nodes[dst].data[NID] = F.zerocopy_from_dgl_ndarray(flat.induced_dstid)
            new_hg.edata[ETYPE] = F.zerocopy_from_dgl_ndarray(flat.induced_etype)
            new_hg.edata[EID] = F.zerocopy_from_dgl_ndarray(flat.induced_eid)

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
            in the graph. (Default: None)

        Returns
        -------
        int
            The number of nodes

        Examples
        --------

        >>> g = dgl.graph(([0, 1], [1, 2]), 'user', 'follows')
        >>> g.number_of_nodes('user')
        3
        >>> g.number_of_nodes()
        3
        """
        return self._graph.number_of_nodes(self.get_ntype_id(ntype))

    def number_of_src_nodes(self, ntype=None):
        """Return the number of nodes of the given SRC node type in the heterograph.

        The heterograph is usually a unidirectional bipartite graph.

        Parameters
        ----------
        ntype : str, optional
            Node type.
            If omitted, there should be only one node type in the SRC category.

        Returns
        -------
        int
            The number of nodes

        Examples
        --------
        >>> g = dgl.bipartite(([0, 1], [1, 2]), 'user', 'plays', 'game')
        >>> g.number_of_src_nodes('user')
        2
        >>> g.number_of_src_nodes()
        2
        >>> g.number_of_nodes('user')
        2
        """
        return self._graph.number_of_nodes(self.get_ntype_id_from_src(ntype))

    def number_of_dst_nodes(self, ntype=None):
        """Return the number of nodes of the given DST node type in the heterograph.

        The heterograph is usually a unidirectional bipartite graph.

        Parameters
        ----------
        ntype : str, optional
            Node type.
            If omitted, there should be only one node type in the DST category.

        Returns
        -------
        int
            The number of nodes

        Examples
        --------
        >>> g = dgl.bipartite(([0, 1], [1, 2]), 'user', 'plays', 'game')
        >>> g.number_of_dst_nodes('game')
        3
        >>> g.number_of_dst_nodes()
        3
        >>> g.number_of_nodes('game')
        3
        """
        return self._graph.number_of_nodes(self.get_ntype_id_from_dst(ntype))

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

        >>> g = dgl.graph(([0, 1], [1, 2]), 'user', 'follows')
        >>> g.number_of_edges(('user', 'follows', 'user'))
        2
        >>> g.number_of_edges('follows')
        2
        >>> g.number_of_edges()
        2
        """
        return self._graph.number_of_edges(self.get_etype_id(etype))

    def __len__(self):
        """Deprecated: please directly call :func:`number_of_nodes`
        """
        dgl_warning('DGLGraph.__len__ is deprecated.'
                    'Please directly call DGLGraph.number_of_nodes.')
        return self.number_of_nodes()

    @property
    def is_multigraph(self):
        """Whether the graph is a multigraph

        Returns
        -------
        bool
            True if the graph is a multigraph, False otherwise.
        """
        return self._graph.is_multigraph()

    @property
    def is_readonly(self):
        """Deprecated: DGLGraph will always be mutable.

        Returns
        -------
        bool
            True if the graph is readonly, False otherwise.
        """
        dgl_warning('DGLGraph.is_readonly is deprecated in v0.5.\n'
                    'DGLGraph now always supports mutable operations like add_nodes'
                    ' and add_edges.')
        return False

    @property
    def idtype(self):
        """The dtype of graph index

        Returns
        -------
        backend dtype object
            th.int32/th.int64 or tf.int32/tf.int64 etc.

        See Also
        --------
        long
        int
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

    def __contains__(self, vid):
        """Deprecated: please directly call :func:`has_nodes`.
        """
        dgl_warning('DGLGraph.__contains__ is deprecated.'
                    ' Please directly call has_nodes.')
        return self.has_nodes(vid)

    def has_nodes(self, vid, ntype=None):
        """Whether the graph has a node with a particular id and type.

        Parameters
        ----------
        vid : int, iterable, tensor
            Node ID(s).
        ntype : str, optional
            The node type. Can be omitted if there is only one node type
            in the graph. (Default: None)

        Returns
        -------
        bool or bool Tensor
            Each element is a bool flag, which is True if the node exists,
            and is False otherwise.

        Examples
        --------
        >>> g.has_nodes(0, 'user')
        True
        >>> g.has_nodes(4, 'user')
        False
        >>> g.has_nodes([0, 1, 2, 3, 4], 'user')
        tensor([True, True, True, False, False])
        """
        ret = self._graph.has_nodes(
            self.get_ntype_id(ntype),
            utils.prepare_tensor(self, vid, "vid"))
        if isinstance(vid, numbers.Integral):
            return bool(F.as_scalar(ret))
        else:
            return F.astype(ret, F.bool)

    def has_node(self, vid, ntype=None):
        """Whether the graph has a node with ids and a particular type.

        DEPRECATED: see :func:`~DGLGraph.has_nodes`
        """
        dgl_warning("DGLGraph.has_node is deprecated. Please use DGLGraph.has_nodes")
        return self.has_nodes(vid, ntype)

    def has_edges_between(self, u, v, etype=None):
        """Whether the graph has an edge (u, v) of type ``etype``.

        Parameters
        ----------
        u : int, iterable of int, Tensor
            Source node ID(s).
        v : int, iterable of int, Tensor
            Destination node ID(s).
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph.

        Returns
        -------
        a : Tensor
            Binary tensor indicating the existence of edges. ``a[i]=1`` if the graph
            contains edge ``(u[i], v[i])`` of type ``etype``, 0 otherwise.

        Examples
        --------

        >>> g.has_edge_between(0, 1, ('user', 'plays', 'game'))
        True
        >>> g.has_edge_between(0, 2, ('user', 'plays', 'game'))
        False
        >>> g.has_edge_between([0, 0], [1, 2], ('user', 'plays', 'game'))
        tensor([1, 0])
        """
        ret = self._graph.has_edges_between(
            self.get_etype_id(etype),
            utils.prepare_tensor(self, u, 'u'),
            utils.prepare_tensor(self, v, 'v'))
        if isinstance(u, numbers.Integral) and isinstance(v, numbers.Integral):
            return bool(F.as_scalar(ret))
        else:
            return F.astype(ret, F.bool)

    def has_edge_between(self, u, v, etype=None):
        """Whether the graph has edges of type ``etype``.

        DEPRECATED: please use :func:`~DGLGraph.has_edge_between`.
        """
        dgl_warning("DGLGraph.has_edge_between is deprecated. "
                    "Please use DGLGraph.has_edges_between")
        return self.has_edges_between(u, v, etype)

    def predecessors(self, v, etype=None):
        """Return the predecessors of node `v` in the graph with the specified
        edge type.

        Node `u` is a predecessor of `v` if an edge `(u, v)` with type `etype`
        exists in the graph.

        Parameters
        ----------
        v : int
            The destination node.
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph. (Default: None)

        Returns
        -------
        tensor
            Array of predecessor node IDs with the specified edge type.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> plays_g = dgl.bipartite(([0, 1, 1, 2], [0, 0, 1, 1]), 'user', 'plays', 'game')
        >>> devs_g = dgl.bipartite(([0, 1], [0, 1]), 'developer', 'develops', 'game')
        >>> g = dgl.hetero_from_relations([plays_g, devs_g])
        >>> g.predecessors(0, 'plays')
        tensor([0, 1])
        >>> g.predecessors(0, 'develops')
        tensor([0])

        See Also
        --------
        successors
        """
        return self._graph.predecessors(self.get_etype_id(etype), v)

    def successors(self, v, etype=None):
        """Return the successors of node `v` in the graph with the specified edge
        type.

        Node `u` is a successor of `v` if an edge `(v, u)` with type `etype` exists
        in the graph.

        Parameters
        ----------
        v : int
            The source node.
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph. (Default: None)

        Returns
        -------
        tensor
            Array of successor node IDs with the specified edge type.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> plays_g = dgl.bipartite(([0, 1, 1, 2], [0, 0, 1, 1]), 'user', 'plays', 'game')
        >>> follows_g = dgl.graph(([0, 1], [1, 2]), 'user', 'follows')
        >>> g = dgl.hetero_from_relations([plays_g, follows_g])
        >>> g.successors(0, 'plays')
        tensor([0])
        >>> g.successors(0, 'follows')
        tensor([1])

        See Also
        --------
        predecessors
        """
        return self._graph.successors(self.get_etype_id(etype), v)

    def edge_id(self, u, v, force_multi=None, return_uv=False, etype=None):
        """Return the edge ID, or an array of edge IDs, between source node
        `u` and destination node `v`, with the specified edge type

        **DEPRECATED**: See edge_ids
        """
        dgl_warning("DGLGraph.edge_id is deprecated. Please use DGLGraph.edge_ids.")
        return self.edge_ids(u, v, force_multi=force_multi,
                             return_uv=return_uv, etype=etype)

    def edge_ids(self, u, v, force_multi=None, return_uv=False, etype=None):
        """Return all edge IDs between source node array `u` and destination
        node array `v` with the specified edge type.

        Parameters
        ----------
        u : int, list, tensor
            The node ID array of source type.
        v : int, list, tensor
            The node ID array of destination type.
        force_multi : bool, optional
            Deprecated (Will be deleted in the future).
            Whether to always treat the graph as a multigraph. See the
            "Returns" for their effects. (Default: False)
        return_uv : bool
            See the "Returns" for their effects. (Default: False)
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph.

        Returns
        -------
        tensor, or (tensor, tensor, tensor)

            * If ``return_uv=False``, return a single edge ID array ``e``.
            ``e[i]`` is the edge ID between ``u[i]`` and ``v[i]``.

            * Otherwise, return three arrays ``(eu, ev, e)``.  ``e[i]`` is the ID
            of an edge between ``eu[i]`` and ``ev[i]``.  All edges between ``u[i]``
            and ``v[i]`` are returned.

        Notes
        -----
        If the graph is a simple graph, ``return_uv=False``, and no edge
        exists between some pairs of ``u[i]`` and ``v[i]``, the result is undefined
        and an empty tensor is returned.

        If the graph is a multi graph, ``return_uv=False``, and multi edges
        exist between some pairs of `u[i]` and `v[i]`, the result is undefined.

        Examples
        --------
        The following example uses PyTorch backend.

        Instantiate a heterograph.

        >>> plays_g = dgl.bipartite(([0, 1, 1, 2], [0, 0, 2, 1]), 'user', 'plays', 'game')
        >>> follows_g = dgl.graph(([0, 1, 1], [1, 2, 2]), 'user', 'follows')
        >>> g = dgl.hetero_from_relations([plays_g, follows_g])

        Query for edge ids.

        >>> plays_g.edge_ids([0], [2], etype=('user', 'plays', 'game'))
        tensor([], dtype=torch.int64)
        >>> plays_g.edge_ids([1], [2], etype=('user', 'plays', 'game'))
        tensor([2])
        >>> g.edge_ids([1], [2], return_uv=True, etype=('user', 'follows', 'user'))
        (tensor([1, 1]), tensor([2, 2]), tensor([1, 2]))
        """
        is_int = isinstance(u, numbers.Integral) and isinstance(v, numbers.Integral)
        u = utils.prepare_tensor(self, u, 'u')
        v = utils.prepare_tensor(self, v, 'v')
        if force_multi is not None:
            dgl_warning("force_multi will be deprecated, " \
                        "Please use return_uv instead")
            return_uv = force_multi

        if return_uv:
            return self._graph.edge_ids_all(self.get_etype_id(etype), u, v)
        else:
            eid = self._graph.edge_ids_one(self.get_etype_id(etype), u, v)
            is_neg_one = F.equal(eid, -1)
            if F.as_scalar(F.sum(is_neg_one, 0)):
                # Raise error since some (u, v) pair is not a valid edge.
                idx = F.nonzero_1d(is_neg_one)
                raise DGLError("Error: (%d, %d) does not form a valid edge." % (
                    F.as_scalar(F.gather_row(u, idx)),
                    F.as_scalar(F.gather_row(v, idx))))
            return F.as_scalar(eid) if is_int else eid

    def find_edges(self, eid, etype=None):
        """Given an edge ID array with the specified type, return the source
        and destination node ID array ``s`` and ``d``.  ``s[i]`` and ``d[i]``
        are source and destination node ID for edge ``eid[i]``.

        Parameters
        ----------
        eid : list, tensor
            The edge ID array.
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph. (Default: None)

        Returns
        -------
        tensor
            The source node ID array.
        tensor
            The destination node ID array.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> g = dgl.bipartite(([0, 1, 1], [0, 0, 2]), 'user', 'plays', 'game')
        >>> g.find_edges([0, 2], ('user', 'plays', 'game'))
        (tensor([0, 1]), tensor([0, 2]))
        >>> g.find_edges([0, 2])
        (tensor([0, 1]), tensor([0, 2]))
        """
        eid = utils.prepare_tensor(self, eid, 'eid')
        if len(eid) == 0:
            empty = F.copy_to(F.tensor([], self.idtype), self.device)
            return empty, empty
        # sanity check
        max_eid = F.as_scalar(F.max(eid, dim=0))
        if max_eid >= self.number_of_edges(etype):
            raise DGLError('Expect edge IDs to be smaller than number of edges ({}). '
                           ' But got {}.'.format(self.number_of_edges(etype), max_eid))
        src, dst, _ = self._graph.find_edges(self.get_etype_id(etype), eid)
        return src, dst

    def in_edges(self, v, form='uv', etype=None):
        """Return the inbound edges of the node(s) with the specified type.

        Parameters
        ----------
        v : int, list, tensor
            The node id(s) of destination type.
        form : str, optional
            The return form. Currently support:

            - ``'eid'`` : one eid tensor
            - ``'all'`` : a tuple ``(u, v, eid)``
            - ``'uv'``  : a pair ``(u, v)``, default
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph. (Default: None)

        Returns
        -------
        tensor or (tensor, tensor, tensor) or (tensor, tensor)
            All inbound edges to ``v`` are returned.

            * If ``form='eid'``, return a tensor for the ids of the
              inbound edges of the nodes with the specified type.
            * If ``form='all'``, return a 3-tuple of tensors
              ``(eu, ev, eid)``. ``eid[i]`` gives the ID of the
              edge from ``eu[i]`` to ``ev[i]``.
            * If ``form='uv'``, return a 2-tuple of tensors ``(eu, ev)``.
              ``eu[i]`` is the source node of an edge to ``ev[i]``.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> g = dgl.bipartite(([0, 1, 1], [0, 1, 2]), 'user', 'plays', 'game')
        >>> g.in_edges([0, 2], form='eid')
        tensor([0, 2])
        >>> g.in_edges([0, 2], form='all')
        (tensor([0, 1]), tensor([0, 2]), tensor([0, 2]))
        >>> g.in_edges([0, 2], form='uv')
        (tensor([0, 1]), tensor([0, 2]))
        """
        v = utils.prepare_tensor(self, v, 'v')
        src, dst, eid = self._graph.in_edges(self.get_etype_id(etype), v)
        if form == 'all':
            return src, dst, eid
        elif form == 'uv':
            return src, dst
        elif form == 'eid':
            return eid
        else:
            raise DGLError('Invalid form: {}. Must be "all", "uv" or "eid".'.format(form))

    def out_edges(self, u, form='uv', etype=None):
        """Return the outbound edges of the node(s) with the specified type.

        Parameters
        ----------
        u : int, list, tensor
            The node id(s) of source type.
        form : str, optional
            The return form. Currently support:

            - ``'eid'`` : one eid tensor
            - ``'all'`` : a tuple ``(u, v, eid)``
            - ``'uv'``  : a pair ``(u, v)``, default
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph. (Default: None)

        Returns
        -------
        tensor or (tensor, tensor, tensor) or (tensor, tensor)
            All outbound edges from ``u`` are returned.

            * If ``form='eid'``, return a tensor for the ids of the outbound edges
              of the nodes with the specified type.
            * If ``form='all'``, return a 3-tuple of tensors ``(eu, ev, eid)``.
              ``eid[i]`` gives the ID of the edge from ``eu[i]`` to ``ev[i]``.
            * If ``form='uv'``, return a 2-tuple of tensors ``(eu, ev)``.
              ``ev[i]`` is the destination node of the edge from ``eu[i]``.

        Examples
        --------

        >>> g = dgl.bipartite(([0, 1, 1], [0, 1, 2]), 'user', 'plays', 'game')
        >>> g.out_edges([0, 1], form='eid')
        tensor([0, 1, 2])
        >>> g.out_edges([0, 1], form='all')
        (tensor([0, 1, 1]), tensor([0, 1, 2]), tensor([0, 1, 2]))
        >>> g.out_edges([0, 1], form='uv')
        (tensor([0, 1, 1]), tensor([0, 1, 2]))
        """
        u = utils.prepare_tensor(self, u, 'u')
        src, dst, eid = self._graph.out_edges(self.get_etype_id(etype), u)
        if form == 'all':
            return src, dst, eid
        elif form == 'uv':
            return src, dst
        elif form == 'eid':
            return eid
        else:
            raise DGLError('Invalid form: {}. Must be "all", "uv" or "eid".'.format(form))

    def all_edges(self, form='uv', order=None, etype=None):
        """Return all edges with the specified type.

        Parameters
        ----------
        form : str, optional
            The return form. Currently support:

            - ``'eid'`` : one eid tensor
            - ``'all'`` : a tuple ``(u, v, eid)``
            - ``'uv'``  : a pair ``(u, v)``, default
        order : str or None
            The order of the returned edges. Currently support:

            - ``'srcdst'`` : sorted by their src and dst ids.
            - ``'eid'``    : sorted by edge Ids.
            - ``None``     : arbitrary order, default
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph. (Default: None)

        Returns
        -------
        tensor or (tensor, tensor, tensor) or (tensor, tensor)

            * If ``form='eid'``, return a tensor for the ids of all edges
              with the specified type.
            * If ``form='all'``, return a 3-tuple of tensors ``(eu, ev, eid)``.
              ``eid[i]`` gives the ID of the edge from ``eu[i]`` to ``ev[i]``.
            * If ``form='uv'``, return a 2-tuple of tensors ``(eu, ev)``.
              ``ev[i]`` is the destination node of the edge from ``eu[i]``.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> g = dgl.bipartite(([1, 0, 1], [1, 0, 2]), 'user', 'plays', 'game')
        >>> g.all_edges(form='eid', order='srcdst')
        tensor([1, 0, 2])
        >>> g.all_edges(form='all', order='srcdst')
        (tensor([0, 1, 1]), tensor([0, 1, 2]), tensor([1, 0, 2]))
        >>> g.all_edges(form='uv', order='eid')
        (tensor([1, 0, 1]), tensor([1, 0, 2]))
        """
        src, dst, eid = self._graph.edges(self.get_etype_id(etype), order)
        if form == 'all':
            return src, dst, eid
        elif form == 'uv':
            return src, dst
        elif form == 'eid':
            return eid
        else:
            raise DGLError('Invalid form: {}. Must be "all", "uv" or "eid".'.format(form))

    def in_degree(self, v, etype=None):
        """Return the in-degree of node ``v`` with edges of type ``etype``.

        DEPRECATED: Please use in_degrees
        """
        dgl_warning("DGLGraph.in_degree is deprecated. Please use DGLGraph.in_degrees")
        return self.in_degrees(v, etype)

    def in_degrees(self, v=ALL, etype=None):
        """Return the in-degrees of nodes v with edges of type ``etype``.

        Parameters
        ----------
        v : int, iterable of int or tensor, optional.
            The node ID array of the destination type. Default is to return the
            degrees of all nodes.
        etype : str or tuple of str or None, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph. (Default: None)

        Returns
        -------
        d : tensor or int
            The in-degree array. ``d[i]`` gives the in-degree of node ``v[i]``
            with edges of type ``etype``. If the argument is an integer, so will
            be the return.

        Examples
        --------
        The following example uses PyTorch backend.

        Instantiate a heterograph.

        >>> plays_g = dgl.bipartite(([0, 1, 1, 2], [0, 0, 2, 1]), 'user', 'plays', 'game')
        >>> follows_g = dgl.graph(([0, 1, 1], [1, 2, 2]), 'user', 'follows')
        >>> g = dgl.hetero_from_relations([plays_g, follows_g])

        Query for node degree.

        >>> g.in_degrees(0, 'plays')
        2
        >>> g.in_degrees(etype='follows')
        tensor([0, 1, 2])
        """
        dsttype = self.to_canonical_etype(etype)[2]
        etid = self.get_etype_id(etype)
        if is_all(v):
            v = self.dstnodes(dsttype)
        deg = self._graph.in_degrees(etid, utils.prepare_tensor(self, v, 'v'))
        if isinstance(v, numbers.Integral):
            return F.as_scalar(deg)
        else:
            return deg

    def out_degree(self, u, etype=None):
        """Return the out-degree of node `u` with edges of type ``etype``.

        DEPRECATED: please use DGL.out_degrees
        """
        dgl_warning("DGLGraph.out_degree is deprecated. Please use DGLGraph.out_degrees")
        return self.out_degrees(u, etype)

    def out_degrees(self, u=ALL, etype=None):
        """Return the out-degrees of nodes u with edges of type ``etype``.

        Parameters
        ----------
        u : list, tensor
            The node ID array of source type. Default is to return the degrees
            of all the nodes.
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph. (Default: None)

        Returns
        -------
        d : tensor
            The out-degree array. ``d[i]`` gives the out-degree of node ``u[i]``
            with edges of type ``etype``.

        Examples
        --------
        The following example uses PyTorch backend.

        Instantiate a heterograph.

        >>> plays_g = dgl.bipartite(([0, 1, 1, 2], [0, 0, 2, 1]), 'user', 'plays', 'game')
        >>> follows_g = dgl.graph(([0, 1, 1], [1, 2, 2]), 'user', 'follows')
        >>> g = dgl.hetero_from_relations([plays_g, follows_g])

        Query for node degree.

        >>> g.out_degrees(0, 'plays')
        1
        >>> g.out_degrees(etype='follows')
        tensor([1, 2, 0])

        See Also
        --------
        out_degree
        """
        srctype = self.to_canonical_etype(etype)[0]
        etid = self.get_etype_id(etype)
        if is_all(u):
            u = self.srcnodes(srctype)
        deg = self._graph.out_degrees(etid, utils.prepare_tensor(self, u, 'u'))
        if isinstance(u, numbers.Integral):
            return F.as_scalar(deg)
        else:
            return deg

    def adjacency_matrix(self, transpose=None, ctx=F.cpu(), scipy_fmt=None, etype=None):
        """Return the adjacency matrix of edges of the given edge type.

        By default, a row of returned adjacency matrix represents the
        destination of an edge and the column represents the source.

        When transpose is True, a row represents the source and a column
        represents a destination.

        Parameters
        ----------
        transpose : bool, optional
            A flag to transpose the returned adjacency matrix. (Default: False)
        ctx : context, optional
            The context of returned adjacency matrix. (Default: cpu)
        scipy_fmt : str, optional
            If specified, return a scipy sparse matrix in the given format.
            Otherwise, return a backend dependent sparse tensor. (Default: None)
        etype : str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph. (Default: None)

        Returns
        -------
        SparseTensor or scipy.sparse.spmatrix
            Adjacency matrix.

        Examples
        --------

        Instantiate a heterogeneous graph.

        >>> follows_g = dgl.graph(([0, 1], [0, 1]), 'user', 'follows')
        >>> devs_g = dgl.bipartite(([0, 1], [0, 2]), 'developer', 'develops', 'game')
        >>> g = dgl.hetero_from_relations([follows_g, devs_g])

        Get a backend dependent sparse tensor. Here we use PyTorch for example.

        >>> g.adjacency_matrix(etype='develops')
        tensor(indices=tensor([[0, 2],
                               [0, 1]]),
               values=tensor([1., 1.]),
               size=(3, 2), nnz=2, layout=torch.sparse_coo)

        Get a scipy coo sparse matrix.

        >>> g.adjacency_matrix(scipy_fmt='coo', etype='develops')
        <3x2 sparse matrix of type '<class 'numpy.int64'>'
        with 2 stored elements in COOrdinate format>
        """
        if transpose is None:
            dgl_warning(
                "Currently adjacency_matrix() returns a matrix with destination as rows"
                " by default.\n\tIn 0.5 the result will have source as rows"
                " (i.e. transpose=True)")
            transpose = False

        etid = self.get_etype_id(etype)
        if scipy_fmt is None:
            return self._graph.adjacency_matrix(etid, transpose, ctx)[0]
        else:
            return self._graph.adjacency_matrix_scipy(etid, transpose, scipy_fmt, False)

    # Alias of ``adjacency_matrix``
    adj = adjacency_matrix

    def adjacency_matrix_scipy(self, transpose=None, fmt='csr', return_edge_ids=None):
        """DEPRECATED: please use ``dgl.adjacency_matrix(transpose, scipy_fmt=fmt)``.
        """
        dgl_warning('DGLGraph.adjacency_matrix_scipy is deprecated. '
                    'Please replace it with:\n\n\t'
                    'DGLGraph.adjacency_matrix(transpose, scipy_fmt="{}").\n'.format(fmt))

        return self.adjacency_matrix(transpose=transpose, scipy_fmt=fmt)

    def incidence_matrix(self, typestr, ctx=F.cpu(), etype=None):
        """Return the incidence matrix representation of edges with the given
        edge type.

        An incidence matrix is an n-by-m sparse matrix, where n is
        the number of nodes and m is the number of edges. Each nnz
        value indicating whether the edge is incident to the node
        or not.

        There are three types of incidence matrices :math:`I`:

        * ``in``:

            - :math:`I[v, e] = 1` if :math:`e` is the in-edge of :math:`v`
              (or :math:`v` is the dst node of :math:`e`);
            - :math:`I[v, e] = 0` otherwise.

        * ``out``:

            - :math:`I[v, e] = 1` if :math:`e` is the out-edge of :math:`v`
              (or :math:`v` is the src node of :math:`e`);
            - :math:`I[v, e] = 0` otherwise.

        * ``both`` (only if source and destination node type are the same):

            - :math:`I[v, e] = 1` if :math:`e` is the in-edge of :math:`v`;
            - :math:`I[v, e] = -1` if :math:`e` is the out-edge of :math:`v`;
            - :math:`I[v, e] = 0` otherwise (including self-loop).

        Parameters
        ----------
        typestr : str
            Can be either ``in``, ``out`` or ``both``
        ctx : context, optional
            The context of returned incidence matrix. (Default: cpu)
        etype : str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph.

        Returns
        -------
        Framework SparseTensor
            The incidence matrix.

        Examples
        --------

        >>> g = dgl.graph(([0, 1], [0, 2]), 'user', 'follows')
        >>> g.incidence_matrix('in')
        tensor(indices=tensor([[0, 2],
                               [0, 1]]),
               values=tensor([1., 1.]),
               size=(3, 2), nnz=2, layout=torch.sparse_coo)
        >>> g.incidence_matrix('out')
        tensor(indices=tensor([[0, 1],
                               [0, 1]]),
               values=tensor([1., 1.]),
               size=(3, 2), nnz=2, layout=torch.sparse_coo)
        >>> g.incidence_matrix('both')
        tensor(indices=tensor([[1, 2],
                               [1, 1]]),
               values=tensor([-1.,  1.]),
               size=(3, 2), nnz=2, layout=torch.sparse_coo)
        """
        etid = self.get_etype_id(etype)
        return self._graph.incidence_matrix(etid, typestr, ctx)[0]

    # Alias of ``incidence_matrix``
    inc = incidence_matrix

    #################################################################
    # Features
    #################################################################

    def node_attr_schemes(self, ntype=None):
        """Return the node feature schemes for the specified type.

        Each feature scheme is a named tuple that stores the shape and data type
        of the node feature.

        Parameters
        ----------
        ntype : str, optional
            The node type. Can be omitted if there is only one node
            type in the graph. Error will be raised otherwise.
            (Default: None)

        Returns
        -------
        dict of str to schemes
            The schemes of node feature columns.

        Examples
        --------
        The following uses PyTorch backend.

        >>> g = dgl.graph(([0, 1], [0, 2]), 'user', 'follows')
        >>> g.nodes['user'].data['h'] = torch.randn(3, 4)
        >>> g.node_attr_schemes('user')
        {'h': Scheme(shape=(4,), dtype=torch.float32)}

        See Also
        --------
        edge_attr_schemes
        """
        return self._node_frames[self.get_ntype_id(ntype)].schemes

    def edge_attr_schemes(self, etype=None):
        """Return the edge feature schemes for the specified type.

        Each feature scheme is a named tuple that stores the shape and data type
        of the edge feature.

        Parameters
        ----------
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph. (Default: None)

        Returns
        -------
        dict of str to schemes
            The schemes of edge feature columns.

        Examples
        --------
        The following uses PyTorch backend.

        >>> g = dgl.bipartite(([0, 1, 1, 2], [0, 0, 2, 1]), 'user', 'plays', 'game')
        >>> g.edges['user', 'plays', 'game'].data['h'] = torch.randn(4, 4)
        >>> g.edge_attr_schemes(('user', 'plays', 'game'))
        {'h': Scheme(shape=(4,), dtype=torch.float32)}

        See Also
        --------
        node_attr_schemes
        """
        return self._edge_frames[self.get_etype_id(etype)].schemes

    def set_n_initializer(self, initializer, field=None, ntype=None):
        """Set the initializer for empty node features.

        Initializer is a callable that returns a tensor given the shape, data type
        and device context.

        When a subset of the nodes are assigned a new feature, initializer is
        used to create feature for the rest of the nodes.

        Parameters
        ----------
        initializer : callable
            The initializer, mapping (shape, data type, context) to tensor.
        field : str, optional
            The feature field name. Default is to set an initializer for all the
            feature fields.
        ntype : str, optional
            The node type. Can be omitted if there is only one node
            type in the graph. Error will be raised otherwise.
            (Default: None)

        Note
        -----
        User defined initializer must follow the signature of
        :func:`dgl.init.base_initializer() <dgl.init.base_initializer>`

        See Also
        --------
        set_e_initializer
        """
        ntid = self.get_ntype_id(ntype)
        self._node_frames[ntid].set_initializer(initializer, field)

    def set_e_initializer(self, initializer, field=None, etype=None):
        """Set the initializer for empty edge features.

        Initializer is a callable that returns a tensor given the shape, data
        type and device context.

        When a subset of the edges are assigned a new feature, initializer is
        used to create feature for rest of the edges.

        Parameters
        ----------
        initializer : callable
            The initializer, mapping (shape, data type, context) to tensor.
        field : str, optional
            The feature field name. Default is set an initializer for all the
            feature fields.
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph. Error will be raised otherwise.
            (Default: None)

        Note
        -----
        User defined initializer must follow the signature of
        :func:`dgl.init.base_initializer() <dgl.init.base_initializer>`

        See Also
        --------
        set_n_initializer
        """
        etid = self.get_etype_id(etype)
        self._edge_frames[etid].set_initializer(initializer, field)

    def _set_n_repr(self, ntid, u, data):
        """Internal API to set node features.

        `data` is a dictionary from the feature name to feature tensor. Each tensor
        is of shape (B, D1, D2, ...), where B is the number of nodes to be updated,
        and (D1, D2, ...) be the shape of the node representation tensor. The
        length of the given node ids must match B (i.e, len(u) == B).

        All updates will be done out of place to work with autograd.

        Parameters
        ----------
        ntid : int
            Node type id.
        u : node, container or tensor
            The node(s).
        data : dict of tensor
            Node representation.
        """
        if is_all(u):
            num_nodes = self._graph.number_of_nodes(ntid)
        else:
            u = utils.prepare_tensor(self, u, 'u')
            num_nodes = len(u)
        for key, val in data.items():
            nfeats = F.shape(val)[0]
            if nfeats != num_nodes:
                raise DGLError('Expect number of features to match number of nodes (len(u)).'
                               ' Got %d and %d instead.' % (nfeats, num_nodes))
            if F.context(val) != self.device:
                raise DGLError('Cannot assign node feature "{}" on device {} to a graph on'
                               ' device {}. Call DGLGraph.to() to copy the graph to the'
                               ' same device.'.format(key, F.context(val), self.device))

        if is_all(u):
            self._node_frames[ntid].update(data)
        else:
            self._node_frames[ntid].update_row(u, data)

    def _get_n_repr(self, ntid, u):
        """Get node(s) representation of a single node type.

        The returned feature tensor batches multiple node features on the first dimension.

        Parameters
        ----------
        ntid : int
            Node type id.
        u : node, container or tensor
            The node(s).

        Returns
        -------
        dict
            Representation dict from feature name to feature tensor.
        """
        if is_all(u):
            return self._node_frames[ntid]
        else:
            u = utils.prepare_tensor(self, u, 'u')
            return self._node_frames[ntid].subframe(u)

    def _pop_n_repr(self, ntid, key):
        """Internal API to get and remove the specified node feature.

        Parameters
        ----------
        ntid : int
            Node type id.
        key : str
            The attribute name.

        Returns
        -------
        Tensor
            The popped representation
        """
        return self._node_frames[ntid].pop(key)

    def _set_e_repr(self, etid, edges, data):
        """Internal API to set edge(s) features.

        `data` is a dictionary from the feature name to feature tensor. Each tensor
        is of shape (B, D1, D2, ...), where B is the number of edges to be updated,
        and (D1, D2, ...) be the shape of the edge representation tensor.

        All update will be done out of place to work with autograd.

        Parameters
        ----------
        etid : int
            Edge type id.
        edges : edges
            Edges can be either

            * A pair of endpoint nodes (u, v), where u is the node ID of source
              node type and v is that of destination node type.
            * A tensor of edge ids of the given type.

            The default value is all the edges.
        data : tensor or dict of tensor
            Edge representation.
        """
        # parse argument
        if not is_all(edges):
            eid = utils.parse_edges_arg_to_eid(self, edges, etid, 'edges')

        # sanity check
        if not utils.is_dict_like(data):
            raise DGLError('Expect dictionary type for feature data.'
                           ' Got "%s" instead.' % type(data))

        if is_all(edges):
            num_edges = self._graph.number_of_edges(etid)
        else:
            num_edges = len(eid)
        for key, val in data.items():
            nfeats = F.shape(val)[0]
            if nfeats != num_edges:
                raise DGLError('Expect number of features to match number of edges.'
                               ' Got %d and %d instead.' % (nfeats, num_edges))
            if F.context(val) != self.device:
                raise DGLError('Cannot assign edge feature "{}" on device {} to a graph on'
                               ' device {}. Call DGLGraph.to() to copy the graph to the'
                               ' same device.'.format(key, F.context(val), self.device))

        # set
        if is_all(edges):
            self._edge_frames[etid].update(data)
        else:
            self._edge_frames[etid].update_row(eid, data)

    def _get_e_repr(self, etid, edges):
        """Internal API to get edge features.

        Parameters
        ----------
        etid : int
            Edge type id.
        edges : edges
            Edges can be a pair of endpoint nodes (u, v), or a
            tensor of edge ids. The default value is all the edges.

        Returns
        -------
        dict
            Representation dict
        """
        # parse argument
        if is_all(edges):
            return dict(self._edge_frames[etid])
        else:
            eid = utils.parse_edges_arg_to_eid(self, edges, etid, 'edges')
            return self._edge_frames[etid].subframe(eid)

    def _pop_e_repr(self, etid, key):
        """Get and remove the specified edge repr of a single edge type.

        Parameters
        ----------
        etid : int
            Edge type id.
        key : str
          The attribute name.

        Returns
        -------
        Tensor
            The popped representation
        """
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
        func : callable or None
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        v : int or iterable of int or tensor, optional
            The (type-specific) node (ids) on which to apply ``func``. (Default: ALL)
        ntype : str, optional
            The node type. Can be omitted if there is only one node type
            in the graph. (Default: None)
        inplace : bool, optional
            **DEPRECATED**. If True, update will be done in place, but autograd will break.
            (Default: False)

        Examples
        --------
        >>> g = dgl.graph(([0, 1], [1, 2]), 'user', 'follows')
        >>> g.nodes['user'].data['h'] = torch.ones(3, 5)
        >>> g.apply_nodes(lambda nodes: {'h': nodes.data['h'] * 2}, ntype='user')
        >>> g.nodes['user'].data['h']
        tensor([[2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.]])

        See Also
        --------
        apply_edges
        """
        if inplace:
            raise DGLError('The `inplace` option is removed in v0.5.')
        ntid = self.get_ntype_id(ntype)
        ntype = self.ntypes[ntid]
        if is_all(v):
            v = self.nodes(ntype)
        else:
            v = utils.prepare_tensor(self, v, 'v')
        ndata = core.invoke_node_udf(self, v, ntype, func, orig_nid=v)
        self._set_n_repr(ntid, v, ndata)

    def apply_edges(self, func, edges=ALL, etype=None, inplace=False):
        """Apply the function on the edges with the same type to update their
        features.

        If None is provided for ``func``, nothing will happen.

        Parameters
        ----------
        func : callable
            Apply function on the edge. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        edges : optional
            Edges on which to apply ``func``. See :func:`send` for valid
            edge specification. (Default: ALL)
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph. (Default: None)
        inplace: bool, optional
            **DEPRECATED**. Must be False.

        Examples
        --------
        >>> g = dgl.bipartite(([0, 1, 1, 2], [0, 0, 2, 1]), 'user', 'plays', 'game')
        >>> g.edges[('user', 'plays', 'game')].data['h'] = torch.ones(4, 5)
        >>> g.apply_edges(lambda edges: {'h': edges.data['h'] * 2})
        >>> g.edges[('user', 'plays', 'game')].data['h']
        tensor([[2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.]])

        See Also
        --------
        apply_nodes
        """
        if inplace:
            raise DGLError('The `inplace` option is removed in v0.5.')
        etid = self.get_etype_id(etype)
        etype = self.canonical_etypes[etid]
        g = self if etype is None else self[etype]
        if is_all(edges):
            eid = ALL
        else:
            eid = utils.parse_edges_arg_to_eid(self, edges, etid, 'edges')
        if core.is_builtin(func):
            if not is_all(eid):
                g = g.edge_subgraph(eid, preserve_nodes=True)
            edata = core.invoke_gsddmm(g, func)
        else:
            edata = core.invoke_edge_udf(g, eid, etype, func)
        self._set_e_repr(etid, eid, edata)

    def send_and_recv(self,
                      edges,
                      message_func,
                      reduce_func,
                      apply_node_func=None,
                      etype=None,
                      inplace=False):
        """Send messages along edges of the specified type, and let destinations
        receive them.

        Optionally, apply a function to update the node features after "receive".

        This is a convenient combination for performing
        :mod:`send <dgl.DGLHeteroGraph.send>` along the ``edges`` and
        :mod:`recv <dgl.DGLHeteroGraph.recv>` for the destinations of the ``edges``.

        **Only works if the graph has one edge type.**  For multiple types, use

        .. code::

           g['edgetype'].send_and_recv(edges, message_func, reduce_func,
                                       apply_node_func, inplace=inplace)

        Parameters
        ----------
        edges : See :func:`send` for valid edge specification.
            Edges on which to apply ``func``.
        message_func : callable
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        reduce_func : callable
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        apply_node_func : callable, optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`. (Default: None)
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph. (Default: None)
        inplace: bool, optional
            **DEPRECATED**. Must be False.

        Examples
        --------

        >>> import dgl
        >>> import dgl.function as fn
        >>> import torch

        >>> follows_g = dgl.graph(([0, 1], [1, 2]), 'user', 'follows')
        >>> plays_g = dgl.bipartite(([0, 1, 1, 2], [0, 0, 1, 1]), 'user', 'plays', 'game')
        >>> g = dgl.hetero_from_relations([follows_g, plays_g])
        >>> g.nodes['user'].data['h'] = torch.tensor([[0.], [1.], [2.]])
        >>> g.send_and_recv(g['follows'].edges(), fn.copy_src('h', 'm'),
        >>>                 fn.sum('m', 'h'), etype='follows')
        >>> g.nodes['user'].data['h']
        tensor([[0.],
                [0.],
                [1.]])
        """
        if inplace:
            raise DGLError('The `inplace` option is removed in v0.5.')
        # edge type
        etid = self.get_etype_id(etype)
        _, dtid = self._graph.metagraph.find_edge(etid)
        etype = self.canonical_etypes[etid]
        # edge IDs
        eid = utils.parse_edges_arg_to_eid(self, edges, etid, 'edges')
        if len(eid) == 0:
            # no computation
            return
        u, v = self.find_edges(eid, etype=etype)
        # call message passing onsubgraph
        ndata = core.message_passing(_create_compute_graph(self, u, v, eid),
                                     message_func, reduce_func, apply_node_func)
        dstnodes = F.unique(v)
        self._set_n_repr(dtid, dstnodes, ndata)

    def pull(self,
             v,
             message_func,
             reduce_func,
             apply_node_func=None,
             etype=None,
             inplace=False):
        """Pull messages from the node(s)' predecessors and then update their features.

        Optionally, apply a function to update the node features after receive.

        This is equivalent to :mod:`send_and_recv <dgl.DGLHeteroGraph.send_and_recv>`
        on the incoming edges of ``v`` with the specified type.

        Other notes:

        * `reduce_func` will be skipped for nodes with no incoming messages.
        * If all ``v`` have no incoming message, this will downgrade to an :func:`apply_nodes`.
        * If some ``v`` have no incoming message, their new feature value will be calculated
          by the column initializer (see :func:`set_n_initializer`). The feature shapes and
          dtypes will be inferred.

        **Only works if the graph has one edge type.** For multiple types, use

        .. code::

           g['edgetype'].pull(v, message_func, reduce_func, apply_node_func, inplace=inplace)

        Parameters
        ----------
        v : int, container or tensor, optional
            The node(s) to be updated.
        message_func : callable
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        reduce_func : callable
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        apply_node_func : callable, optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`. (Default: None)
        etype : str or tuple of str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph. (Default: None)
        inplace: bool, optional
            **DEPRECATED**. Must be False.

        Examples
        --------

        >>> import dgl
        >>> import dgl.function as fn
        >>> import torch

        Instantiate a heterograph.

        >>> follows_g = dgl.graph(([0, 1], [1, 2]), 'user', 'follows')
        >>> plays_g = dgl.bipartite(([0, 2], [0, 1]), 'user', 'plays', 'game')
        >>> g = dgl.hetero_from_relations([follows_g, plays_g])
        >>> g.nodes['user'].data['h'] = torch.tensor([[0.], [1.], [2.]])

        Pull.

        >>> g['follows'].pull(2, fn.copy_src('h', 'm'), fn.sum('m', 'h'), etype='follows')
        >>> g.nodes['user'].data['h']
        tensor([[0.],
                [1.],
                [1.]])
        """
        if inplace:
            raise DGLError('The `inplace` option is removed in v0.5.')
        v = utils.prepare_tensor(self, v, 'v')
        if len(v) == 0:
            # no computation
            return
        etid = self.get_etype_id(etype)
        _, dtid = self._graph.metagraph.find_edge(etid)
        etype = self.canonical_etypes[etid]
        g = self if etype is None else self[etype]
        # call message passing on subgraph
        src, dst, eid = g.in_edges(v, form='all')
        ndata = core.message_passing(_create_compute_graph(self, src, dst, eid, v),
                                     message_func, reduce_func, apply_node_func)
        self._set_n_repr(dtid, v, ndata)

    def push(self,
             u,
             message_func,
             reduce_func,
             apply_node_func=None,
             etype=None,
             inplace=False):
        """Send message from the node(s) to their successors and update them.

        This is equivalent to performing
        :mod:`send_and_recv <DGLHeteroGraph.send_and_recv>` along the outbound
        edges from ``u``.

        **Only works if the graph has one edge type.** For multiple types, use

        .. code::

           g['edgetype'].push(u, message_func, reduce_func, apply_node_func, inplace=inplace)

        Parameters
        ----------
        u : int, container or tensor
            The node(s) to push out messages.
        message_func : callable
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        reduce_func : callable
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        apply_node_func : callable, optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`. (Default: None)
        etype : str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph. (Default: None)
        inplace: bool, optional
            **DEPRECATED**. Must be False.

        Examples
        --------

        >>> import dgl
        >>> import dgl.function as fn
        >>> import torch

        Instantiate a heterograph.

        >>> g = dgl.graph(([0, 0], [1, 2]), 'user', 'follows')
        >>> g.nodes['user'].data['h'] = torch.tensor([[0.], [1.], [2.]])

        Push.

        >>> g['follows'].push(0, fn.copy_src('h', 'm'), fn.sum('m', 'h'), etype='follows')
        >>> g.nodes['user'].data['h']
        tensor([[0.],
                [0.],
                [0.]])
        """
        if inplace:
            raise DGLError('The `inplace` option is removed in v0.5.')
        edges = self.out_edges(u, form='eid', etype=etype)
        self.send_and_recv(edges, message_func, reduce_func, apply_node_func, etype=etype)

    def update_all(self,
                   message_func,
                   reduce_func,
                   apply_node_func=None,
                   etype=None):
        """Send messages through all edges and update all nodes.

        Optionally, apply a function to update the node features after receive.

        This is equivalent to
        :mod:`send_and_recv <dgl.DGLHeteroGraph.send_and_recv>` over all edges
        of the specified type.

        **Only works if the graph has one edge type.** For multiple types, use

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
            a :mod:`Node UDF <dgl.udf>`. (Default: None)
        etype : str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph. (Default: None)

        Examples
        --------
        >>> import torch
        >>> import dgl
        >>> import dgl.function as fn

        Instantiate a heterograph.

        >>> g = dgl.graph(([0, 1, 2], [1, 2, 2]), 'user', 'follows')

        Update all.

        >>> g.nodes['user'].data['h'] = torch.tensor([[0.], [1.], [2.]])
        >>> g['follows'].update_all(fn.copy_src('h', 'm'), fn.sum('m', 'h'), etype='follows')
        >>> g.nodes['user'].data['h']
        tensor([[0.],
                [0.],
                [3.]])
        """
        etid = self.get_etype_id(etype)
        etype = self.canonical_etypes[etid]
        _, dtid = self._graph.metagraph.find_edge(etid)
        g = self if etype is None else self[etype]
        ndata = core.message_passing(g, message_func, reduce_func, apply_node_func)
        self._set_n_repr(dtid, ALL, ndata)

    #################################################################
    # Message passing on heterograph
    #################################################################

    def multi_update_all(self, etype_dict, cross_reducer, apply_node_func=None):
        r"""Send and receive messages along all edges.

        This is equivalent to
        :mod:`multi_send_and_recv <dgl.DGLHeteroGraph.multi_send_and_recv>`
        over all edges.

        Parameters
        ----------
        etype_dict : dict
            Mapping an edge type (str or tuple of str) to the type specific
            configuration (3-tuples). Each 3-tuple represents
            (msg_func, reduce_func, apply_node_func):

            * msg_func: callable
                  Message function on the edges. The function should be
                  an :mod:`Edge UDF <dgl.udf>`.
            * reduce_func: callable
                  Reduce function on the nodes. The function should be
                  a :mod:`Node UDF <dgl.udf>`.
            * apply_node_func : callable, optional
                  Apply function on the nodes. The function should be
                  a :mod:`Node UDF <dgl.udf>`. (Default: None)
        cross_reducer : str
            Cross type reducer. One of ``"sum"``, ``"min"``, ``"max"``, ``"mean"``, ``"stack"``.
        apply_node_func : callable
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`. (Default: None)
        inplace: bool, optional
            **DEPRECATED**. Must be False.

        Examples
        --------
        >>> import dgl
        >>> import dgl.function as fn
        >>> import torch

        Instantiate a heterograph.

        >>> g1 = dgl.graph(([0, 1], [1, 1]), 'user', 'follows')
        >>> g2 = dgl.bipartite(([0], [1]), 'game', 'attracts', 'user')
        >>> g = dgl.hetero_from_relations([g1, g2])
        >>> g.nodes['user'].data['h'] = torch.tensor([[1.], [2.]])
        >>> g.nodes['game'].data['h'] = torch.tensor([[1.]])

        Update all.

        >>> g.multi_update_all(
        >>>     {'follows': (fn.copy_src('h', 'm'), fn.sum('m', 'h')),
        >>>      'attracts': (fn.copy_src('h', 'm'), fn.sum('m', 'h'))},
        >>> "sum")
        >>> g.nodes['user'].data['h']
        tensor([[0.],
                [4.]])
        """
        all_out = defaultdict(list)
        merge_order = defaultdict(list)
        for etype, args in etype_dict.items():
            etid = self.get_etype_id(etype)
            _, dtid = self._graph.metagraph.find_edge(etid)
            args = pad_tuple(args, 3)
            if args is None:
                raise DGLError('Invalid arguments for edge type "{}". Should be '
                               '(msg_func, reduce_func, [apply_node_func])'.format(etype))
            mfunc, rfunc, afunc = args
            all_out[dtid].append(core.message_passing(self[etype], mfunc, rfunc, afunc))
            merge_order[dtid].append(etid)  # use edge type id as merge order hint
        for dtid, frames in all_out.items():
            # merge by cross_reducer
            self._node_frames[dtid].update(
                reduce_dict_data(frames, cross_reducer, merge_order[dtid]))
            # apply
            if apply_node_func is not None:
                self.apply_nodes(apply_node_func, ALL, self.ntypes[dtid])

    #################################################################
    # Message propagation
    #################################################################

    def prop_nodes(self,
                   nodes_generator,
                   message_func,
                   reduce_func,
                   apply_node_func=None,
                   etype=None):
        """Propagate messages using graph traversal by sequentially triggering
        :func:`pull()` on nodes.

        The traversal order is specified by the ``nodes_generator``. It generates
        node frontiers, which is a list or a tensor of nodes. The nodes in the
        same frontier will be triggered together, while nodes in different frontiers
        will be triggered according to the generating order.

        Parameters
        ----------
        nodes_generator : iterable, each element is a list or a tensor of node ids
            The generator of node frontiers. It specifies which nodes perform
            :func:`pull` at each timestep.
        message_func : callable
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        reduce_func : callable
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        apply_node_func : callable, optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`. (Default: None)
        etype : str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph. (Default: None)

        Examples
        --------
        >>> import torch
        >>> import dgl
        >>> import dgl.function as fn

        Instantiate a heterogrph and perform multiple rounds of message passing.

        >>> g = dgl.graph(([0, 1, 2, 3], [2, 3, 4, 4]), 'user', 'follows')
        >>> g.nodes['user'].data['h'] = torch.tensor([[1.], [2.], [3.], [4.], [5.]])
        >>> g['follows'].prop_nodes([[2, 3], [4]], fn.copy_src('h', 'm'),
        >>>                         fn.sum('m', 'h'), etype='follows')
        tensor([[1.],
                [2.],
                [1.],
                [2.],
                [3.]])

        See Also
        --------
        prop_edges
        """
        for node_frontier in nodes_generator:
            self.pull(node_frontier, message_func, reduce_func, apply_node_func, etype=etype)

    def prop_edges(self,
                   edges_generator,
                   message_func,
                   reduce_func,
                   apply_node_func=None,
                   etype=None):
        """Propagate messages using graph traversal by sequentially triggering
        :func:`send_and_recv()` on edges.

        The traversal order is specified by the ``edges_generator``. It generates
        edge frontiers. The edge frontiers should be of *valid edges type*.
        See :func:`send` for more details.

        Edges in the same frontier will be triggered together, and edges in
        different frontiers will be triggered according to the generating order.

        Parameters
        ----------
        edges_generator : generator
            The generator of edge frontiers.
        message_func : callable
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        reduce_func : callable
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        apply_node_func : callable, optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`. (Default: None)
        etype : str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph. (Default: None)

        Examples
        --------
        >>> import torch
        >>> import dgl
        >>> import dgl.function as fn

        Instantiate a heterogrph and perform multiple rounds of message passing.

        >>> g = dgl.graph(([0, 1, 2, 3], [2, 3, 4, 4]), 'user', 'follows')
        >>> g.nodes['user'].data['h'] = torch.tensor([[1.], [2.], [3.], [4.], [5.]])
        >>> g['follows'].prop_edges([[0, 1], [2, 3]], fn.copy_src('h', 'm'),
        >>>                         fn.sum('m', 'h'), etype='follows')
        >>> g.nodes['user'].data['h']
        tensor([[1.],
                [2.],
                [1.],
                [2.],
                [3.]])

        See Also
        --------
        prop_nodes
        """
        for edge_frontier in edges_generator:
            self.send_and_recv(edge_frontier, message_func, reduce_func,
                               apply_node_func, etype=etype)

    #################################################################
    # Misc
    #################################################################

    def filter_nodes(self, predicate, nodes=ALL, ntype=None):
        """Return a tensor of node IDs with the given node type that satisfy
        the given predicate.

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
        ntype : str, optional
            The node type. Can be omitted if there is only one node type
            in the graph. (Default: None)

        Returns
        -------
        tensor
            Node ids indicating the nodes that satisfy the predicate.

        Examples
        --------
        >>> import torch
        >>> import dgl
        >>> import dgl.function as fn
        >>> g = dgl.graph([], 'user', 'follows', num_nodes=4)
        >>> g.nodes['user'].data['h'] = torch.tensor([[0.], [1.], [1.], [0.]])
        >>> g.filter_nodes(lambda nodes: (nodes.data['h'] == 1.).squeeze(1), ntype='user')
        tensor([1, 2])
        """
        with self.local_scope():
            self.apply_nodes(lambda nbatch: {'_mask' : predicate(nbatch)}, nodes, ntype)
            ntype = self.ntypes[0] if ntype is None else ntype
            mask = self.nodes[ntype].data['_mask']
            if is_all(nodes):
                return F.nonzero_1d(mask)
            else:
                v = utils.prepare_tensor(self, nodes, 'nodes')
                return F.boolean_mask(v, F.gather_row(mask, v))

    def filter_edges(self, predicate, edges=ALL, etype=None):
        """Return a tensor of edge IDs with the given edge type that satisfy
        the given predicate.

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
        etype : str, optional
            The edge type. Can be omitted if there is only one edge type
            in the graph. (Default: None)

        Returns
        -------
        tensor
            Edge ids indicating the edges that satisfy the predicate.

        Examples
        --------
        >>> import torch
        >>> import dgl
        >>> import dgl.function as fn
        >>> g = dgl.graph(([0, 0, 1, 2], [0, 1, 2, 3]), 'user', 'follows')
        >>> g.edges['follows'].data['h'] = torch.tensor([[0.], [1.], [1.], [0.]])
        >>> g.filter_edges(lambda edges: (edges.data['h'] == 1.).squeeze(1), etype='follows')
        tensor([1, 2])
        """
        with self.local_scope():
            self.apply_edges(lambda ebatch: {'_mask' : predicate(ebatch)}, edges, etype)
            etype = self.canonical_etypes[0] if etype is None else etype
            mask = self.edges[etype].data['_mask']
            if is_all(edges):
                return F.nonzero_1d(mask)
            else:
                if isinstance(edges, tuple):
                    e = self.edge_ids(edges[0], edges[1], etype=etype)
                else:
                    e = utils.prepare_tensor(self, edges, 'edges')
                return F.boolean_mask(e, F.gather_row(mask, e))

    @property
    def device(self):
        """Get the device context of this graph.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> g = dgl.bipartite(([0, 1, 1, 2], [0, 0, 2, 1]), 'user', 'plays', 'game')
        >>> print(g.device)
        device(type='cpu')
        >>> g = g.to('cuda:0')
        >>> print(g.device)
        device(type='cuda', index=0)

        Returns
        -------
        Device context object
        """
        return F.to_backend_ctx(self._graph.ctx)

    def to(self, device, **kwargs):  # pylint: disable=invalid-name
        """Move ndata, edata and graph structure to the targeted device (cpu/gpu).

        Parameters
        ----------
        device : Framework-specific device context object
            The context to move data to.
        kwargs : Key-word arguments.
            Key-word arguments fed to the framework copy function.

        Returns
        -------
        g : DGLHeteroGraph
          Moved DGLHeteroGraph of the targeted mode.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import torch
        >>> g = dgl.bipartite(([0, 1, 1, 2], [0, 0, 2, 1]), 'user', 'plays', 'game')
        >>> g.nodes['user'].data['h'] = torch.tensor([[0.], [1.], [2.]])
        >>> g.edges['plays'].data['h'] = torch.tensor([[0.], [1.], [2.], [3.]])
        >>> g1 = g.to(torch.device('cuda:0'))
        >>> print(g1.device)
        device(type='cuda', index=0)
        >>> print(g.device)
        device(type='cpu')
        """
        if device is None or self.device == device:
            return self

        ret = copy.copy(self)

        # 1. Copy graph structure
        ret._graph = self._graph.copy_to(utils.to_dgl_context(device))

        # 2. Copy features
        # TODO(minjie): handle initializer
        new_nframes = []
        for nframe in self._node_frames:
            new_nframes.append(nframe.to(device, **kwargs))
        ret._node_frames = new_nframes

        new_eframes = []
        for eframe in self._edge_frames:
            new_eframes.append(eframe.to(device, **kwargs))
        ret._edge_frames = new_eframes

        # 2. Copy misc info
        if self._batch_num_nodes is not None:
            new_bnn = {k : F.copy_to(num, device, **kwargs)
                       for k, num in self._batch_num_nodes.items()}
            ret._batch_num_nodes = new_bnn
        if self._batch_num_edges is not None:
            new_bne = {k : F.copy_to(num, device, **kwargs)
                       for k, num in self._batch_num_edges.items()}
            ret._batch_num_edges = new_bne

        return ret

    def cpu(self):
        """Return a new copy of this graph on CPU.

        Returns
        -------
        DGLHeteroGraph
            Graph on CPU.

        See Also
        --------
        to
        """
        return self.to(F.cpu())

    def clone(self):
        """Return a heterograph object that is a clone of current graph.

        Returns
        -------
        DGLHeteroGraph
            The graph object that is a clone of current graph.
        """
        # XXX(minjie): Do a shallow copy first to clone some internal metagraph information.
        #   Not a beautiful solution though.
        ret = copy.copy(self)

        # Clone the graph structure
        meta_edges = []
        for s_ntype, _, d_ntype in self.canonical_etypes:
            meta_edges.append((self.get_ntype_id(s_ntype), self.get_ntype_id(d_ntype)))

        metagraph = graph_index.from_edge_list(meta_edges, True)
        # rebuild graph idx
        num_nodes_per_type = [self.number_of_nodes(c_ntype) for c_ntype in self.ntypes]
        relation_graphs = [self._graph.get_relation_graph(self.get_etype_id(c_etype))
                           for c_etype in self.canonical_etypes]
        ret._graph = heterograph_index.create_heterograph_from_relations(
            metagraph, relation_graphs, utils.toindex(num_nodes_per_type, "int64"))

        # Clone the frames
        ret._node_frames = [fr.clone() for fr in self._node_frames]
        ret._edge_frames = [fr.clone() for fr in self._edge_frames]

        return ret

    def local_var(self):
        """Return a heterograph object that can be used in a local function scope.

        The returned graph object shares the feature data and graph structure of this graph.
        However, any out-place mutation to the feature data will not reflect to this graph,
        thus making it easier to use in a function scope.

        If set, the local graph object will use same initializers for node features and
        edge features.

        Returns
        -------
        DGLHeteroGraph
            The graph object that can be used as a local variable.

        Notes
        -----
        Internally, the returned graph shares the same feature tensors, but construct a new
        dictionary structure (aka. Frame) so adding/removing feature tensors from the returned
        graph will not reflect to the original graph. However, inplace operations do change
        the shared tensor values, so will be reflected to the original graph. This function
        also has little overhead when the number of feature tensors in this graph is small.

        Examples
        --------
        The following example uses PyTorch backend.

        Avoid accidentally overriding existing feature data. This is quite common when
        implementing a NN module:

        >>> def foo(g):
        >>>     g = g.local_var()
        >>>     g.edata['h'] = torch.ones((g.number_of_edges(), 3))
        >>>     return g.edata['h']
        >>>
        >>> g = dgl.bipartite(([0, 1, 1], [0, 0, 2]), 'user', 'plays', 'game')
        >>> g.edata['h'] = torch.zeros((g.number_of_edges(), 3))
        >>> newh = foo(g)        # get tensor of all ones
        >>> print(g.edata['h'])  # still get tensor of all zeros

        Automatically garbage collect locally-defined tensors without the need to manually
        ``pop`` the tensors.

        >>> def foo(g):
        >>>     g = g.local_var()
        >>>     # This 'h' feature will stay local and be GCed when the function exits
        >>>     g.edata['h'] = torch.ones((g.number_of_edges(), 3))
        >>>     return g.edata['h']
        >>>
        >>> g = dgl.bipartite(([0, 1, 1], [0, 0, 2]), 'user', 'plays', 'game')
        >>> h = foo(g)
        >>> print('h' in g.edata)
        False

        See Also
        --------
        local_var
        """
        ret = copy.copy(self)
        ret._node_frames = [fr.clone() for fr in self._node_frames]
        ret._edge_frames = [fr.clone() for fr in self._edge_frames]
        return ret

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
        >>>         g.edata['h'] = torch.ones((g.number_of_edges(), 3))
        >>>         return g.edata['h']
        >>>
        >>> g = dgl.bipartite(([0, 1, 1], [0, 0, 2]), 'user', 'plays', 'game')
        >>> g.edata['h'] = torch.zeros((g.number_of_edges(), 3))
        >>> newh = foo(g)        # get tensor of all ones
        >>> print(g.edata['h'])  # still get tensor of all zeros

        Automatically garbage collect locally-defined tensors without the need to manually
        ``pop`` the tensors.

        >>> def foo(g):
        >>>     with g.local_scope():
        >>>         # This 'h' feature will stay local and be GCed when the function exits
        >>>         g.edata['h'] = torch.ones((g.number_of_edges(), 3))
        >>>         return g.edata['h']
        >>>
        >>> g = dgl.bipartite(([0, 1, 1], [0, 0, 2]), 'user', 'plays', 'game')
        >>> h = foo(g)
        >>> print('h' in g.edata)
        False

        See Also
        --------
        local_var
        """
        old_nframes = self._node_frames
        old_eframes = self._edge_frames
        self._node_frames = [fr.clone() for fr in self._node_frames]
        self._edge_frames = [fr.clone() for fr in self._edge_frames]
        yield
        self._node_frames = old_nframes
        self._edge_frames = old_eframes

    def is_homogeneous(self):
        """Return if the graph is homogeneous."""
        return len(self.ntypes) == 1 and len(self.etypes) == 1

    def formats(self, formats=None):
        r"""Get a cloned graph with the specified sparse format(s) or query
        for the usage status of sparse formats

        The API copies both the graph structure and the features.

        If the input graph has multiple edge types, they will have the same
        sparse format.

        Parameters
        ----------
        formats : str or list of str or None

            * If formats is None, return the usage status of sparse formats
            * Otherwise, it can be ``'coo'``/``'csr'``/``'csc'`` or a sublist of
            them, specifying the sparse formats to use.

        Returns
        -------
        dict or DGLGraph

            * If formats is None, the result will be a dict recording the usage
              status of sparse formats.
            * Otherwise, a DGLGraph will be returned, which is a clone of the
              original graph with the specified sparse format(s) ``formats``.

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        **Homographs or Heterographs with A Single Edge Type**

        >>> g = dgl.graph([(0, 2), (0, 3), (1, 2)])
        >>> g.ndata['h'] = torch.ones(4, 1)
        >>> # Check status of format usage
        >>> g.formats()
        {'created': ['coo'], 'not created': ['csr', 'csc']}
        >>> # Get a clone of the graph with 'csr' format
        >>> csr_g = g.formats('csr')
        >>> # Only allowed formats will be displayed in the status query
        >>> csr_g.formats()
        {'created': ['csr'], 'not created': []}
        >>> # Features are copied as well
        >>> csr_g.ndata['h']
        tensor([[1.],
                [1.],
                [1.],
                [1.]])

        **Heterographs with Multiple Edge Types**

        >>> g = dgl.heterograph({
        >>>     ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
        >>>                                 torch.tensor([0, 0, 1, 1])),
        >>>     ('developer', 'develops', 'game'): (torch.tensor([0, 1]),
        >>>                                         torch.tensor([0, 1]))
        >>>     })
        >>> g.formats()
        {'created': ['coo'], 'not created': ['csr', 'csc']}
        >>> # Get a clone of the graph with 'csr' format
        >>> csr_g = g.formats('csr')
        >>> # Only allowed formats will be displayed in the status query
        >>> csr_g.formats()
        {'created': ['csr'], 'not created': []}
        """
        if formats is None:
            # Return the format information
            return self._graph.formats()
        else:
            # Convert the graph to use another format
            ret = copy.copy(self)
            ret._graph = self._graph.formats(formats)
            return ret

    def create_format_(self):
        r"""Create all sparse matrices allowed for the graph.

        By default, we create sparse matrices for a graph only when necessary.
        In some cases we may want to create them immediately (e.g. in a
        multi-process data loader), which can be achieved via this API.

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        **Homographs or Heterographs with A Single Edge Type**

        >>> g = dgl.graph([(0, 2), (0, 3), (1, 2)])
        >>> g.format()
        {'created': ['coo'], 'not created': ['csr', 'csc']}
        >>> g.create_format_()
        >>> g.format()
        {'created': ['coo', 'csr', 'csc'], 'not created': []}

        **Heterographs with Multiple Edge Types**

        >>> g = dgl.heterograph({
        >>>     ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
        >>>                                 torch.tensor([0, 0, 1, 1])),
        >>>     ('developer', 'develops', 'game'): (torch.tensor([0, 1]),
        >>>                                         torch.tensor([0, 1]))
        >>>     })
        >>> g.format()
        {'created': ['coo'], 'not created': ['csr', 'csc']}
        >>> g.create_format_()
        >>> g.format()
        {'created': ['coo', 'csr', 'csc'], 'not created': []}
        """
        return self._graph.create_format_()

    def astype(self, idtype):
        """Cast this graph to use another ID type.

        Features are copied (shallow copy) to the new graph.

        Parameters
        ----------
        idtype : Data type object.
            New ID type. Can only be int32 or int64.

        Returns
        -------
        DGLHeteroGraph
            Graph in the new ID type.
        """
        if idtype is None:
            return self
        if not idtype in (F.int32, F.int64):
            raise DGLError("ID type must be int32 or int64, but got {}.".format(idtype))
        if self.idtype == idtype:
            return self
        bits = 32 if idtype == F.int32 else 64
        ret = copy.copy(self)
        ret._graph = self._graph.asbits(bits)
        return ret

    # TODO: Formats should not be specified, just saving all the materialized formats
    def shared_memory(self, name, formats=('coo', 'csr', 'csc')):
        """Return a copy of this graph in shared memory, without node data or edge data.

        It moves the graph index to shared memory and returns a DGLHeterograph object which
        has the same graph structure, node types and edge types but does not contain node data
        or edge data.

        Parameters
        ----------
        name : str
            The name of the shared memory.
        formats : str or a list of str (optional)
            Desired formats to be materialized.

        Returns
        -------
        HeteroGraph
            The graph in shared memory
        """
        assert len(name) > 0, "The name of shared memory cannot be empty"
        assert len(formats) > 0
        if isinstance(formats, str):
            formats = [formats]
        for fmt in formats:
            assert fmt in ("coo", "csr", "csc"), '{} is not coo, csr or csc'.format(fmt)
        gidx = self._graph.shared_memory(name, self.ntypes, self.etypes, formats)
        return DGLHeteroGraph(gidx, self.ntypes, self.etypes)


    def long(self):
        """Cast this graph to use int64 IDs.

        Features are copied (shallow copy) to the new graph.

        Returns
        -------
        DGLHeteroGraph
            The graph object

        Examples
        --------

        >>> g = dgl.bipartite(([0, 1, 1], [0, 0, 2]), 'user', 'plays', 'game',
        >>>                   idtype=torch.int32)
        >>> g_long = g.long() # Convert g to int64 indexed, not changing the original `g`

        See Also
        --------
        int
        idtype
        astype
        """
        return self.astype(F.int64)

    def int(self):
        """Return a heterograph object use int32 as index dtype,
        with the ndata and edata as the original object

        Returns
        -------
        DGLHeteroGraph
            The graph object

        Examples
        --------

        >>> g = dgl.bipartite(([0, 1, 1], [0, 0, 2]), 'user', 'plays', 'game',
        >>>                   idtype=torch.int64)
        >>> g_int = g.int() # Convert g to int32 indexed, not changing the original `g`

        See Also
        --------
        long
        idtype
        astype
        """
        return self.astype(F.int32)

    #################################################################
    # DEPRECATED: from the old DGLGraph
    #################################################################

    def from_networkx(self, nx_graph, node_attrs=None, edge_attrs=None):
        """DEPRECATED: please use

            ``dgl.from_networkx(nx_graph, node_attrs, edge_attrs)``

        which will return a new graph created from the networkx graph.
        """
        raise DGLError('DGLGraph.from_networkx is deprecated. Please call the following\n\n'
                       '\t dgl.from_networkx(nx_graph, node_attrs, edge_attrs)\n\n'
                       ', which creates a new DGLGraph from the networkx graph.')

    def from_scipy_sparse_matrix(self, spmat, multigraph=None):
        """DEPRECATED: please use

            ``dgl.from_scipy(spmat)``

        which will return a new graph created from the scipy matrix.
        """
        raise DGLError('DGLGraph.from_scipy_sparse_matrix is deprecated. '
                       'Please call the following\n\n'
                       '\t dgl.from_scipy(spmat)\n\n'
                       ', which creates a new DGLGraph from the scipy matrix.')

    def register_apply_node_func(self, func):
        """Deprecated: please directly call :func:`apply_nodes` with ``func``
        as argument.
        """
        raise DGLError('DGLGraph.register_apply_node_func is deprecated.'
                       ' Please directly call apply_nodes with func as the argument.')

    def register_apply_edge_func(self, func):
        """Deprecated: please directly call :func:`apply_edges` with ``func``
        as argument.
        """
        raise DGLError('DGLGraph.register_apply_edge_func is deprecated.'
                       ' Please directly call apply_edges with func as the argument.')

    def register_message_func(self, func):
        """Deprecated: please directly call :func:`update_all` with ``func``
        as argument.
        """
        raise DGLError('DGLGraph.register_message_func is deprecated.'
                       ' Please directly call update_all with func as the argument.')

    def register_reduce_func(self, func):
        """Deprecated: please directly call :func:`update_all` with ``func``
        as argument.
        """
        raise DGLError('DGLGraph.register_reduce_func is deprecated.'
                       ' Please directly call update_all with func as the argument.')

    def group_apply_edges(self, group_by, func, edges=ALL, etype=None, inplace=False):
        """**DEPRECATED**: The API is removed in 0.5."""
        raise DGLError('DGLGraph.group_apply_edges is removed in 0.5.')

    def send(self, edges, message_func, etype=None):
        """Send messages along the given edges with the same edge type.

        DEPRECATE: please use send_and_recv, update_all.
        """
        raise DGLError('DGLGraph.send is deprecated. As a replacement, use DGLGraph.apply_edges\n'
                       ' API to compute messages as edge data. Then use DGLGraph.send_and_recv\n'
                       ' and set the message function as dgl.function.copy_e to conduct message\n'
                       ' aggregation.')

    def recv(self, v, reduce_func, apply_node_func=None, etype=None, inplace=False):
        r"""Receive and reduce incoming messages and update the features of node(s) :math:`v`.

        DEPRECATE: please use send_and_recv, update_all.
        """
        raise DGLError('DGLGraph.recv is deprecated. As a replacement, use DGLGraph.apply_edges\n'
                       ' API to compute messages as edge data. Then use DGLGraph.send_and_recv\n'
                       ' and set the message function as dgl.function.copy_e to conduct message\n'
                       ' aggregation.')

    def multi_recv(self, v, reducer_dict, cross_reducer, apply_node_func=None, inplace=False):
        r"""Receive messages from multiple edge types and perform aggregation.

        DEPRECATE: please use multi_send_and_recv, multi_update_all.
        """
        raise DGLError('DGLGraph.multi_recv is deprecated. As a replacement,\n'
                       ' use DGLGraph.apply_edges API to compute messages as edge data.\n'
                       ' Then use DGLGraph.multi_send_and_recv and set the message function\n'
                       ' as dgl.function.copy_e to conduct message aggregation.')

    def multi_send_and_recv(self, etype_dict, cross_reducer, apply_node_func=None, inplace=False):
        r"""**DEPRECATED**: The API is removed in v0.5."""
        raise DGLError('DGLGraph.multi_pull is removed in v0.5. As a replacement,\n'
                       ' use DGLGraph.edge_subgraph to extract the subgraph first \n'
                       ' and then call DGLGraph.multi_update_all.')

    def multi_pull(self, v, etype_dict, cross_reducer, apply_node_func=None, inplace=False):
        r"""**DEPRECATED**: The API is removed in v0.5."""
        raise DGLError('DGLGraph.multi_pull is removed in v0.5. As a replacement,\n'
                       ' use DGLGraph.edge_subgraph to extract the subgraph first \n'
                       ' and then call DGLGraph.multi_update_all.')

    def readonly(self, readonly_state=True):
        """Deprecated: DGLGraph will always be mutable."""
        dgl_warning('DGLGraph.readonly is deprecated in v0.5.\n'
                    'DGLGraph now always supports mutable operations like add_nodes'
                    ' and add_edges.')

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
    if (len(etypes) == 1 and len(ntypes) == 1):
        return [(ntypes[0], etypes[0], ntypes[0])]
    src, dst, eid = metagraph.edges(order="eid")
    rst = [(ntypes[sid], etypes[eid], ntypes[did]) for sid, did, eid in zip(src, dst, eid)]
    return rst

def is_unibipartite(graph):
    """Internal function that returns whether the given graph is a uni-directional
    bipartite graph.

    Parameters
    ----------
    graph : GraphIndex
        Input graph

    Returns
    -------
    bool
        True if the graph is a uni-bipartite.
    """
    src, dst, _ = graph.edges()
    return set(src.tonumpy()).isdisjoint(set(dst.tonumpy()))

def find_src_dst_ntypes(ntypes, metagraph):
    """Internal function to split ntypes into SRC and DST categories.

    If the metagraph is not a uni-bipartite graph (so that the SRC and DST categories
    are not well-defined), return None.

    For node types that are isolated (i.e, no relation is associated with it), they
    are assigned to the SRC category.

    Parameters
    ----------
    ntypes : list of str
        Node type list
    metagraph : GraphIndex
        Meta graph.

    Returns
    -------
    (dict[int, str], dict[int, str]) or None
        Node types belonging to SRC and DST categories. Types are stored in
        a dictionary from type name to type id. Return None if the graph is
        not uni-bipartite.
    """
    ret = _CAPI_DGLFindSrcDstNtypes(metagraph)
    if ret is None:
        return None
    else:
        src, dst = ret
        srctypes = {ntypes[tid] : tid for tid in src}
        dsttypes = {ntypes[tid] : tid for tid in dst}
        return srctypes, dsttypes

def pad_tuple(tup, length, pad_val=None):
    """Pad the given tuple to the given length.

    If the input is not a tuple, convert it to a tuple of length one.
    Return None if pad fails.
    """
    if not isinstance(tup, tuple):
        tup = (tup, )
    if len(tup) > length:
        return None
    elif len(tup) == length:
        return tup
    else:
        return tup + (pad_val,) * (length - len(tup))

def reduce_dict_data(frames, reducer, order=None):
    """Merge tensor dictionaries into one. Resolve conflict fields using reducer.

    Parameters
    ----------
    frames : list[dict[str, Tensor]]
        Input tensor dictionaries
    reducer : str
        One of "sum", "max", "min", "mean", "stack"
    order : list[Int], optional
        Merge order hint. Useful for "stack" reducer.
        If provided, each integer indicates the relative order
        of the ``frames`` list. Frames are sorted according to this list
        in ascending order. Tie is not handled so make sure the order values
        are distinct.

    Returns
    -------
    dict[str, Tensor]
        Merged frame
    """
    if len(frames) == 1 and reducer != 'stack':
        # Directly return the only one input. Stack reducer requires
        # modifying tensor shape.
        return frames[0]
    if reducer == 'stack':
        # Stack order does not matter. However, it must be consistent!
        if order:
            assert len(order) == len(frames)
            sorted_with_key = sorted(zip(frames, order), key=lambda x: x[1])
            frames = list(zip(*sorted_with_key))[0]
        def merger(flist):
            return F.stack(flist, 1)
    else:
        redfn = getattr(F, reducer, None)
        if redfn is None:
            raise DGLError('Invalid cross type reducer. Must be one of '
                           '"sum", "max", "min", "mean" or "stack".')
        def merger(flist):
            return redfn(F.stack(flist, 0), 0) if len(flist) > 1 else flist[0]
    keys = set()
    for frm in frames:
        keys.update(frm.keys())
    ret = {}
    for k in keys:
        flist = []
        for frm in frames:
            if k in frm:
                flist.append(frm[k])
        ret[k] = merger(flist)
    return ret

def combine_frames(frames, ids):
    """Merge the frames into one frame, taking the common columns.

    Return None if there is no common columns.

    Parameters
    ----------
    frames : List[Frame]
        List of frames
    ids : List[int]
        List of frame IDs

    Returns
    -------
    Frame
        The resulting frame
    """
    # find common columns and check if their schemes match
    schemes = {key: scheme for key, scheme in frames[ids[0]].schemes.items()}
    for frame_id in ids:
        frame = frames[frame_id]
        for key, scheme in list(schemes.items()):
            if key in frame.schemes:
                if frame.schemes[key] != scheme:
                    raise DGLError('Cannot concatenate column %s with shape %s and shape %s' %
                                   (key, frame.schemes[key], scheme))
            else:
                del schemes[key]

    if len(schemes) == 0:
        return None

    # concatenate the columns
    to_cat = lambda key: [frames[i][key] for i in ids if frames[i].num_rows > 0]
    cols = {key: F.cat(to_cat(key), dim=0) for key in schemes}
    return Frame(cols)

def combine_names(names, ids=None):
    """Combine the selected names into one new name.

    Parameters
    ----------
    names : list of str
        String names
    ids : numpy.ndarray, optional
        Selected index

    Returns
    -------
    str
    """
    if ids is None:
        return '+'.join(sorted(names))
    else:
        selected = sorted([names[i] for i in ids])
        return '+'.join(selected)

class DGLBlock(DGLHeteroGraph):
    """Subclass that signifies the graph is a block created from
    :func:`dgl.to_block`.
    """
    # (BarclayII) I'm making a subclass because I don't want to make another version of
    # serialization that contains the is_block flag.
    is_block = True

    def __repr__(self):
        if len(self.srctypes) == 1 and len(self.dsttypes) == 1 and len(self.etypes) == 1:
            ret = 'Block(num_src_nodes={srcnode}, num_dst_nodes={dstnode}, num_edges={edge})'
            return ret.format(
                srcnode=self.number_of_src_nodes(),
                dstnode=self.number_of_dst_nodes(),
                edge=self.number_of_edges())
        else:
            ret = ('Block(num_src_nodes={srcnode},\n'
                   '      num_dst_nodes={dstnode},\n'
                   '      num_edges={edge},\n'
                   '      metagraph={meta})')
            nsrcnode_dict = {ntype : self.number_of_src_nodes(ntype)
                             for ntype in self.srctypes}
            ndstnode_dict = {ntype : self.number_of_dst_nodes(ntype)
                             for ntype in self.dsttypes}
            nedge_dict = {etype : self.number_of_edges(etype)
                          for etype in self.canonical_etypes}
            meta = str(self.metagraph().edges(keys=True))
            return ret.format(
                srcnode=nsrcnode_dict, dstnode=ndstnode_dict, edge=nedge_dict, meta=meta)


def _create_compute_graph(graph, u, v, eid, recv_nodes=None):
    """Create a computation graph from the given edges.

    The compute graph is a uni-directional bipartite graph with only
    one edge type. Similar to subgraph extraction, it stores the original node IDs
    in the srcdata[NID] and dstdata[NID] and extracts features accordingly.
    Edges are not relabeled.

    This function is typically used during message passing to generate
    a graph that contains only the active set of edges.

    Parameters
    ----------
    graph : DGLGraph
        The input graph.
    u : Tensor
        Src nodes.
    v : Tensor
        Dst nodes.
    eid : Tensor
        Edge IDs.
    recv_nodes : Tensor
        Nodes that receive messages. If None, it is equal to unique(v).
        Otherwise, it must be a superset of v and can contain nodes
        that have no incoming edges.

    Returns
    -------
    DGLGraph
        A computation graph.
    """
    if len(u) == 0:
        # The computation graph has no edge and will not trigger message
        # passing. However, because of the apply node phase, we still construct
        # an empty graph to continue.
        unique_src = new_u = new_v = u
        assert recv_nodes is not None
        unique_dst, _ = utils.relabel(recv_nodes)
    else:
        # relabel u and v to starting from 0
        unique_src, src_map = utils.relabel(u)
        if recv_nodes is None:
            unique_dst, dst_map = utils.relabel(v)
        else:
            unique_dst, dst_map = utils.relabel(recv_nodes)
        new_u = F.gather_row(src_map, u)
        new_v = F.gather_row(dst_map, v)

    srctype, etype, dsttype = graph.canonical_etypes[0]
    # create graph
    hgidx = heterograph_index.create_unitgraph_from_coo(
        2, len(unique_src), len(unique_dst), new_u, new_v, ['coo', 'csr', 'csc'])
    # create frame
    srcframe = graph._node_frames[graph.get_ntype_id(srctype)].subframe(unique_src)
    srcframe[NID] = unique_src
    dstframe = graph._node_frames[graph.get_ntype_id(dsttype)].subframe(unique_dst)
    dstframe[NID] = unique_dst
    eframe = graph._edge_frames[0].subframe(eid)
    eframe[EID] = eid

    return DGLHeteroGraph(hgidx, ([srctype], [dsttype]), [etype],
                          node_frames=[srcframe, dstframe],
                          edge_frames=[eframe])

_init_api("dgl.heterograph")
