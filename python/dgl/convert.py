"""Module for converting graph from/to other object."""
# pylint: disable=dangerous-default-value
from collections import defaultdict
import numpy as np
import networkx as nx

from . import backend as F
from . import heterograph_index
from .heterograph import DGLHeteroGraph, combine_frames
from . import graph_index
from . import utils
from .base import NTYPE, ETYPE, NID, EID, DGLError, dgl_warning

__all__ = [
    'graph',
    'bipartite',
    'hetero_from_relations',
    'hetero_from_shared_memory',
    'heterograph',
    'to_hetero',
    'to_homo',
    'from_scipy',
    'from_networkx',
    'to_networkx',
]

def graph(data,
          ntype='_N', etype='_E',
          num_nodes=None,
          validate=True,
          formats=['coo', 'csr', 'csc'],
          idtype=None,
          device=None,
          card=None,
          **deprecated_kwargs):
    """Create a graph with one type of nodes and edges.

    In the sparse matrix perspective, :func:`dgl.graph` creates a graph
    whose adjacency matrix must be square while :func:`dgl.bipartite`
    creates a graph that does not necessarily have square adjacency matrix.

    Parameters
    ----------
    data : graph data
        Data to initialize graph structure. Supported data formats are

        (1) list of edge pairs (e.g. [(0, 2), (3, 1), ...])
        (2) pair of vertex IDs representing end nodes (e.g. ([0, 3, ...],  [2, 1, ...]))
        (3) scipy sparse matrix
        (4) networkx graph

    ntype : str, optional
        Node type name. (Default: _N)
    etype : str, optional
        Edge type name. (Default: _E)
    num_nodes : int, optional
        Number of nodes in the graph. If None, infer from input data, i.e.
        the largest node ID plus 1. (Default: None)
    validate : bool, optional
        If True, check if node ids are within cardinality, the check process may take
        some time. (Default: True)
        If False and card is not None, user would receive a warning.
    formats : str or list of str
        It can be ``'coo'``/``'csr'``/``'csc'`` or a sublist of them,
        Force the storage formats.  Default: ``['coo', 'csr', 'csc']``.
    idtype : int32, int64, optional
        Integer ID type. Valid options are int32 or int64. If None, try infer from
        the given data.
    device : Device context, optional
        Device on which the graph is created. Default: infer from data.
    card : int, optional
        Deprecated (see :attr:`num_nodes`). Cardinality (number of nodes in the graph).
        If None, infer from input data, i.e. the largest node ID plus 1. (Default: None)

    Returns
    -------
    DGLHeteroGraph

    Examples
    --------
    Create from pairs of edges with form (src, dst)

    >>> g = dgl.graph([(0, 2), (0, 3), (1, 2)])

    Create from source and destination vertex ID lists

    >>> u = [0, 0, 1]
    >>> v = [2, 3, 2]
    >>> g = dgl.graph((u, v))

    The IDs can also be stored in framework-specific tensors

    >>> import torch
    >>> u = torch.tensor([0, 0, 1])
    >>> v = torch.tensor([2, 3, 2])
    >>> g = dgl.graph((u, v))

    Create from scipy sparse matrix

    >>> from scipy.sparse import coo_matrix
    >>> spmat = coo_matrix(([1,1,1], ([0, 0, 1], [2, 3, 2])), shape=(4, 4))
    >>> g = dgl.graph(spmat)

    Create from networkx graph

    >>> import networkx as nx
    >>> nxg = nx.path_graph(3)
    >>> g = dgl.graph(nxg)

    Specify node and edge type names

    >>> g = dgl.graph(..., 'user', 'follows')
    >>> g.ntypes
    ['user']
    >>> g.etypes
    ['follows']
    >>> g.canonical_etypes
    [('user', 'follows', 'user')]

    Check if node ids are within num_nodes specified

    >>> g = dgl.graph(([0, 1, 2], [1, 2, 0]), num_nodes=2, validate=True)
    ...
    dgl._ffi.base.DGLError: Invalid node id 2 (should be less than cardinality 2).
    >>> g = dgl.graph(([0, 1, 2], [1, 2, 0]), num_nodes=3, validate=True)
    Graph(num_nodes=3, num_edges=3,
          ndata_schemes={}
          edata_schemes={})
    """
    if len(deprecated_kwargs) != 0:
        raise DGLError("Key word arguments {} have been removed from dgl.graph()."
                       " They are moved to dgl.from_scipy() and dgl.from_networkx()."
                       " Please refer to their API documents for more details.".format(
                           deprecated_kwargs.keys()))

    if isinstance(data, DGLHeteroGraph):
        return data.astype(idtype).to(device)

    if card is not None:
        dgl_warning("Argument 'card' will be deprecated. "
                    "Please use num_nodes={} instead.".format(card))
        num_nodes = card

    u, v, urange, vrange = utils.graphdata2tensors(data, idtype)
    if num_nodes is not None:  # override the number of nodes
        urange, vrange = num_nodes, num_nodes

    g = create_from_edges(u, v, ntype, etype, ntype, urange, vrange,
                          validate, formats=formats)

    return g.to(device)

def bipartite(data,
              utype='_U', etype='_E', vtype='_V',
              num_nodes=None,
              validate=True,
              formats=['coo', 'csr', 'csc'],
              idtype=None,
              device=None,
              card=None,
              **deprecated_kwargs):
    """Create a bipartite graph.

    The result graph is directed and edges must be from ``utype`` nodes
    to ``vtype`` nodes. Nodes of each type have their own ID counts.

    In the sparse matrix perspective, :func:`dgl.graph` creates a graph
    whose adjacency matrix must be square while :func:`dgl.bipartite`
    creates a graph that does not necessarily have square adjacency matrix.

    Parameters
    ----------
    data : graph data
        Data to initialize graph structure. Supported data formats are

        (1) list of edge pairs (e.g. [(0, 2), (3, 1), ...])
        (2) pair of vertex IDs representing end nodes (e.g. ([0, 3, ...],  [2, 1, ...]))
        (3) scipy sparse matrix
        (4) networkx graph

    utype : str, optional
        Source node type name. (Default: _U)
    etype : str, optional
        Edge type name. (Default: _E)
    vtype : str, optional
        Destination node type name. (Default: _V)
    num_nodes : 2-tuple of int, optional
        Number of nodes in the source and destination group. If None, infer from input data,
        i.e. the largest node ID plus 1 for each type. (Default: None)
    validate : bool, optional
        If True, check if node ids are within cardinality, the check process may take
        some time. (Default: True)
        If False and card is not None, user would receive a warning.
    formats : str or list of str
        It can be ``'coo'``/``'csr'``/``'csc'`` or a sublist of them,
        Force the storage formats.  Default: ``['coo', 'csr', 'csc']``.
    idtype : int32, int64, optional
        Integer ID type. Valid options are int32 or int64. If None, try infer from
        the given data.
    device : Device context, optional
        Device on which the graph is created. Default: infer from data.
    card : 2-tuple of int, optional
        Deprecated (see :attr:`num_nodes`). Cardinality (number of nodes in the source and
        destination group). If None, infer from input data, i.e. the largest node ID plus 1
        for each type. (Default: None)

    Returns
    -------
    DGLHeteroGraph

    Examples
    --------
    Create from pairs of edges

    >>> g = dgl.bipartite([(0, 2), (0, 3), (1, 2)], 'user', 'plays', 'game')
    >>> g.ntypes
    ['user', 'game']
    >>> g.etypes
    ['plays']
    >>> g.canonical_etypes
    [('user', 'plays', 'game')]
    >>> g.number_of_nodes('user')
    2
    >>> g.number_of_nodes('game')
    4
    >>> g.number_of_edges('plays')  # 'plays' could be omitted here
    3

    Create from source and destination vertex ID lists

    >>> u = [0, 0, 1]
    >>> v = [2, 3, 2]
    >>> g = dgl.bipartite((u, v))

    The IDs can also be stored in framework-specific tensors

    >>> import torch
    >>> u = torch.tensor([0, 0, 1])
    >>> v = torch.tensor([2, 3, 2])
    >>> g = dgl.bipartite((u, v))

    Create from scipy sparse matrix. Since scipy sparse matrix has explicit
    shape, the cardinality of the result graph is derived from that.

    >>> from scipy.sparse import coo_matrix
    >>> spmat = coo_matrix(([1,1,1], ([0, 0, 1], [2, 3, 2])), shape=(4, 4))
    >>> g = dgl.bipartite(spmat, 'user', 'plays', 'game')
    >>> g.number_of_nodes('user')
    4
    >>> g.number_of_nodes('game')
    4

    Create from networkx graph. The given graph must follow the bipartite
    graph convention in networkx. Each node has a ``bipartite`` attribute
    with values 0 or 1. The result graph has two types of nodes and only
    edges from ``bipartite=0`` to ``bipartite=1`` will be included.

    >>> import networkx as nx
    >>> nxg = nx.complete_bipartite_graph(3, 4)
    >>> g = dgl.bipartite(nxg, 'user', 'plays', 'game')
    >>> g.number_of_nodes('user')
    3
    >>> g.number_of_nodes('game')
    4
    >>> g.edges()
    (tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]), tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]))

    Check if node ids are within num_nodes specified

    >>> g = dgl.bipartite(([0, 1, 2], [1, 2, 3]), num_nodes=(2, 4), validate=True)
    ...
    dgl._ffi.base.DGLError: Invalid node id 2 (should be less than cardinality 2).
    >>> g = dgl.bipartite(([0, 1, 2], [1, 2, 3]), num_nodes=(3, 4), validate=True)
    >>> g
    Graph(num_nodes={'_U': 3, '_V': 4},
          num_edges={('_U', '_E', '_V'): 3},
          metagraph=[('_U', '_V')])
    """
    if len(deprecated_kwargs) != 0:
        raise DGLError("Key word arguments {} have been removed from dgl.graph()."
                       " They are moved to dgl.from_scipy() and dgl.from_networkx()."
                       " Please refer to their API documents for more details.".format(
                           deprecated_kwargs.keys()))

    if utype == vtype:
        raise DGLError('utype should not be equal to vtype. Use ``dgl.graph`` instead.')
    if card is not None:
        dgl_warning("Argument 'card' will be deprecated. "
                    "Please use num_nodes={} instead.".format(card))
        num_nodes = card

    u, v, urange, vrange = utils.graphdata2tensors(data, idtype, bipartite=True)
    if num_nodes is not None:  # override the number of nodes
        urange, vrange = num_nodes

    g = create_from_edges(
        u, v, utype, etype, vtype, urange, vrange, validate,
        formats=formats)

    return g.to(device)

def hetero_from_relations(rel_graphs, num_nodes_per_type=None):
    """Create a heterograph from graphs representing connections of each relation.

    The input is a list of heterographs where the ``i``th graph contains edges of type
    :math:`(s_i, e_i, d_i)`.

    If two graphs share a same node type, the number of nodes for the corresponding type
    should be the same. See **Examples** for details.

    Parameters
    ----------
    rel_graphs : list of DGLHeteroGraph
        Each element corresponds to a heterograph for one (src, edge, dst) relation.
    num_nodes_per_type : dict[str, Tensor], optional
        Number of nodes per node type.  If not given, DGL will infer the number of nodes
        from the given relation graphs.

    Returns
    -------
    DGLHeteroGraph
        A heterograph consisting of all relations.

    Examples
    --------

    >>> import dgl
    >>> follows_g = dgl.graph([(0, 1), (1, 2)], 'user', 'follows')
    >>> plays_g = dgl.bipartite([(0, 0), (3, 1)], 'user', 'plays', 'game')
    >>> devs_g = dgl.bipartite([(0, 0), (1, 1)], 'developer', 'develops', 'game')
    >>> g = dgl.hetero_from_relations([follows_g, plays_g, devs_g])

    will raise an error as we have 3 nodes of type 'user' in follows_g and 4 nodes of type
    'user' in plays_g.

    We have two possible methods to avoid the construction.

    **Method 1**: Manually specify the number of nodes for all types when constructing
    the relation graphs.

    >>> # A graph with 4 nodes of type 'user'
    >>> follows_g = dgl.graph([(0, 1), (1, 2)], 'user', 'follows', num_nodes=4)
    >>> # A bipartite graph with 4 nodes of src type ('user') and 2 nodes of dst type ('game')
    >>> plays_g = dgl.bipartite([(0, 0), (3, 1)], 'user', 'plays', 'game', num_nodes=(4, 2))
    >>> devs_g = dgl.bipartite([(0, 0), (1, 1)], 'developer', 'develops', 'game')
    >>> g = dgl.hetero_from_relations([follows_g, plays_g, devs_g])
    >>> print(g)
    Graph(num_nodes={'user': 4, 'game': 2, 'developer': 2},
          num_edges={('user', 'follows', 'user'): 2, ('user', 'plays', 'game'): 2,
                     ('developer', 'develops', 'game'): 2},
          metagraph=[('user', 'user'), ('user', 'game'), ('developer', 'game')])

    ``devs_g`` does not have nodes of type ``'user'`` so no error will be raised.

    **Method 2**: Construct a heterograph at once without intermediate relation graphs,
    in which case we will infer the number of nodes for each type.

    >>> g = dgl.heterograph({
    >>>     ('user', 'follows', 'user'): [(0, 1), (1, 2)],
    >>>     ('user', 'plays', 'game'): [(0, 0), (3, 1)],
    >>>     ('developer', 'develops', 'game'): [(0, 0), (1, 1)]
    >>> })
    >>> print(g)
    Graph(num_nodes={'user': 4, 'game': 2, 'developer': 2},
          num_edges={('user', 'follows', 'user'): 2,
                     ('user', 'plays', 'game'): 2,
                     ('developer', 'develops', 'game'): 2},
          metagraph=[('user', 'user'), ('user', 'game'), ('developer', 'game')])
    """
    utils.check_all_same_idtype(rel_graphs, 'rel_graphs')
    utils.check_all_same_device(rel_graphs, 'rel_graphs')
    # TODO(minjie): this API can be generalized as a union operation of the input graphs
    # TODO(minjie): handle node/edge data
    # infer meta graph
    meta_edges_src, meta_edges_dst = [], []
    ntypes = []
    etypes = []
    # TODO(BarclayII): I'm keeping the node type names sorted because even if
    # the metagraph is the same, the same node type name in different graphs may
    # map to different node type IDs.
    # In the future, we need to lower the type names into C++.
    if num_nodes_per_type is None:
        ntype_set = set()
        for rgrh in rel_graphs:
            assert len(rgrh.etypes) == 1
            stype, etype, dtype = rgrh.canonical_etypes[0]
            ntype_set.add(stype)
            ntype_set.add(dtype)
        ntypes = list(sorted(ntype_set))
    else:
        ntypes = list(sorted(num_nodes_per_type.keys()))
        num_nodes_per_type = utils.toindex([num_nodes_per_type[ntype] for ntype in ntypes], "int64")
    ntype_dict = {ntype: i for i, ntype in enumerate(ntypes)}
    for rgrh in rel_graphs:
        stype, etype, dtype = rgrh.canonical_etypes[0]
        meta_edges_src.append(ntype_dict[stype])
        meta_edges_dst.append(ntype_dict[dtype])
        etypes.append(etype)
    # metagraph is DGLGraph, currently still using int64 as index dtype
    metagraph = graph_index.from_coo(len(ntypes), meta_edges_src, meta_edges_dst, True)

    # create graph index
    hgidx = heterograph_index.create_heterograph_from_relations(
        metagraph, [rgrh._graph for rgrh in rel_graphs], num_nodes_per_type)
    retg = DGLHeteroGraph(hgidx, ntypes, etypes)
    for i, rgrh in enumerate(rel_graphs):
        for ntype in rgrh.ntypes:
            retg.nodes[ntype].data.update(rgrh.nodes[ntype].data)
        retg._edge_frames[i].update(rgrh._edge_frames[0])
    return retg

def hetero_from_shared_memory(name):
    """Create a heterograph from shared memory with the given name.

    The newly created graph will have the same node types and edge types as the original graph.
    But it does not have node features or edges features.

    Paramaters
    ----------
    name : str
        The name of the share memory

    Returns
    -------
    HeteroGraph (in shared memory)
    """
    g, ntypes, etypes = heterograph_index.create_heterograph_from_shared_memory(name)
    return DGLHeteroGraph(g, ntypes, etypes)

def heterograph(data_dict,
                num_nodes_dict=None,
                validate=True,
                formats=['coo', 'csr', 'csc'],
                idtype=None,
                device=None):
    """Create a heterogeneous graph from a dictionary between edge types and edge lists.

    Parameters
    ----------
    data_dict : dict
        The dictionary between edge types and edge list data.

        The edge types are specified as a triplet of (source node type name, edge type
        name, destination node type name).

        The edge list data can be anything acceptable by :func:`dgl.graph` or
        :func:`dgl.bipartite`, or objects returned by the two functions themselves.
    num_nodes_dict : dict[str, int]
        The number of nodes for each node type.

        By default DGL infers the number of nodes for each node type from ``data_dict``
        by taking the maximum node ID plus one for each node type.
    validate : bool, optional
        If True, check if node ids are within cardinality, the check process may take
        some time. (Default: True)
        If False and num_nodes_dict is not None, user would receive a warning.
    formats : str or list of str
        It can be ``'coo'``/``'csr'``/``'csc'`` or a sublist of them,
        Force the storage formats.  Default: ``['coo', 'csr', 'csc']``.
    idtype : int32, int64, optional
        Integer ID type. Valid options are int32 or int64. If None, try infer from
        the given data.
    device : Device context, optional
        Device on which the graph is created. Default: infer from data.

    Returns
    -------
    DGLHeteroGraph

    Examples
    --------
    >>> g = dgl.heterograph({
    ...     ('user', 'follows', 'user'): [(0, 1), (1, 2)],
    ...     ('user', 'plays', 'game'): [(0, 0), (1, 0), (1, 1), (2, 1)],
    ...     ('developer', 'develops', 'game'): [(0, 0), (1, 1)],
    ...     })
    """
    # Try infer idtype
    if idtype is None:
        for data in data_dict.values():
            if isinstance(data, tuple) and len(data) == 2 and F.is_tensor(data[0]):
                idtype = F.dtype(data[0])
                break

    # Convert all data to edge tensors first.
    data_dict = {(sty, ety, dty) : utils.graphdata2tensors(data, idtype, bipartite=(sty != dty))
                 for (sty, ety, dty), data in data_dict.items()}

    # infer number of nodes for each node type
    if num_nodes_dict is None:
        num_nodes_dict = defaultdict(int)
        for (srctype, etype, dsttype), data in data_dict.items():
            _, _, nsrc, ndst = data
            num_nodes_dict[srctype] = max(num_nodes_dict[srctype], nsrc)
            num_nodes_dict[dsttype] = max(num_nodes_dict[dsttype], ndst)

    rel_graphs = []
    for (srctype, etype, dsttype), data in data_dict.items():
        u, v, _, _ = data
        if srctype == dsttype:
            rel_graphs.append(graph(
                (u, v), srctype, etype,
                num_nodes=num_nodes_dict[srctype],
                validate=validate,
                formats=formats,
                idtype=idtype, device=device))
        else:
            rel_graphs.append(bipartite(
                (u, v), srctype, etype, dsttype,
                num_nodes=(num_nodes_dict[srctype], num_nodes_dict[dsttype]),
                validate=validate,
                formats=formats,
                idtype=idtype, device=device))

    return hetero_from_relations(rel_graphs, num_nodes_dict)


def to_hetero(G, ntypes, etypes, ntype_field=NTYPE, etype_field=ETYPE,
              metagraph=None):
    """Convert the given homogeneous graph to a heterogeneous graph.

    The input graph should have only one type of nodes and edges. Each node and edge
    stores an integer feature (under ``ntype_field`` and ``etype_field``), representing
    the type id, which can be used to retrieve the type names stored
    in the given ``ntypes`` and ``etypes`` arguments.

    The function will automatically distinguish edge types that have the same given
    type IDs but different src and dst type IDs. For example, we allow both edges A and B
    to have the same type ID 0, but one has (0, 1) and the other as (2, 3) as the
    (src, dst) type IDs. In this case, the function will "split" edge type 0 into two types:
    (0, ty_A, 1) and (2, ty_B, 3). In another word, these two edges share the same edge
    type name, but can be distinguished by a canonical edge type tuple.

    Parameters
    ----------
    G : DGLHeteroGraph
        Input homogeneous graph.
    ntypes : list of str
        The node type names.
    etypes : list of str
        The edge type names.
    ntype_field : str, optional
        The feature field used to store node type. (Default: ``dgl.NTYPE``)
    etype_field : str, optional
        The feature field used to store edge type. (Default: ``dgl.ETYPE``)
    metagraph : networkx MultiDiGraph, optional
        Metagraph of the returned heterograph.
        If provided, DGL assumes that G can indeed be described with the given metagraph.
        If None, DGL will infer the metagraph from the given inputs, which would be
        potentially slower for large graphs.

    Returns
    -------
    DGLHeteroGraph
        A heterograph. The parent node and edge ID are stored in the column
        ``dgl.NID`` and ``dgl.EID`` respectively for all node/edge types.

    Notes
    -----
    The returned node and edge types may not necessarily be in the same order as
    ``ntypes`` and ``etypes``.  And edge types may be duplicated if the source
    and destination types differ.

    The node IDs of a single type in the returned heterogeneous graph is ordered
    the same as the nodes with the same ``ntype_field`` feature. Edge IDs of
    a single type is similar.

    Examples
    --------

    >>> g1 = dgl.bipartite([(0, 1), (1, 2)], 'user', 'develops', 'activity')
    >>> g2 = dgl.bipartite([(0, 0), (1, 1)], 'developer', 'develops', 'game')
    >>> hetero_g = dgl.hetero_from_relations([g1, g2])
    >>> print(hetero_g)
    Graph(num_nodes={'user': 2, 'activity': 3, 'developer': 2, 'game': 2},
          num_edges={('user', 'develops', 'activity'): 2, ('developer', 'develops', 'game'): 2},
          metagraph=[('user', 'activity'), ('developer', 'game')])

    We first convert the heterogeneous graph to a homogeneous graph.

    >>> homo_g = dgl.to_homo(hetero_g)
    >>> print(homo_g)
    Graph(num_nodes=9, num_edges=4,
          ndata_schemes={'_TYPE': Scheme(shape=(), dtype=torch.int64),
                         '_ID': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={'_TYPE': Scheme(shape=(), dtype=torch.int64),
                         '_ID': Scheme(shape=(), dtype=torch.int64)})
    >>> homo_g.ndata
    {'_TYPE': tensor([0, 0, 1, 1, 1, 2, 2, 3, 3]), '_ID': tensor([0, 1, 0, 1, 2, 0, 1, 0, 1])}
    Nodes 0, 1 for 'user', 2, 3, 4 for 'activity', 5, 6 for 'developer', 7, 8 for 'game'
    >>> homo_g.edata
    {'_TYPE': tensor([0, 0, 1, 1]), '_ID': tensor([0, 1, 0, 1])}
    Edges 0, 1 for ('user', 'develops', 'activity'), 2, 3 for ('developer', 'develops', 'game')

    Now convert the homogeneous graph back to a heterogeneous graph.

    >>> hetero_g_2 = dgl.to_hetero(homo_g, hetero_g.ntypes, hetero_g.etypes)
    >>> print(hetero_g_2)
    Graph(num_nodes={'user': 2, 'activity': 3, 'developer': 2, 'game': 2},
          num_edges={('user', 'develops', 'activity'): 2, ('developer', 'develops', 'game'): 2},
          metagraph=[('user', 'activity'), ('developer', 'game')])

    See Also
    --------
    dgl.to_homo
    """
    # TODO(minjie): use hasattr to support DGLGraph input; should be fixed once
    #  DGLGraph is merged with DGLHeteroGraph
    if (hasattr(G, 'ntypes') and len(G.ntypes) > 1
            or hasattr(G, 'etypes') and len(G.etypes) > 1):
        raise DGLError('The input graph should be homogenous and have only one '
                       ' type of nodes and edges.')

    num_ntypes = len(ntypes)
    idtype = G.idtype
    device = G.device

    ntype_ids = F.asnumpy(G.ndata[ntype_field])
    etype_ids = F.asnumpy(G.edata[etype_field])

    # relabel nodes to per-type local IDs
    ntype_count = np.bincount(ntype_ids, minlength=num_ntypes)
    ntype_offset = np.insert(np.cumsum(ntype_count), 0, 0)
    ntype_ids_sortidx = np.argsort(ntype_ids)
    ntype_local_ids = np.zeros_like(ntype_ids)
    node_groups = []
    for i in range(num_ntypes):
        node_group = ntype_ids_sortidx[ntype_offset[i]:ntype_offset[i+1]]
        node_groups.append(node_group)
        ntype_local_ids[node_group] = np.arange(ntype_count[i])

    src, dst = G.all_edges(order='eid')
    src = F.asnumpy(src)
    dst = F.asnumpy(dst)
    src_local = ntype_local_ids[src]
    dst_local = ntype_local_ids[dst]
    # a 2D tensor of shape (E, 3). Each row represents the (stid, etid, dtid) tuple.
    edge_ctids = np.stack([ntype_ids[src], etype_ids, ntype_ids[dst]], 1)

    # infer metagraph and canonical edge types
    # No matter which branch it takes, the code will generate a 2D tensor of shape (E_m, 3),
    # E_m is the set of all possible canonical edge tuples. Each row represents the
    # (stid, dtid, dtid) tuple. We then compute a 2D tensor of shape (E, E_m) using the
    # above ``edge_ctids`` matrix. Each element i,j indicates whether the edge i is of the
    # canonical edge type j. We can then group the edges of the same type together.
    if metagraph is None:
        canonical_etids, _, etype_remapped = \
                utils.make_invmap(list(tuple(_) for _ in edge_ctids), False)
        etype_mask = (etype_remapped[None, :] == np.arange(len(canonical_etids))[:, None])
    else:
        ntypes_invmap = {nt: i for i, nt in enumerate(ntypes)}
        etypes_invmap = {et: i for i, et in enumerate(etypes)}
        canonical_etids = []
        for i, (srctype, dsttype, etype) in enumerate(metagraph.edges(keys=True)):
            srctype_id = ntypes_invmap[srctype]
            etype_id = etypes_invmap[etype]
            dsttype_id = ntypes_invmap[dsttype]
            canonical_etids.append((srctype_id, etype_id, dsttype_id))
        canonical_etids = np.asarray(canonical_etids)
        etype_mask = (edge_ctids[None, :] == canonical_etids[:, None]).all(2)
    edge_groups = [etype_mask[i].nonzero()[0] for i in range(len(canonical_etids))]

    rel_graphs = []
    for i, (stid, etid, dtid) in enumerate(canonical_etids):
        src_of_etype = src_local[edge_groups[i]]
        dst_of_etype = dst_local[edge_groups[i]]
        if stid == dtid:
            rel_graph = graph(
                (src_of_etype, dst_of_etype), ntypes[stid], etypes[etid],
                num_nodes=ntype_count[stid], validate=False,
                idtype=idtype, device=device)
        else:
            rel_graph = bipartite(
                (src_of_etype,
                 dst_of_etype), ntypes[stid], etypes[etid], ntypes[dtid],
                num_nodes=(ntype_count[stid], ntype_count[dtid]),
                validate=False, idtype=idtype, device=device)
        rel_graphs.append(rel_graph)

    hg = hetero_from_relations(rel_graphs,
                               {ntype: count for ntype, count in zip(
                                   ntypes, ntype_count)})

    ntype2ngrp = {ntype : node_groups[ntid] for ntid, ntype in enumerate(ntypes)}

    # features
    for key, data in G.ndata.items():
        for ntid, ntype in enumerate(hg.ntypes):
            rows = F.copy_to(F.tensor(ntype2ngrp[ntype]), F.context(data))
            hg._node_frames[ntid][key] = F.gather_row(data, rows)
    for key, data in G.edata.items():
        for etid in range(len(hg.canonical_etypes)):
            rows = F.copy_to(F.tensor(edge_groups[etid]), F.context(data))
            hg._edge_frames[etid][key] = F.gather_row(data, rows)

    for ntid, ntype in enumerate(hg.ntypes):
        hg._node_frames[ntid][NID] = F.tensor(ntype2ngrp[ntype])

    for etid in range(len(hg.canonical_etypes)):
        hg._edge_frames[etid][EID] = F.tensor(edge_groups[etid])

    return hg

def to_homo(G):
    """Convert the given heterogeneous graph to a homogeneous graph.

    The returned graph has only one type of nodes and edges.

    Node and edge types are stored as features in the returned graph. Each feature
    is an integer representing the type id, which can be used to retrieve the type
    names stored in ``G.ntypes`` and ``G.etypes`` arguments.

    Parameters
    ----------
    G : DGLHeteroGraph
        Input heterogeneous graph.

    Returns
    -------
    DGLHeteroGraph
        A homogeneous graph. The parent node and edge type/ID are stored in
        columns ``dgl.NTYPE/dgl.NID`` and ``dgl.ETYPE/dgl.EID`` respectively.

    Examples
    --------

    >>> follows_g = dgl.graph([(0, 1), (1, 2)], 'user', 'follows')
    >>> devs_g = dgl.bipartite([(0, 0), (1, 1)], 'developer', 'develops', 'game')
    >>> hetero_g = dgl.hetero_from_relations([follows_g, devs_g])
    >>> homo_g = dgl.to_homo(hetero_g)
    >>> homo_g.ndata
    {'_TYPE': tensor([0, 0, 0, 1, 1, 2, 2]), '_ID': tensor([0, 1, 2, 0, 1, 0, 1])}
    First three nodes for 'user', next two for 'developer' and the last two for 'game'
    >>> homo_g.edata
    {'_TYPE': tensor([0, 0, 1, 1]), '_ID': tensor([0, 1, 0, 1])}
    First two edges for 'follows', next two for 'develops'

    See Also
    --------
    dgl.to_hetero
    """
    num_nodes_per_ntype = [G.number_of_nodes(ntype) for ntype in G.ntypes]
    offset_per_ntype = np.insert(np.cumsum(num_nodes_per_ntype), 0, 0)
    srcs = []
    dsts = []
    etype_ids = []
    eids = []
    ntype_ids = []
    nids = []
    total_num_nodes = 0

    for ntype_id, ntype in enumerate(G.ntypes):
        num_nodes = G.number_of_nodes(ntype)
        total_num_nodes += num_nodes
        # Type ID is always in int64
        ntype_ids.append(F.full_1d(num_nodes, ntype_id, F.int64, F.cpu()))
        nids.append(F.arange(0, num_nodes, G.idtype))

    for etype_id, etype in enumerate(G.canonical_etypes):
        srctype, _, dsttype = etype
        src, dst = G.all_edges(etype=etype, order='eid')
        num_edges = len(src)
        srcs.append(src + int(offset_per_ntype[G.get_ntype_id(srctype)]))
        dsts.append(dst + int(offset_per_ntype[G.get_ntype_id(dsttype)]))
        # Type ID is always in int64
        etype_ids.append(F.full_1d(num_edges, etype_id, F.int64, F.cpu()))
        eids.append(F.arange(0, num_edges, G.idtype))

    retg = graph((F.cat(srcs, 0), F.cat(dsts, 0)), num_nodes=total_num_nodes,
                 validate=False, idtype=G.idtype, device=G.device)

    # copy features
    comb_nf = combine_frames(G._node_frames, range(len(G.ntypes)))
    comb_ef = combine_frames(G._edge_frames, range(len(G.etypes)))
    if comb_nf is not None:
        retg.ndata.update(comb_nf)
    if comb_ef is not None:
        retg.edata.update(comb_ef)

    # assign node type and id mapping field.
    retg.ndata[NTYPE] = F.copy_to(F.cat(ntype_ids, 0), G.device)
    retg.ndata[NID] = F.copy_to(F.cat(nids, 0), G.device)
    retg.edata[ETYPE] = F.copy_to(F.cat(etype_ids, 0), G.device)
    retg.edata[EID] = F.copy_to(F.cat(eids, 0), G.device)

    return retg

def from_scipy(sp_mat,
               ntype='_N', etype='_E',
               eweight_name=None,
               formats=['coo', 'csr', 'csc'],
               idtype=None):
    """Create a DGLGraph from a SciPy sparse matrix.

    Parameters
    ----------
    sp_mat : SciPy sparse matrix
        SciPy sparse matrix.
    ntype : str
        Type name for both source and destination nodes
    etype : str
        Type name for edges
    eweight_name : str, optional
        If given, the edge weights in the matrix will be
        stored in ``edata[eweight_name]``.
    formats : str or list of str
        It can be ``'coo'``/``'csr'``/``'csc'`` or a sublist of them,
        Force the storage formats.  Default: ``['coo', 'csr', 'csc']``.
    idtype : int32, int64, optional
        Integer ID type. Must be int32 or int64. Default: int64.

    Returns
    -------
    g : DGLGraph
    """
    u, v, urange, vrange = utils.graphdata2tensors(sp_mat, idtype)
    g = create_from_edges(u, v, ntype, etype, ntype, urange, vrange,
                          validate=False, formats=formats)
    if eweight_name is not None:
        g.edata[eweight_name] = F.tensor(sp_mat.data)
    return g

def from_networkx(nx_graph, *,
                  ntype='_N', etype='_E',
                  node_attrs=None,
                  edge_attrs=None,
                  edge_id_attr_name='id',
                  formats=['coo', 'csr', 'csc'],
                  idtype=None):
    """Create a DGLGraph from networkx.

    Parameters
    ----------
    nx_graph : networkx.Graph
        NetworkX graph.
    ntype : str
        Type name for both source and destination nodes
    etype : str
        Type name for edges
    node_attrs : list of str
        Names for node features to retrieve from the NetworkX graph (Default: None)
    edge_attrs : list of str
        Names for edge features to retrieve from the NetworkX graph (Default: None)
    edge_id_attr_name : str, optional
        Key name for edge ids in the NetworkX graph. If not found, we
        will consider the graph not to have pre-specified edge ids. (Default: 'id')
    formats : str or list of str
        It can be ``'coo'``/``'csr'``/``'csc'`` or a sublist of them,
        Force the storage formats.  Default: ``['coo', 'csr', 'csc']``.
    idtype : int32, int64, optional
        Integer ID type. Must be int32 or int64. Default: int64.

    Returns
    -------
    g : DGLGraph
    """
    # Relabel nodes using consecutive integers
    nx_graph = nx.convert_node_labels_to_integers(nx_graph, ordering='sorted')
    if not nx_graph.is_directed():
        nx_graph = nx_graph.to_directed()

    g = graph(nx_graph, ntype, etype,
              formats=formats,
              idtype=idtype)

    # nx_graph.edges(data=True) returns src, dst, attr_dict
    if nx_graph.number_of_edges() > 0:
        has_edge_id = edge_id_attr_name in next(iter(nx_graph.edges(data=True)))[-1]
    else:
        has_edge_id = False

    # handle features
    # copy attributes
    def _batcher(lst):
        if F.is_tensor(lst[0]):
            return F.cat([F.unsqueeze(x, 0) for x in lst], dim=0)
        else:
            return F.tensor(lst)
    if node_attrs is not None:
        # mapping from feature name to a list of tensors to be concatenated
        attr_dict = defaultdict(list)
        for nid in range(g.number_of_nodes()):
            for attr in node_attrs:
                attr_dict[attr].append(nx_graph.nodes[nid][attr])
        for attr in node_attrs:
            g.ndata[attr] = F.copy_to(_batcher(attr_dict[attr]), g.device)

    if edge_attrs is not None:
        # mapping from feature name to a list of tensors to be concatenated
        attr_dict = defaultdict(lambda: [None] * g.number_of_edges())
        # each defaultdict value is initialized to be a list of None
        # None here serves as placeholder to be replaced by feature with
        # corresponding edge id
        if has_edge_id:
            num_edges = g.number_of_edges()
            for _, _, attrs in nx_graph.edges(data=True):
                if attrs[edge_id_attr_name] >= num_edges:
                    raise DGLError('Expect the pre-specified edge ids to be'
                                   ' smaller than the number of edges --'
                                   ' {}, got {}.'.format(num_edges, attrs['id']))
                for key in edge_attrs:
                    attr_dict[key][attrs[edge_id_attr_name]] = attrs[key]
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
            g.edata[attr] = F.copy_to(_batcher(attr_dict[attr]), g.device)

    return g

def to_networkx(g, node_attrs=None, edge_attrs=None):
    """Convert to networkx graph.

    The edge id will be saved as the 'id' edge attribute.

    Parameters
    ----------
    g : DGLGraph or DGLHeteroGraph
        For DGLHeteroGraphs, we currently only support the
        case of one node type and one edge type.
    node_attrs : iterable of str, optional
        The node attributes to be copied. (Default: None)
    edge_attrs : iterable of str, optional
        The edge attributes to be copied. (Default: None)

    Returns
    -------
    networkx.DiGraph
        The nx graph
    """
    if g.device != F.cpu():
        raise DGLError('Cannot convert a CUDA graph to networkx. Call g.cpu() first.')
    if not g.is_homogeneous():
        raise DGLError('dgl.to_networkx only supports homogeneous graphs.')
    src, dst = g.edges()
    src = F.asnumpy(src)
    dst = F.asnumpy(dst)
    # xiangsx: Always treat graph as multigraph
    nx_graph = nx.MultiDiGraph()
    nx_graph.add_nodes_from(range(g.number_of_nodes()))
    for eid, (u, v) in enumerate(zip(src, dst)):
        nx_graph.add_edge(u, v, id=eid)

    if node_attrs is not None:
        for nid, attr in nx_graph.nodes(data=True):
            feat_dict = g._get_n_repr(0, nid)
            attr.update({key: F.squeeze(feat_dict[key], 0) for key in node_attrs})
    if edge_attrs is not None:
        for _, _, attr in nx_graph.edges(data=True):
            eid = attr['id']
            feat_dict = g._get_e_repr(0, eid)
            attr.update({key: F.squeeze(feat_dict[key], 0) for key in edge_attrs})
    return nx_graph

DGLHeteroGraph.to_networkx = to_networkx

############################################################
# Internal APIs
############################################################

def create_from_edges(u, v,
                      utype, etype, vtype,
                      urange, vrange,
                      validate=True,
                      formats=['coo', 'csr', 'csc']):
    """Internal function to create a graph from incident nodes with types.

    utype could be equal to vtype

    Parameters
    ----------
    u : Tensor
        Source node IDs.
    v : Tensor
        Dest node IDs.
    utype : str
        Source node type name.
    etype : str
        Edge type name.
    vtype : str
        Destination node type name.
    urange : int, optional
        The source node ID range. If None, the value is the maximum
        of the source node IDs in the edge list plus 1. (Default: None)
    vrange : int, optional
        The destination node ID range. If None, the value is the
        maximum of the destination node IDs in the edge list plus 1. (Default: None)
    validate : bool, optional
        If True, checks if node IDs are within range.
    formats : str or list of str
        It can be ``'coo'``/``'csr'``/``'csc'`` or a sublist of them,
        Force the storage formats.  Default: ``['coo', 'csr', 'csc']``.

    Returns
    -------
    DGLHeteroGraph
    """
    if validate:
        if urange is not None and len(u) > 0 and \
            urange <= F.as_scalar(F.max(u, dim=0)):
            raise DGLError('Invalid node id {} (should be less than cardinality {}).'.format(
                urange, F.as_scalar(F.max(u, dim=0))))
        if vrange is not None and len(v) > 0 and \
            vrange <= F.as_scalar(F.max(v, dim=0)):
            raise DGLError('Invalid node id {} (should be less than cardinality {}).'.format(
                vrange, F.as_scalar(F.max(v, dim=0))))

    if utype == vtype:
        num_ntypes = 1
    else:
        num_ntypes = 2

    if 'coo' in formats:
        hgidx = heterograph_index.create_unitgraph_from_coo(
            num_ntypes, urange, vrange, u, v, formats)
    else:
        hgidx = heterograph_index.create_unitgraph_from_coo(
            num_ntypes, urange, vrange, u, v, ['coo']).formats(formats)
    if utype == vtype:
        return DGLHeteroGraph(hgidx, [utype], [etype])
    else:
        return DGLHeteroGraph(hgidx, [utype, vtype], [etype])
