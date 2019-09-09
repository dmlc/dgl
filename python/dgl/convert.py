"""Module for converting graph from/to other object."""
from collections import defaultdict
import numpy as np
import scipy as sp
import networkx as nx

from . import backend as F
from . import heterograph_index
from .heterograph import DGLHeteroGraph
from . import graph_index
from . import utils

__all__ = [
    'graph',
    'bipartite',
    'hetero_from_relations',
    'hetero_from_homo',
    'hetero_to_homo',
    'to_networkx',
]

def graph(data, ntype='_N', etype='_E', card=None, **kwargs):
    """Create a graph.

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
    card : int, optional
        Cardinality (number of nodes in the graph). If None, infer from input data.
        (Default: None)
    kwargs : key-word arguments, optional
        Other key word arguments.

    Returns
    -------
    DGLHeteroGraph
    """
    if card is not None:
        urange, vrange = card, card
    else:
        urange, vrange = None, None
    if isinstance(data, tuple):
        u, v = data
        return create_from_edges(u, v, ntype, etype, ntype, urange, vrange)
    elif isinstance(data, list):
        return create_from_edge_list(data, ntype, etype, ntype, urange, vrange)
    elif isinstance(data, sp.sparse.spmatrix):
        return create_from_scipy(data, ntype, etype, ntype)
    elif isinstance(data, nx.Graph):
        return create_from_networkx(data, ntype, etype, **kwargs)
    else:
        raise DGLError('Unsupported graph data type:', type(data))

def bipartite(data, utype='_U', etype='_E', vtype='_V', card=None, **kwargs):
    """Create a bipartite graph.

    The result graph is directed and edges are from ``utype`` nodes
    to ``vtype`` nodes.

    Examples
    --------
    TBD

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
    card : pair of int, optional
        Cardinality (number of nodes in the source and destination group). If None,
        infer from input data.  (Default: None)
    kwargs : key-word arguments, optional
        Other key word arguments.

    Returns
    -------
    DGLHeteroGraph
    """
    if utype == vtype:
        raise DGLError('utype should not be equal to vtype. Use ``dgl.graph`` instead.')
    if card is not None:
        urange, vrange = card
    else:
        urange, vrange = None, None
    if isinstance(data, tuple):
        u, v = data
        return create_from_edges(u, v, utype, etype, vtype, urange, vrange)
    elif isinstance(data, list):
        return create_from_edge_list(data, utype, etype, vtype, urange, vrange)
    elif isinstance(data, sp.sparse.spmatrix):
        return create_from_scipy(data, utype, etype, vtype)
    elif isinstance(data, nx.Graph):
        return create_from_networkx_bipartite(data, utype, etype, vtype, **kwargs)
    else:
        raise DGLError('Unsupported graph data type:', type(data))

def hetero_from_relations(rel_graphs):
    """Create a heterograph from per-relation graphs.

    TODO(minjie): this API can be generalized as a union operation of
    the input graphs

    TODO(minjie): handle node/edge data

    Parameters
    ----------
    rel_graphs : list of DGLHeteroGraph
        Graph for each relation.

    Returns
    -------
    DGLHeteroGraph
        A heterograph.
    """
    # infer meta graph
    ntype_dict = {}  # ntype -> ntid
    meta_edges = []
    ntypes = []
    etypes = []
    for rgrh in rel_graphs:
        assert len(rgrh.etypes) == 1
        stype, etype, dtype = rgrh.canonical_etypes[0]
        if not stype in ntype_dict:
            ntype_dict[stype] = len(ntypes)
            ntypes.append(stype)
        stid = ntype_dict[stype]
        if not dtype in ntype_dict:
            ntype_dict[dtype] = len(ntypes)
            ntypes.append(dtype)
        dtid = ntype_dict[dtype]
        meta_edges.append((stid, dtid))
        etypes.append(etype)
    metagraph = graph_index.from_edge_list(meta_edges, True, True)
    # create graph index
    hgidx = heterograph_index.create_heterograph_from_relations(
        metagraph, [rgrh._graph for rgrh in rel_graphs])
    return DGLHeteroGraph(hgidx, ntypes, etypes)

def hetero_from_homo(graph, ntypes, etypes, ntype_field='type', etype_field='type'):
    """Create a heterograph from a DGLGraph.

    Node and edge types are stored as features. Each feature must be an integer
    representing the type id, which can be used to retrieve the type names stored
    in the given ``ntypes`` and ``etypes`` arguments.

    Examples
    --------
    TBD

    Parameters
    ----------
    graph : DGLGraph
        Input homogenous graph.
    ntypes : list of str
        The node type names.
    etypes : list of str
        The edge type names.
    ntype_field : str, optional
        The feature field used to store node type. (Default: 'type')
    etype_field : str, optional
        The feature field used to store edge type. (Default: 'type')

    Returns
    -------
    DGLHeteroGraph
        A heterograph.
        The parent node and edge ID are stored in the column dgl.NID and dgl.EID
        respectively for all node/edge types.
    """
    pass

def hetero_to_homo(hgraph, ntype_field='type', etype_field='type'):
    """Convert a heterograph to a DGLGraph.

    Node and edge types are stored as features in the returned graph. Each feature
    is an integer representing the type id,  which can be used to retrieve the type
    names stored in ``hgraph.ntypes`` and ``hgraph.etypes`` arguments.

    Examples
    --------
    TBD

    Parameters
    ----------
    hgraph : DGLHeteroGraph
        Input heterogenous graph.
    ntype_field : str, optional
        The feature field used to store node type. (Default: 'type')
    etype_field : str, optional
        The feature field used to store edge type. (Default: 'type')

    Returns
    -------
    DGLGraph
        A homogenous graph.
        The parent node and edge type/ID are stored in columns dgl.NTYPE/dgl.NID and
        dgl.ETYPE/dgl.EID respectively.
    """
    pass

############################################################
# Internal APIs
############################################################

def create_from_edges(u, v, utype, etype, vtype, urange=None, vrange=None):
    """Internal function to create a graph from incident nodes with types.

    utype could be equal to vtype

    Parameters
    ----------
    u : iterable of int
        List of source node IDs.
    v : iterable of int
        List of destination node IDs.
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

    Returns
    -------
    DGLHeteroGraph
    """
    urange = urange or (max(u) + 1)
    vrange = vrange or (max(v) + 1)
    if utype == vtype:
        urange = vrange = max(urange, vrange)
        num_ntypes = 1
    else:
        num_ntypes = 2
    u = utils.toindex(u)
    v = utils.toindex(v)
    hgidx = heterograph_index.create_unitgraph_from_coo(num_ntypes, urange, vrange, u, v)
    if utype == vtype:
        return DGLHeteroGraph(hgidx, [utype], [etype])
    else:
        return DGLHeteroGraph(hgidx, [utype, vtype], [etype])

def create_from_edge_list(elist, utype, etype, vtype, urange=None, vrange=None):
    """Internal function to create a graph from a list of edge tuples with types.

    utype could be equal to vtype

    Examples
    --------
    TBD

    Parameters
    ----------
    elist : iterable of int pairs
        List of (src, dst) node ID pairs.
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

    Returns
    -------
    DGLHeteroGraph
    """
    if len(elist) == 0:
        u, v = [], []
    else:
        u, v = zip(*elist)
        u = list(u)
        v = list(v)
    return create_from_edges(u, v, utype, etype, vtype, urange, vrange)

def create_from_scipy(spmat, utype, etype, vtype, with_edge_id=False):
    """Internal function to create a graph from a scipy sparse matrix with types.

    Parameters
    ----------
    spmat : scipy.sparse.spmatrix
        The adjacency matrix whose rows represent sources and columns
        represent destinations.
    utype : str
        Source node type name.
    etype : str
        Edge type name.
    vtype : str
        Destination node type name.
    with_edge_id : bool
        If True, the entries in the sparse matrix are treated as edge IDs.
        Otherwise, the entries are ignored and edges will be added in
        (source, destination) order.

    Returns
    -------
    DGLHeteroGraph
    """
    num_src, num_dst = spmat.shape
    num_ntypes = 1 if utype == vtype else 2
    if spmat.getformat() == 'coo':
        row = utils.toindex(spmat.row)
        col = utils.toindex(spmat.col)
        hgidx = heterograph_index.create_unitgraph_from_coo(
            num_ntypes, num_src, num_dst, row, col)
    else:
        spmat = spmat.tocsr()
        indptr = utils.toindex(spmat.indptr)
        indices = utils.toindex(spmat.indices)
        # TODO(minjie): with_edge_id is only reasonable for csr matrix. How to fix?
        data = utils.toindex(spmat.data if with_edge_id else list(range(len(indices))))
        hgidx = heterograph_index.create_unitgraph_from_csr(
            num_ntypes, num_src, num_dst, indptr, indices, data)
    return DGLHeteroGraph(hgidx, [utype, vtype], [etype])

def create_from_networkx(nx_graph,
                         ntype, etype,
                         edge_id_attr_name='id',
                         node_attrs=None,
                         edge_attrs=None):
    """Create graph that has only one set of nodes and edges.
    """
    if not nx_graph.is_directed():
        nx_graph = nx_graph.to_directed()

    # Relabel nodes using consecutive integers
    nx_graph = nx.convert_node_labels_to_integers(nx_graph, ordering='sorted')

    # nx_graph.edges(data=True) returns src, dst, attr_dict
    if nx_graph.number_of_edges() > 0:
        has_edge_id = edge_id_attr_name in next(iter(nx_graph.edges(data=True)))[-1]
    else:
        has_edge_id = False

    if has_edge_id:
        num_edges = nx_graph.number_of_edges()
        src = np.zeros((num_edges,), dtype=np.int64)
        dst = np.zeros((num_edges,), dtype=np.int64)
        for u, v, attr in nx_graph.edges(data=True):
            eid = attr[edge_id_attr_name]
            src[eid] = u
            dst[eid] = v
    else:
        src = []
        dst = []
        for e in nx_graph.edges:
            src.append(e[0])
            dst.append(e[1])
    src = utils.toindex(src)
    dst = utils.toindex(dst)
    num_nodes = nx_graph.number_of_nodes()
    g = create_from_edges(src, dst, ntype, etype, ntype, num_nodes, num_nodes)

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
            g.ndata[attr] = _batcher(attr_dict[attr])

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
            g.edata[attr] = _batcher(attr_dict[attr])

    return g

def create_from_networkx_bipartite(nx_graph,
                                   utype, etype, vtype,
                                   edge_id_attr_name='id',
                                   node_attrs=None,
                                   edge_attrs=None):
    """Create graph that has only one set of nodes and edges.

    The input graph must follow the bipartite graph convention of networkx.
    Each node has an attribute ``bipartite`` with values 0 and 1 indicating which
    set it belongs to. Edges are all from node set 0 to node set 1.
    """
    if not nx_graph.is_directed():
        nx_graph = nx_graph.to_directed()

    top_nodes = {n for n, d in nx_graph.nodes(data=True) if d['bipartite'] == 0}
    bottom_nodes = set(nx_graph) - top_nodes
    top_nodes = sorted(top_nodes)
    bottom_nodes = sorted(bottom_nodes)
    top_map = {n : i for i, n in enumerate(top_nodes)}
    bottom_map = {n : i for i, n in enumerate(bottom_nodes)}

    if nx_graph.number_of_edges() > 0:
        has_edge_id = edge_id_attr_name in next(iter(nx_graph.edges(data=True)))[-1]
    else:
        has_edge_id = False

    if has_edge_id:
        num_edges = nx_graph.number_of_edges()
        src = np.zeros((num_edges,), dtype=np.int64)
        dst = np.zeros((num_edges,), dtype=np.int64)
        for u, v, attr in nx_graph.edges(data=True):
            eid = attr[edge_id_attr_name]
            src[eid] = top_map[u]
            dst[eid] = bottom_map[v]
    else:
        src = []
        dst = []
        for e in nx_graph.edges:
            src.append(top_map[e[0]])
            dst.append(bottom_map[e[1]])
    src = utils.toindex(src)
    dst = utils.toindex(dst)
    g = create_from_edges(src, dst, utype, etype, vtype, len(top_nodes), len(bottom_nodes))

    # TODO attributes
    assert node_attrs is None
    assert edge_attrs is None
    return g

def create_from_networkx_multi(nx_graph,
                               edge_id_attr_name='id',
                               node_type_attr_name='type',
                               node_attrs=None,
                               edge_attrs=None):
    """Create a graph containing multiple types from networkx.

    The input is an nx.MultiDiGraph. The edge key is treated as edge
    types. The node types are stored as node attributes.

    Examples
    --------
    TBD

    Parameters
    ----------
    nx_graph : networkx.MultiDiGraph
        The networkx graph.
    edge_id_attr_name : str, optional
        The edge attribute name for edge type-specific IDs.
        The attribute contents must be integers.
        If the IDs of the same type are not consecutive integers, its
        nodes will be relabeled using consecutive integers.  The new
        node ordering will inherit that of the sorted IDs.
        If None is provided, the edge order would be arbitrary.
    node_type_attr_name : str, optional
        The node attribute name for the node type.
        The attribute contents must be strings.
    node_attrs : iterable of str, optional
        The node attributes whose data would be copied.
    edge_attrs : iterable of str, optional
        The edge attributes whose data would be copied.

    Returns
    -------
    DGLHeteroGraph
    """
    pass

def to_networkx(g, node_attrs=None, edge_attrs=None):
    """Convert to networkx graph.

    See Also
    --------
    DGLGraph.to_networkx
    """
    return g.to_networkx(node_attrs, edge_attrs)
