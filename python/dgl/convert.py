"""Module for converting graph from/to other object."""

import networkx as nx
from . import heterograph_index
from .heterograph import DGLHeteroGraph
from . import graph_index
from . import utils

__all__ = [
    'from_edges',
    'from_edges2',
    'from_edge_list2',
    'from_scipy2',
    'hetero_from_relations',
    'hetero_from_homo',
    'hetero_to_homo',
]

def from_edges(u, v, ntype=None, etype=None, nrange=None):
    """TBD
    """
    pass

def from_edges2(u, v, utype, etype, vtype, urange=None, vrange=None):
    """Create a graph from incident nodes with types.

    Examples
    --------
    TBD

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
    u = utils.toindex(u)
    v = utils.toindex(v)
    hgidx = heterograph_index.create_bipartite_from_coo(urange, vrange, u, v)
    if utype == vtype:
        return DGLHeteroGraph(hgidx, [utype], [etype])
    else:
        return DGLHeteroGraph(hgidx, [utype, vtype], [etype])

def from_edge_list2(elist, utype, etype, vtype, urange=None, vrange=None):
    """Create a graph from a list of edge tuples with types.

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
    u, v = zip(*elist)
    u = list(u)
    v = list(v)
    return from_edges2(u, v, utype, etype, vtype, urange, vrange)

def from_scipy2(spmat, utype, etype, vtype, with_edge_id=False):
    """Create a graph from a scipy sparse matrix with types.

    Parameters
    ----------
    spmat : scipy.sparse.spmatrix
        The bipartite graph matrix whose rows represent sources and columns
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
    if spmat.getformat() == 'coo':
        row = utils.toindex(spmat.row)
        col = utils.toindex(spmat.col)
        hgidx = heterograph_index.create_bipartite_from_coo(num_src, num_dst, row, col)
    else:
        spmat = spmat.tocsr()
        indptr = utils.toindex(spmat.indptr)
        indices = utils.toindex(spmat.indices)
        # TODO(minjie): with_edge_id is only reasonable for csr matrix. How to fix?
        data = utils.toindex(spmat.data if with_edge_id else list(range(len(indices))))
        hgidx = heterograph_index.create_bipartite_from_csr(num_src, num_dst, indptr, indices, data)
    return DGLHeteroGraph(hgidx, [utype, vtype], [etype])

def graph(data, utype='_N', etype='_E', vtype='_N', card=None):
    pass

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
    etype_dict = {}  # etype -> etid
    meta_edges = []
    ntypes = []
    etypes = []
    for rg in rel_graphs:
        assert len(rg.etypes) == 1
        stype, etype, dtype = rg.canonical_etypes[0]
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
        metagraph, [rg._graph for rg in rel_graphs])
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
