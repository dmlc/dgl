"""Module for converting graph from/to other object."""

__all__ = [
    'bipartite_from_edges',
    'bipartite_from_edge_list',
    'bipartite_from_scipy',
    'hetero_from_relations',
    'hetero_from_homo',
    'hetero_to_homo',
]

def bipartite_from_edges(u, v, src_type, edge_type, dst_type, num_src=None, num_dst=None):
    """Create a bipartite graph given edges in two index arrays.

    Examples
    --------
    TBD

    Parameters
    ----------
    u : iterable of int
        List of source node IDs.
    v : iterable of int
        List of destination node IDs.
    src_type : str
        Source node type name.
    edge_type : str
        Edge type name.
    dst_type : str
        Destination node type name.
    num_src : int, optional
        The number of nodes of source type. If None, the value is the maximum
        of the source node IDs in the edge list plus 1. (Default: None)
    num_dst : int, optional
        The number of nodes of destination type. If None, the value is the
        maximum of the destination node IDs in the edge list plus 1. (Default: None)

    Returns
    -------
    DGLHeteroGraph
        A bipartite graph.
    """
    # TODO
    num_src = num_src or (max(u) + 1)
    num_dst = num_dst or (max(v) + 1)
    u = utils.toindex(u)
    v = utils.toindex(v)
    return heterograph_index.create_bipartite_from_coo(num_src, num_dst, u, v)

def bipartite_from_edge_list(elist, src_type, edge_type, dst_type, num_src=None, num_dst=None):
    """Create a bipartite graph given a list of edge tuples

    Examples
    --------
    TBD

    Parameters
    ----------
    elist : iterable of int pairs
        List of (src, dst) node ID pairs.
    src_type : str
        Source node type name.
    edge_type : str
        Edge type name.
    dst_type : str
        Destination node type name.
    num_src : int, optional
        The number of nodes of source type. If None, the value is the maximum
        of the source node IDs in the edge list plus 1. (Default: None)
    num_dst : int, optional
        The number of nodes of destination type. If None, the value is the
        maximum of the destination node IDs in the edge list plus 1. (Default: None)

    Returns
    -------
    DGLHeteroGraph
        A bipartite graph.
    """
    # TODO
    pass

def bipartite_from_scipy(spmat, src_type, edge_type, dst_type, with_edge_id=False):
    """Create a bipartite graph from a scipy sparse matrix.

    Parameters
    ----------
    spmat : scipy.sparse.spmatrix
        The bipartite graph matrix whose rows represent sources and columns
        represent destinations.
    src_type : str
        Source node type name.
    edge_type : str
        Edge type name.
    dst_type : str
        Destination node type name.
    with_edge_id : bool
        If True, the entries in the sparse matrix are treated as edge IDs.
        Otherwise, the entries are ignored and edges will be added in
        (source, destination) order.

    Returns
    -------
    DGLHeteroGraph
        A bipartite graph.
    """
    # TODO(handle both csr and coo)
    spmat = spmat.tocsr()
    num_src, num_dst = spmat.shape
    indptr = utils.toindex(spmat.indptr)
    indices = utils.toindex(spmat.indices)
    data = utils.toindex(spmat.data if with_edge_id else list(range(len(indices))))
    return heterograph_index.create_bipartite_from_csr(num_src, num_dst, indptr, indices, data)

def hetero_from_relations(meta_graph, rel_graphs):
    """Create a heterograph from meta-graph and per-relation graphs.

    Examples
    --------
    TBD

    Parameters
    ----------
    meta_graph : networkx.MultiDiGraph
        The meta graph.
    rel_graphs : dict of graph data
        The key is the relation name and the value is any graph data that
        can be converted to a bipartite graph (e.g. edge list, scipy sparse matrix)

    Returns
    -------
    DGLHeteroGraph
        A heterograph.
    """
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


    pass

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

def hetero_from_networkx(
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
