"""Functions for extracting subgraphs.

The module only contains functions for extracting subgraphs deterministically.
For stochastic subgraph extraction, please see functions under :mod:`dgl.sampling`.
"""
from collections.abc import Mapping

from ._ffi.function import _init_api
from .base import DGLError
from . import backend as F
from . import graph_index
from . import heterograph_index
from . import ndarray as nd
from .heterograph import DGLHeteroGraph
from . import utils

__all__ = ['node_subgraph', 'edge_subgraph', 'node_type_subgraph', 'edge_type_subgraph',
           'in_subgraph', 'out_subgraph']

def node_subgraph(graph, nodes):
    """Return the subgraph induced on given nodes.

    The metagraph of the returned subgraph is the same as the parent graph.
    Features are copied from the original graph.

    Parameters
    ----------
    graph : DGLGraph
        The graph to extract subgraphs from.
    nodes : list or dict[str->list or iterable]
        A dictionary mapping node types to node ID array for constructing
        subgraph. All nodes must exist in the graph.

        If the graph only has one node type, one can just specify a list,
        tensor, or any iterable of node IDs intead.

        The node ID array can be either an interger tensor or a bool tensor.
        When a bool tensor is used, it is automatically converted to
        an interger tensor using the semantic of np.where(nodes_idx == True).

        Note: When using bool tensor, only backend (torch, tensorflow, mxnet)
        tensors are supported.

    Returns
    -------
    G : DGLGraph
        The subgraph.

        The nodes and edges in the subgraph are relabeled using consecutive
        integers from 0.

        One can retrieve the mapping from subgraph node/edge ID to parent
        node/edge ID via ``dgl.NID`` and ``dgl.EID`` node/edge features of the
        subgraph.

    Examples
    --------
    The following example uses PyTorch backend.

    Instantiate a heterograph.

    >>> g = dgl.heterograph({
    ...     ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 2, 1]),
    ...     ('user', 'follows', 'user'): ([0, 1, 1], [1, 2, 2])})
    >>> # Set node features
    >>> g.nodes['user'].data['h'] = torch.tensor([[0.], [1.], [2.]])

    Get subgraphs.

    >>> g.subgraph({'user': [4, 5]})
    Traceback (most recent call last):
        ...
    dgl._ffi.base.DGLError: ...
    >>> sub_g = g.subgraph({'user': [1, 2]})
    >>> print(sub_g)
    Graph(num_nodes={'user': 2, 'game': 0},
          num_edges={('user', 'plays', 'game'): 0, ('user', 'follows', 'user'): 2},
          metagraph=[('user', 'game'), ('user', 'user')])

    Get subgraphs using boolean mask tensor.

    >>> sub_g = g.subgraph({'user': th.tensor([False, True, True])})
    >>> print(sub_g)
    Graph(num_nodes={'user': 2, 'game': 0},
          num_edges={('user', 'plays', 'game'): 0, ('user', 'follows', 'user'): 2},
          metagraph=[('user', 'game'), ('user', 'user')])

    Get the original node/edge indices.

    >>> sub_g['follows'].ndata[dgl.NID] # Get the node indices in the raw graph
    tensor([1, 2])
    >>> sub_g['follows'].edata[dgl.EID] # Get the edge indices in the raw graph
    tensor([1, 2])

    Get the copied node features.

    >>> sub_g.nodes['user'].data['h']
    tensor([[1.],
            [2.]])
    >>> sub_g.nodes['user'].data['h'] += 1
    >>> g.nodes['user'].data['h']          # Features are not shared.
    tensor([[0.],
            [1.],
            [2.]])

    See Also
    --------
    edge_subgraph
    """
    if graph.is_block:
        raise DGLError('Extracting subgraph from a block graph is not allowed.')
    if not isinstance(nodes, Mapping):
        assert len(graph.ntypes) == 1, \
            'need a dict of node type and IDs for graph with multiple node types'
        nodes = {graph.ntypes[0]: nodes}

    def _process_nodes(ntype, v):
        if F.is_tensor(v) and F.dtype(v) == F.bool:
            return F.astype(F.nonzero_1d(F.copy_to(v, graph.device)), graph.idtype)
        else:
            return utils.prepare_tensor(graph, v, 'nodes["{}"]'.format(ntype))
    induced_nodes = [_process_nodes(ntype, nodes.get(ntype, [])) for ntype in graph.ntypes]
    sgi = graph._graph.node_subgraph(induced_nodes)
    induced_edges = sgi.induced_edges
    return _create_hetero_subgraph(graph, sgi, induced_nodes, induced_edges)

DGLHeteroGraph.subgraph = node_subgraph

def edge_subgraph(graph, edges, preserve_nodes=False):
    """Return the subgraph induced on given edges.

    The metagraph of the returned subgraph is the same as the parent graph.

    Features are copied from the original graph.

    Parameters
    ----------
    graph : DGLGraph
        The graph to extract subgraphs from.
    edges : dict[(str, str, str), Tensor]
        A dictionary mapping edge types to edge ID array for constructing
        subgraph. All edges must exist in the subgraph.

        The edge types are characterized by triplets of
        ``(src type, etype, dst type)``.

        If the graph only has one edge type, one can just specify a list,
        tensor, or any iterable of edge IDs intead.

        The edge ID array can be either an interger tensor or a bool tensor.
        When a bool tensor is used, it is automatically converted to
        an interger tensor using the semantic of np.where(edges_idx == True).

        Note: When using bool tensor, only backend (torch, tensorflow, mxnet)
        tensors are supported.

    preserve_nodes : bool
        Whether to preserve all nodes or not. If false, all nodes
        without edges will be removed. (Default: False)

    Returns
    -------
    G : DGLGraph
        The subgraph.

        The nodes and edges are relabeled using consecutive integers from 0.

        One can retrieve the mapping from subgraph node/edge ID to parent
        node/edge ID via ``dgl.NID`` and ``dgl.EID`` node/edge features of the
        subgraph.

    Examples
    --------
    The following example uses PyTorch backend.

    Instantiate a heterograph.

    >>> g = dgl.heterograph({
    ...     ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 2, 1]),
    ...     ('user', 'follows', 'user'): ([0, 1, 1], [1, 2, 2])})
    >>> # Set edge features
    >>> g.edges['follows'].data['h'] = torch.tensor([[0.], [1.], [2.]])

    Get subgraphs.

    >>> g.edge_subgraph({('user', 'follows', 'user'): [5, 6]})
    Traceback (most recent call last):
        ...
    dgl._ffi.base.DGLError: ...
    >>> sub_g = g.edge_subgraph({('user', 'follows', 'user'): [1, 2],
    >>>                          ('user', 'plays', 'game'): [2]})
    >>> print(sub_g)
    Graph(num_nodes={'user': 2, 'game': 1},
          num_edges={('user', 'plays', 'game'): 1, ('user', 'follows', 'user'): 2},
          metagraph=[('user', 'game'), ('user', 'user')])

    Get subgraphs using boolean mask tensor.
    >>> sub_g = g.edge_subgraph({('user', 'follows', 'user'): th.tensor([False, True, True]),
    >>>                   ('user', 'plays', 'game'): th.tensor([False, False, True, False])})
    >>> sub_g
    Graph(num_nodes={'user': 2, 'game': 1},
        num_edges={('user', 'plays', 'game'): 1, ('user', 'follows', 'user'): 2},
        metagraph=[('user', 'game'), ('user', 'user')])

    Get the original node/edge indices.

    >>> sub_g['follows'].ndata[dgl.NID] # Get the node indices in the raw graph
    tensor([1, 2])
    >>> sub_g['plays'].edata[dgl.EID]   # Get the edge indices in the raw graph
    tensor([2])

    Get the copied node features.

    >>> sub_g.edges['follows'].data['h']
    tensor([[1.],
            [2.]])
    >>> sub_g.edges['follows'].data['h'] += 1
    >>> g.edges['follows'].data['h']          # Features are not shared.
    tensor([[0.],
            [1.],
            [2.]])

    See Also
    --------
    subgraph
    """
    if graph.is_block:
        raise DGLError('Extracting subgraph from a block graph is not allowed.')
    if not isinstance(edges, Mapping):
        assert len(graph.canonical_etypes) == 1, \
            'need a dict of edge type and IDs for graph with multiple edge types'
        edges = {graph.canonical_etypes[0]: edges}

    def _process_edges(etype, e):
        if F.is_tensor(e) and F.dtype(e) == F.bool:
            return F.astype(F.nonzero_1d(F.copy_to(e, graph.device)), graph.idtype)
        else:
            return utils.prepare_tensor(graph, e, 'edges["{}"]'.format(etype))

    edges = {graph.to_canonical_etype(etype): e for etype, e in edges.items()}
    induced_edges = [
        _process_edges(cetype, edges.get(cetype, []))
        for cetype in graph.canonical_etypes]
    sgi = graph._graph.edge_subgraph(induced_edges, preserve_nodes)
    induced_nodes = sgi.induced_nodes
    return _create_hetero_subgraph(graph, sgi, induced_nodes, induced_edges)

DGLHeteroGraph.edge_subgraph = edge_subgraph

def in_subgraph(g, nodes):
    """Return the subgraph induced on the inbound edges of all edge types of the
    given nodes.

    All the nodes are preserved regardless of whether they have an edge or not.

    The metagraph of the returned subgraph is the same as the parent graph.

    Features are copied from the original graph.

    Parameters
    ----------
    g : DGLGraph
        Full graph structure.
    nodes : tensor or dict
        Node ids to sample neighbors from. The allowed types
        are dictionary of node types to node id tensors, or simply node id tensor if
        the given graph g has only one type of nodes.

    Returns
    -------
    DGLGraph
        The subgraph.

        One can retrieve the mapping from subgraph edge ID to parent
        edge ID via ``dgl.EID`` edge features of the subgraph.

    Examples
    --------
    The following example uses PyTorch backend.

    Instantiate a heterograph.

    >>> g = dgl.heterograph({
    ...     ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 2, 1]),
    ...     ('user', 'follows', 'user'): ([0, 1, 1], [1, 2, 2])})
    >>> # Set edge features
    >>> g.edges['follows'].data['h'] = torch.tensor([[0.], [1.], [2.]])

    Get subgraphs.

    >>> sub_g = g.in_subgraph({'user': [2], 'game': [2]})
    >>> print(sub_g)
    Graph(num_nodes={'game': 3, 'user': 3},
          num_edges={('user', 'plays', 'game'): 1, ('user', 'follows', 'user'): 2},
          metagraph=[('user', 'game', 'plays'), ('user', 'user', 'follows')])

    Get the original node/edge indices.

    >>> sub_g.edges['plays'].data[dgl.EID]
    tensor([2])
    >>> sub_g.edges['follows'].data[dgl.EID]
    tensor([1, 2])

    Get the copied edge features.

    >>> sub_g.edges['follows'].data['h']
    tensor([[1.],
            [2.]])
    >>> sub_g.edges['follows'].data['h'] += 1
    >>> g.edges['follows'].data['h']          # Features are not shared.
    tensor([[0.],
            [1.],
            [2.]])

    See also
    --------
    out_subgraph
    """
    if g.is_block:
        raise DGLError('Extracting subgraph of a block graph is not allowed.')
    if not isinstance(nodes, dict):
        if len(g.ntypes) > 1:
            raise DGLError("Must specify node type when the graph is not homogeneous.")
        nodes = {g.ntypes[0] : nodes}
    nodes = utils.prepare_tensor_dict(g, nodes, 'nodes')
    nodes_all_types = []
    for ntype in g.ntypes:
        if ntype in nodes:
            nodes_all_types.append(F.to_dgl_nd(nodes[ntype]))
        else:
            nodes_all_types.append(nd.NULL[g._idtype_str])

    sgi = _CAPI_DGLInSubgraph(g._graph, nodes_all_types)
    induced_edges = sgi.induced_edges
    return _create_hetero_subgraph(g, sgi, None, induced_edges)

DGLHeteroGraph.in_subgraph = in_subgraph

def out_subgraph(g, nodes):
    """Return the subgraph induced on the outbound edges of all edge types of the
    given nodes.

    All the nodes are preserved regardless of whether they have an edge or not.

    The metagraph of the returned subgraph is the same as the parent graph.

    Features are copied from the original graph.

    Parameters
    ----------
    g : DGLGraph
        Full graph structure.
    nodes : tensor or dict
        Node ids to sample neighbors from. The allowed types
        are dictionary of node types to node id tensors, or simply node id tensor if
        the given graph g has only one type of nodes.

    Returns
    -------
    DGLGraph
        The subgraph.

        One can retrieve the mapping from subgraph edge ID to parent
        edge ID via ``dgl.EID`` edge features of the subgraph.

    Examples
    --------
    The following example uses PyTorch backend.

    Instantiate a heterograph.

    >>> g = dgl.heterograph({
    ...     ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 2, 1]),
    ...     ('user', 'follows', 'user'): ([0, 1, 1], [1, 2, 2])})
    >>> # Set edge features
    >>> g.edges['follows'].data['h'] = torch.tensor([[0.], [1.], [2.]])

    Get subgraphs.

    >>> sub_g = g.out_subgraph({'user': [1]})
    >>> print(sub_g)
    Graph(num_nodes={'game': 3, 'user': 3},
          num_edges={('user', 'plays', 'game'): 2, ('user', 'follows', 'user'): 2},
          metagraph=[('user', 'game', 'plays'), ('user', 'user', 'follows')])

    Get the original node/edge indices.

    >>> sub_g.edges['plays'].data[dgl.EID]
    tensor([1, 2])
    >>> sub_g.edges['follows'].data[dgl.EID]
    tensor([1, 2])

    Get the copied edge features.

    >>> sub_g.edges['follows'].data['h']
    tensor([[1.],
            [2.]])
    >>> sub_g.edges['follows'].data['h'] += 1
    >>> g.edges['follows'].data['h']          # Features are not shared.
    tensor([[0.],
            [1.],
            [2.]])

    See also
    --------
    in_subgraph
    """
    if g.is_block:
        raise DGLError('Extracting subgraph of a block graph is not allowed.')
    if not isinstance(nodes, dict):
        if len(g.ntypes) > 1:
            raise DGLError("Must specify node type when the graph is not homogeneous.")
        nodes = {g.ntypes[0] : nodes}
    nodes = utils.prepare_tensor_dict(g, nodes, 'nodes')
    nodes_all_types = []
    for ntype in g.ntypes:
        if ntype in nodes:
            nodes_all_types.append(F.to_dgl_nd(nodes[ntype]))
        else:
            nodes_all_types.append(nd.NULL[g._idtype_str])

    sgi = _CAPI_DGLOutSubgraph(g._graph, nodes_all_types)
    induced_edges = sgi.induced_edges
    return _create_hetero_subgraph(g, sgi, None, induced_edges)

DGLHeteroGraph.out_subgraph = out_subgraph

def node_type_subgraph(graph, ntypes):
    """Return the subgraph induced on given node types.

    The metagraph of the returned subgraph is the subgraph of the original
    metagraph induced from the node types.

    Features are shared with the original graph.

    Parameters
    ----------
    graph : DGLGraph
        The graph to extract subgraphs from.
    ntypes : list[str]
        The node types

    Returns
    -------
    G : DGLGraph
        The subgraph.

    Examples
    --------
    The following example uses PyTorch backend.

    Instantiate a heterograph.

    >>> g = dgl.heterograph({
    ...     ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 2, 1]),
    ...     ('user', 'follows', 'user'): ([0, 1, 1], [1, 2, 2])})
    >>> # Set node features
    >>> g.nodes['user'].data['h'] = torch.tensor([[0.], [1.], [2.]])

    Get subgraphs.

    >>> sub_g = g.node_type_subgraph(['user'])
    >>> print(sub_g)
    Graph(num_nodes=3, num_edges=3,
          ndata_schemes={'h': Scheme(shape=(1,), dtype=torch.float32)}
          edata_schemes={})

    Get the shared node features.

    >>> sub_g.nodes['user'].data['h']
    tensor([[0.],
            [1.],
            [2.]])
    >>> sub_g.nodes['user'].data['h'] += 1
    >>> g.nodes['user'].data['h']          # Features are shared.
    tensor([[1.],
            [2.],
            [3.]])

    See Also
    --------
    edge_type_subgraph
    """
    ntid = [graph.get_ntype_id(ntype) for ntype in ntypes]
    stids, dtids, etids = graph._graph.metagraph.edges('eid')
    stids, dtids, etids = stids.tonumpy(), dtids.tonumpy(), etids.tonumpy()
    etypes = []
    for stid, dtid, etid in zip(stids, dtids, etids):
        if stid in ntid and dtid in ntid:
            etypes.append(graph.canonical_etypes[etid])
    return edge_type_subgraph(graph, etypes)

DGLHeteroGraph.node_type_subgraph = node_type_subgraph

def edge_type_subgraph(graph, etypes):
    """Return the subgraph induced on given edge types.

    The metagraph of the returned subgraph is the subgraph of the original metagraph
    induced from the edge types.

    Features are shared with the original graph.

    Parameters
    ----------
    graph : DGLGraph
        The graph to extract subgraphs from.
    etypes : list[str or tuple]
        The edge types

    Returns
    -------
    G : DGLGraph
        The subgraph.

    Examples
    --------
    The following example uses PyTorch backend.

    Instantiate a heterograph.

    >>> g = dgl.heterograph({
    ...     ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 2, 1]),
    ...     ('user', 'follows', 'user'): ([0, 1, 1], [1, 2, 2])})
    >>> # Set edge features
    >>> g.edges['follows'].data['h'] = torch.tensor([[0.], [1.], [2.]])

    Get subgraphs.

    >>> sub_g = g.edge_type_subgraph(['follows'])
    >>> print(sub_g)
    Graph(num_nodes=3, num_edges=3,
          ndata_schemes={}
          edata_schemes={'h': Scheme(shape=(1,), dtype=torch.float32)})

    Get the shared edge features.

    >>> sub_g.edges['follows'].data['h']
    tensor([[0.],
            [1.],
            [2.]])
    >>> sub_g.edges['follows'].data['h'] += 1
    >>> g.edges['follows'].data['h']          # Features are shared.
    tensor([[1.],
            [2.],
            [3.]])

    See Also
    --------
    node_type_subgraph
    """
    etype_ids = [graph.get_etype_id(etype) for etype in etypes]
    # meta graph is homograph, still using int64
    meta_src, meta_dst, _ = graph._graph.metagraph.find_edges(utils.toindex(etype_ids, "int64"))
    rel_graphs = [graph._graph.get_relation_graph(i) for i in etype_ids]
    meta_src = meta_src.tonumpy()
    meta_dst = meta_dst.tonumpy()
    ntypes_invmap = {n: i for i, n in enumerate(set(meta_src) | set(meta_dst))}
    mapped_meta_src = [ntypes_invmap[v] for v in meta_src]
    mapped_meta_dst = [ntypes_invmap[v] for v in meta_dst]
    node_frames = [graph._node_frames[i] for i in ntypes_invmap]
    edge_frames = [graph._edge_frames[i] for i in etype_ids]
    induced_ntypes = [graph._ntypes[i] for i in ntypes_invmap]
    induced_etypes = [graph._etypes[i] for i in etype_ids]   # get the "name" of edge type
    num_nodes_per_induced_type = [graph.number_of_nodes(ntype) for ntype in induced_ntypes]

    metagraph = graph_index.from_edge_list((mapped_meta_src, mapped_meta_dst), True)
    # num_nodes_per_type should be int64
    hgidx = heterograph_index.create_heterograph_from_relations(
        metagraph, rel_graphs, utils.toindex(num_nodes_per_induced_type, "int64"))
    hg = DGLHeteroGraph(hgidx, induced_ntypes, induced_etypes, node_frames, edge_frames)
    return hg

DGLHeteroGraph.edge_type_subgraph = edge_type_subgraph

#################### Internal functions ####################

def _create_hetero_subgraph(parent, sgi, induced_nodes, induced_edges):
    """Internal function to create a subgraph.

    Parameters
    ----------
    parent : DGLGraph
        The parent DGLGraph.
    sgi : HeteroSubgraphIndex
        Subgraph object returned by CAPI.
    induced_nodes : list[Tensor] or None
        Induced node IDs. Will store it as the dgl.NID ndata unless it
        is None, which means the induced node IDs are the same as the parent node IDs.
    induced_edges : list[Tensor] or None
        Induced edge IDs. Will store it as the dgl.EID ndata unless it
        is None, which means the induced edge IDs are the same as the parent edge IDs.

    Returns
    -------
    DGLGraph
        Graph
    """
    node_frames = utils.extract_node_subframes(parent, induced_nodes)
    edge_frames = utils.extract_edge_subframes(parent, induced_edges)
    hsg = DGLHeteroGraph(sgi.graph, parent.ntypes, parent.etypes)
    utils.set_new_frames(hsg, node_frames=node_frames, edge_frames=edge_frames)
    return hsg

_init_api("dgl.subgraph")
