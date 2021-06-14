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

def node_subgraph(graph, nodes, store_ids=True):
    """Return a subgraph induced on the given nodes.

    A node-induced subgraph is a subset of the nodes of a graph together with
    any edges whose endpoints are both in this subset. In addition to extracting
    the subgraph, DGL conducts the following:

    * Relabel the extracted nodes to IDs starting from zero.

    * Copy the features of the extracted nodes and edges to the resulting graph.
      The copy is *lazy* and incurs data movement only when needed.

    If the graph is heterogeneous, DGL extracts a subgraph per relation and composes
    them as the resulting graph. Thus, the resulting graph has the same set of relations
    as the input one.

    Parameters
    ----------
    graph : DGLGraph
        The graph to extract subgraphs from.
    nodes : nodes or dict[str, nodes]
        The nodes to form the subgraph. The allowed nodes formats are:

        * Int Tensor: Each element is a node ID. The tensor must have the same device type
          and ID data type as the graph's.
        * iterable[int]: Each element is a node ID.
        * Bool Tensor: Each :math:`i^{th}` element is a bool flag indicating whether
          node :math:`i` is in the subgraph.

        If the graph is homogeneous, one can directly pass the above formats.
        Otherwise, the argument must be a dictionary with keys being node types
        and values being the nodes.
    store_ids : bool, optional
        If True, it will store the raw IDs of the extracted nodes and edges in the ``ndata``
        and ``edata`` of the resulting graph under name ``dgl.NID`` and ``dgl.EID``,
        respectively.

    Returns
    -------
    G : DGLGraph
        The subgraph.

    Notes
    -----

    This function discards the batch information. Please use
    :func:`dgl.DGLGraph.set_batch_num_nodes`
    and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
    to maintain the information.

    Examples
    --------
    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch

    Extract a subgraph from a homogeneous graph.

    >>> g = dgl.graph(([0, 1, 2, 3, 4], [1, 2, 3, 4, 0]))  # 5-node cycle
    >>> sg = dgl.node_subgraph(g, [0, 1, 4])
    >>> sg
    Graph(num_nodes=3, num_edges=2,
          ndata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)})
    >>> sg.edges()
    (tensor([0, 2]), tensor([1, 0]))
    >>> sg.ndata[dgl.NID]  # original node IDs
    tensor([0, 1, 4])
    >>> sg.edata[dgl.EID]  # original edge IDs
    tensor([0, 4])

    Specify nodes using a boolean mask.

    >>> nodes = torch.tensor([True, True, False, False, True])  # choose nodes [0, 1, 4]
    >>> dgl.node_subgraph(g, nodes)
    Graph(num_nodes=3, num_edges=2,
          ndata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)})

    The resulting subgraph also copies features from the parent graph.

    >>> g.ndata['x'] = torch.arange(10).view(5, 2)
    >>> sg = dgl.node_subgraph(g, [0, 1, 4])
    >>> sg
    Graph(num_nodes=3, num_edges=2,
          ndata_schemes={'x': Scheme(shape=(2,), dtype=torch.int64),
                         '_ID': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)})
    >>> sg.ndata['x']
    tensor([[0, 1],
            [2, 3],
            [8, 9]])

    Extract a subgraph from a hetergeneous graph.

    >>> g = dgl.heterograph({
    >>>     ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 2, 1]),
    >>>     ('user', 'follows', 'user'): ([0, 1, 1], [1, 2, 2])
    >>> })
    >>> sub_g = dgl.node_subgraph(g, {'user': [1, 2]})
    >>> sub_g
    Graph(num_nodes={'user': 2, 'game': 0},
          num_edges={('user', 'plays', 'game'): 0, ('user', 'follows', 'user'): 2},
          metagraph=[('user', 'game'), ('user', 'user')])

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

    induced_nodes = []
    for ntype in graph.ntypes:
        nids = nodes.get(ntype, F.copy_to(F.tensor([], graph.idtype), graph.device))
        induced_nodes.append(_process_nodes(ntype, nids))
    sgi = graph._graph.node_subgraph(induced_nodes)
    induced_edges = sgi.induced_edges
    return _create_hetero_subgraph(graph, sgi, induced_nodes, induced_edges, store_ids)

DGLHeteroGraph.subgraph = utils.alias_func(node_subgraph)

def edge_subgraph(graph, edges, preserve_nodes=False, store_ids=True):
    """Return a subgraph induced on the given edges.

    An edge-induced subgraph is equivalent to creating a new graph
    with the same number of nodes using the given edges.  In addition to extracting
    the subgraph, DGL conducts the following:

    * Relabel the incident nodes to IDs starting from zero. Isolated nodes are removed.

    * Copy the features of the extracted nodes and edges to the resulting graph.
      The copy is *lazy* and incurs data movement only when needed.

    If the graph is heterogeneous, DGL extracts a subgraph per relation and composes
    them as the resulting graph. Thus, the resulting graph has the same set of relations
    as the input one.

    Parameters
    ----------
    graph : DGLGraph
        The graph to extract the subgraph from.
    edges : dict[(str, str, str), edges]
        The edges to form the subgraph. The allowed edges formats are:

        * Int Tensor: Each element is an edge ID. The tensor must have the same device type
          and ID data type as the graph's.
        * iterable[int]: Each element is an edge ID.
        * Bool Tensor: Each :math:`i^{th}` element is a bool flag indicating whether
          edge :math:`i` is in the subgraph.

        If the graph is homogeneous, one can directly pass the above formats.
        Otherwise, the argument must be a dictionary with keys being edge types
        and values being the edge IDs.
    preserve_nodes : bool, optional
        If True, do not relabel the incident nodes and remove the isolated nodes
        in the extracted subgraph. (Default: False)
    store_ids : bool, optional
        If True, it will store the IDs of the extracted nodes and edges in the ``ndata``
        and ``edata`` of the resulting graph under name ``dgl.NID`` and ``dgl.EID``,
        respectively.

    Returns
    -------
    G : DGLGraph
        The subgraph.

    Notes
    -----

    This function discards the batch information. Please use
    :func:`dgl.DGLGraph.set_batch_num_nodes`
    and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
    to maintain the information.

    Examples
    --------
    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch

    Extract a subgraph from a homogeneous graph.

    >>> g = dgl.graph(([0, 1, 2, 3, 4], [1, 2, 3, 4, 0]))  # 5-node cycle
    >>> sg = dgl.edge_subgraph(g, [0, 4])
    >>> sg
    Graph(num_nodes=3, num_edges=2,
          ndata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)})
    >>> sg.edges()
    (tensor([0, 1]), tensor([2, 0]))
    >>> sg.ndata[dgl.NID]  # original node IDs
    tensor([0, 4, 1])
    >>> sg.edata[dgl.EID]  # original edge IDs
    tensor([0, 4])

    Extract a subgraph without node relabeling.

    >>> sg = dgl.edge_subgraph(g, [0, 4], preserve_nodes=True)
    >>> sg
    Graph(num_nodes=5, num_edges=2,
          ndata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)})
    >>> sg.edges()
    (tensor([0, 4]), tensor([1, 0]))

    Specify edges using a boolean mask.

    >>> nodes = torch.tensor([True, False, False, False, True])  # choose edges [0, 4]
    >>> dgl.edge_subgraph(g, nodes)
    Graph(num_nodes=3, num_edges=2,
          ndata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)})

    The resulting subgraph also copies features from the parent graph.

    >>> g.ndata['x'] = torch.arange(10).view(5, 2)
    >>> sg = dgl.edge_subgraph(g, [0, 4])
    >>> sg
    Graph(num_nodes=3, num_edges=2,
          ndata_schemes={'x': Scheme(shape=(2,), dtype=torch.int64),
                         '_ID': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)})
    >>> sg.ndata[dgl.NID]
    tensor([0, 4, 1])
    >>> sg.ndata['x']
    tensor([[0, 1],
            [8, 9],
            [2, 3]])

    Extract a subgraph from a hetergeneous graph.

    >>> g = dgl.heterograph({
    >>>     ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 2, 1]),
    >>>     ('user', 'follows', 'user'): ([0, 1, 1], [1, 2, 2])
    >>> })
    >>> sub_g = dgl.edge_subgraph(g, {('user', 'follows', 'user'): [1, 2],
    ...                               ('user', 'plays', 'game'): [2]})
    >>> print(sub_g)
    Graph(num_nodes={'user': 2, 'game': 1},
          num_edges={('user', 'plays', 'game'): 1, ('user', 'follows', 'user'): 2},
          metagraph=[('user', 'game'), ('user', 'user')])

    See Also
    --------
    node_subgraph
    """
    if graph.is_block and not preserve_nodes:
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
    induced_edges = []
    for cetype in graph.canonical_etypes:
        eids = edges.get(cetype, F.copy_to(F.tensor([], graph.idtype), graph.device))
        induced_edges.append(_process_edges(cetype, eids))
    sgi = graph._graph.edge_subgraph(induced_edges, preserve_nodes)
    induced_nodes = sgi.induced_nodes
    return _create_hetero_subgraph(graph, sgi, induced_nodes, induced_edges, store_ids)

DGLHeteroGraph.edge_subgraph = utils.alias_func(edge_subgraph)

def in_subgraph(g, nodes):
    """Return the subgraph induced on the inbound edges of all the edge types of the
    given nodes.

    An edge-induced subgraph is equivalent to creating a new graph
    with the same number of nodes using the given edges.  In addition to extracting
    the subgraph, DGL conducts the following:

    * Copy the features of the extracted nodes and edges to the resulting graph.
      The copy is *lazy* and incurs data movement only when needed.

    * Store the IDs of the extracted edges in the ``edata``
      of the resulting graph under name ``dgl.EID``.

    If the graph is heterogeneous, DGL extracts a subgraph per relation and composes
    them as the resulting graph. Thus, the resulting graph has the same set of relations
    as the input one.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    nodes : nodes or dict[str, nodes]
        The nodes to form the subgraph. The allowed nodes formats are:

        * Int Tensor: Each element is an ID. The tensor must have the same device type
          and ID data type as the graph's.
        * iterable[int]: Each element is an ID.

        If the graph is homogeneous, one can directly pass the above formats.
        Otherwise, the argument must be a dictionary with keys being node types
        and values being the nodes.

    Returns
    -------
    DGLGraph
        The subgraph.

    Notes
    -----

    This function discards the batch information. Please use
    :func:`dgl.DGLGraph.set_batch_num_nodes`
    and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
    to maintain the information.

    Examples
    --------
    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch

    Extract a subgraph from a homogeneous graph.

    >>> g = dgl.graph(([0, 1, 2, 3, 4], [1, 2, 3, 4, 0]))  # 5-node cycle
    >>> g.edata['w'] = torch.arange(10).view(5, 2)
    >>> sg = dgl.in_subgraph(g, [2, 0])
    >>> sg
    Graph(num_nodes=5, num_edges=2,
          ndata_schemes={}
          edata_schemes={'w': Scheme(shape=(2,), dtype=torch.int64),
                         '_ID': Scheme(shape=(), dtype=torch.int64)})
    >>> sg.edges()
    (tensor([1, 4]), tensor([2, 0]))
    >>> sg.edata[dgl.EID]  # original edge IDs
    tensor([1, 4])
    >>> sg.edata['w']  # also extract the features
    tensor([[2, 3],
            [8, 9]])

    Extract a subgraph from a heterogeneous graph.

    >>> g = dgl.heterograph({
    ...     ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 2, 1]),
    ...     ('user', 'follows', 'user'): ([0, 1, 1], [1, 2, 2])})
    >>> sub_g = g.in_subgraph({'user': [2], 'game': [2]})
    >>> sub_g
    Graph(num_nodes={'game': 3, 'user': 3},
          num_edges={('user', 'plays', 'game'): 1, ('user', 'follows', 'user'): 2},
          metagraph=[('user', 'game', 'plays'), ('user', 'user', 'follows')])

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

DGLHeteroGraph.in_subgraph = utils.alias_func(in_subgraph)

def out_subgraph(g, nodes):
    """Return the subgraph induced on the out-bound edges of all the edge types of the
    given nodes.

    An edge-induced subgraph is equivalent to creating a new graph
    with the same number of nodes using the given edges.  In addition to extracting
    the subgraph, DGL conducts the following:

    * Copy the features of the extracted nodes and edges to the resulting graph.
      The copy is *lazy* and incurs data movement only when needed.

    * Store the IDs of the extracted edges in the ``edata``
      of the resulting graph under name ``dgl.EID``.

    If the graph is heterogeneous, DGL extracts a subgraph per relation and composes
    them as the resulting graph. Thus, the resulting graph has the same set of relations
    as the input one.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    nodes : nodes or dict[str, nodes]
        The nodes to form the subgraph. The allowed nodes formats are:

        * Int Tensor: Each element is a node ID. The tensor must have the same device type
          and ID data type as the graph's.
        * iterable[int]: Each element is a node ID.

        If the graph is homogeneous, one can directly pass the above formats.
        Otherwise, the argument must be a dictionary with keys being node types
        and values being the nodes.

    Returns
    -------
    DGLGraph
        The subgraph.

    Notes
    -----

    This function discards the batch information. Please use
    :func:`dgl.DGLGraph.set_batch_num_nodes`
    and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
    to maintain the information.

    Examples
    --------
    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch

    Extract a subgraph from a homogeneous graph.

    >>> g = dgl.graph(([0, 1, 2, 3, 4], [1, 2, 3, 4, 0]))  # 5-node cycle
    >>> g.edata['w'] = torch.arange(10).view(5, 2)
    >>> sg = dgl.out_subgraph(g, [2, 0])
    >>> sg
    Graph(num_nodes=5, num_edges=2,
          ndata_schemes={}
          edata_schemes={'w': Scheme(shape=(2,), dtype=torch.int64),
                         '_ID': Scheme(shape=(), dtype=torch.int64)})
    >>> sg.edges()
    (tensor([2, 0]), tensor([3, 1]))
    >>> sg.edata[dgl.EID]  # original edge IDs
    tensor([2, 0])
    >>> sg.edata['w']  # also extract the features
    tensor([[4, 5],
            [0, 1]])

    Extract a subgraph from a heterogeneous graph.

    >>> g = dgl.heterograph({
    ...     ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 2, 1]),
    ...     ('user', 'follows', 'user'): ([0, 1, 1], [1, 2, 2])})
    >>> sub_g = g.out_subgraph({'user': [1]})
    >>> sub_g
    Graph(num_nodes={'game': 3, 'user': 3},
          num_edges={('user', 'plays', 'game'): 2, ('user', 'follows', 'user'): 2},
          metagraph=[('user', 'game', 'plays'), ('user', 'user', 'follows')])

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

DGLHeteroGraph.out_subgraph = utils.alias_func(out_subgraph)

def node_type_subgraph(graph, ntypes):
    """Return the subgraph induced on given node types.

    A node-type-induced subgraph contains all the nodes of the given subset of
    the node types of a graph and any edges whose endpoints are both in this subset.
    In addition to extracting the subgraph, DGL also copies the features of the
    extracted nodes and edges to the resulting graph.
    The copy is *lazy* and incurs data movement only when needed.

    Parameters
    ----------
    graph : DGLGraph
        The graph to extract subgraphs from.
    ntypes : list[str]
        The type names of the nodes in the subgraph.

    Returns
    -------
    G : DGLGraph
        The subgraph.

    Notes
    -----

    This function discards the batch information. Please use
    :func:`dgl.DGLGraph.set_batch_num_nodes`
    and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
    to maintain the information.

    Examples
    --------
    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch

    Instantiate a heterograph.

    >>> g = dgl.heterograph({
    >>>     ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 2, 1]),
    >>>     ('user', 'follows', 'user'): ([0, 1, 1], [1, 2, 2])
    >>> })
    >>> # Set node features
    >>> g.nodes['user'].data['h'] = torch.tensor([[0.], [1.], [2.]])

    Get subgraphs.

    >>> sub_g = g.node_type_subgraph(['user'])
    >>> print(sub_g)
    Graph(num_nodes=3, num_edges=3,
          ndata_schemes={'h': Scheme(shape=(1,), dtype=torch.float32)}
          edata_schemes={})

    Get the extracted node features.

    >>> sub_g.nodes['user'].data['h']
    tensor([[0.],
            [1.],
            [2.]])

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
    if len(etypes) == 0:
        raise DGLError('There are no edges among nodes of the specified types.')
    return edge_type_subgraph(graph, etypes)

DGLHeteroGraph.node_type_subgraph = utils.alias_func(node_type_subgraph)

def edge_type_subgraph(graph, etypes):
    """Return the subgraph induced on given edge types.

    An edge-type-induced subgraph contains all the edges of the given subset of
    the edge types of a graph. It also contains all nodes of a particular type
    if some nodes of the type are incident to these edges.
    In addition to extracting the subgraph, DGL also copies the features of the
    extracted nodes and edges to the resulting graph.
    The copy is *lazy* and incurs data movement only when needed.

    Parameters
    ----------
    graph : DGLGraph
        The graph to extract subgraphs from.
    etypes : list[str] or list[(str, str, str)]
        The type names of the edges in the subgraph. The allowed type name
        formats are:

        * ``(str, str, str)`` for source node type, edge type and destination node type.
        * or one ``str`` for the edge type name  if the name can uniquely identify a
          triplet format in the graph.

    Returns
    -------
    G : DGLGraph
        The subgraph.

    Notes
    -----

    This function discards the batch information. Please use
    :func:`dgl.DGLGraph.set_batch_num_nodes`
    and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
    to maintain the information.

    Examples
    --------
    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch

    Instantiate a heterograph.

    >>> g = dgl.heterograph({
    >>>     ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 2, 1]),
    >>>     ('user', 'follows', 'user'): ([0, 1, 1], [1, 2, 2])
    >>> })
    >>> # Set edge features
    >>> g.edges['follows'].data['h'] = torch.tensor([[0.], [1.], [2.]])

    Get subgraphs.

    >>> sub_g = g.edge_type_subgraph(['follows'])
    >>> sub_g
    Graph(num_nodes=3, num_edges=3,
          ndata_schemes={}
          edata_schemes={'h': Scheme(shape=(1,), dtype=torch.float32)})

    Get the shared edge features.

    >>> sub_g.edges['follows'].data['h']
    tensor([[0.],
            [1.],
            [2.]])

    See Also
    --------
    node_type_subgraph
    """
    etype_ids = [graph.get_etype_id(etype) for etype in etypes]
    # meta graph is homogeneous graph, still using int64
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

DGLHeteroGraph.edge_type_subgraph = utils.alias_func(edge_type_subgraph)

#################### Internal functions ####################

def _create_hetero_subgraph(parent, sgi, induced_nodes, induced_edges, store_ids=True):
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
    store_ids : bool
        If True, it will store the raw IDs of the extracted nodes and edges in the ``ndata``
        and ``edata`` of the resulting graph under name ``dgl.NID`` and ``dgl.EID``,
        respectively.

    Returns
    -------
    DGLGraph
        Graph
    """
    node_frames = utils.extract_node_subframes(parent, induced_nodes, store_ids)
    edge_frames = utils.extract_edge_subframes(parent, induced_edges, store_ids)
    hsg = DGLHeteroGraph(sgi.graph, parent.ntypes, parent.etypes)
    utils.set_new_frames(hsg, node_frames=node_frames, edge_frames=edge_frames)
    return hsg

_init_api("dgl.subgraph")
