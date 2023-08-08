"""Functions for extracting subgraphs.

The module only contains functions for extracting subgraphs deterministically.
For stochastic subgraph extraction, please see functions under :mod:`dgl.sampling`.
"""
from collections.abc import Mapping

from . import backend as F, graph_index, heterograph_index, utils
from ._ffi.function import _init_api
from .base import DGLError
from .heterograph import DGLGraph
from .utils import context_of, recursive_apply

__all__ = [
    "node_subgraph",
    "edge_subgraph",
    "node_type_subgraph",
    "edge_type_subgraph",
    "in_subgraph",
    "out_subgraph",
    "khop_in_subgraph",
    "khop_out_subgraph",
]


def node_subgraph(
    graph, nodes, *, relabel_nodes=True, store_ids=True, output_device=None
):
    """Return a subgraph induced on the given nodes.

    A node-induced subgraph is a graph with edges whose endpoints are both in the
    specified node set. In addition to extracting the subgraph, DGL also copies
    the features of the extracted nodes and edges to the resulting graph. The copy
    is *lazy* and incurs data movement only when needed.

    If the graph is heterogeneous, DGL extracts a subgraph per relation and composes
    them as the resulting graph. Thus, the resulting graph has the same set of relations
    as the input one.

    Parameters
    ----------
    graph : DGLGraph
        The graph to extract subgraphs from.
    nodes : nodes or dict[str, nodes]
        The nodes to form the subgraph, which cannot have any duplicate value. The result
        will be undefined otherwise. The allowed nodes formats are:

        * Int Tensor: Each element is a node ID. The tensor must have the same device type
          and ID data type as the graph's.
        * iterable[int]: Each element is a node ID.
        * Bool Tensor: Each :math:`i^{th}` element is a bool flag indicating whether
          node :math:`i` is in the subgraph.

        If the graph is homogeneous, one can directly pass the above formats.
        Otherwise, the argument must be a dictionary with keys being node types
        and values being the node IDs in the above formats.
    relabel_nodes : bool, optional
        If True, the extracted subgraph will only have the nodes in the specified node set
        and it will relabel the nodes in order.
    store_ids : bool, optional
        If True, it will store the raw IDs of the extracted edges in the ``edata`` of the
        resulting graph under name ``dgl.EID``; if ``relabel_nodes`` is ``True``, it will
        also store the raw IDs of the specified nodes in the ``ndata`` of the resulting
        graph under name ``dgl.NID``.
    output_device : Framework-specific device context object, optional
        The output device.  Default is the same as the input graph.

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
    Graph(num_nodes={'game': 0, 'user': 2},
          num_edges={('user', 'follows', 'user'): 2, ('user', 'plays', 'game'): 0},
          metagraph=[('user', 'user', 'follows'), ('user', 'game', 'plays')])

    See Also
    --------
    edge_subgraph
    """
    if graph.is_block:
        raise DGLError("Extracting subgraph from a block graph is not allowed.")
    if not isinstance(nodes, Mapping):
        assert (
            len(graph.ntypes) == 1
        ), "need a dict of node type and IDs for graph with multiple node types"
        nodes = {graph.ntypes[0]: nodes}

    def _process_nodes(ntype, v):
        if F.is_tensor(v) and F.dtype(v) == F.bool:
            return F.astype(
                F.nonzero_1d(F.copy_to(v, graph.device)), graph.idtype
            )
        else:
            return utils.prepare_tensor(graph, v, 'nodes["{}"]'.format(ntype))

    nodes = {ntype: _process_nodes(ntype, v) for ntype, v in nodes.items()}
    device = context_of(nodes)

    induced_nodes = [
        nodes.get(ntype, F.copy_to(F.tensor([], graph.idtype), device))
        for ntype in graph.ntypes
    ]
    sgi = graph._graph.node_subgraph(induced_nodes)
    induced_edges = sgi.induced_edges
    if not relabel_nodes:
        sgi = graph._graph.edge_subgraph(induced_edges, True)
    # (BarclayII) should not write induced_nodes = sgi.induced_nodes due to the same
    # bug in #1453.
    induced_nodes_or_device = induced_nodes if relabel_nodes else device
    subg = _create_hetero_subgraph(
        graph, sgi, induced_nodes_or_device, induced_edges, store_ids=store_ids
    )
    return subg if output_device is None else subg.to(output_device)


DGLGraph.subgraph = utils.alias_func(node_subgraph)


def edge_subgraph(
    graph, edges, *, relabel_nodes=True, store_ids=True, output_device=None
):
    """Return a subgraph induced on the given edges.

    An edge-induced subgraph is equivalent to creating a new graph using the given
    edges. In addition to extracting the subgraph, DGL also copies the features
    of the extracted nodes and edges to the resulting graph. The copy is *lazy*
    and incurs data movement only when needed.

    If the graph is heterogeneous, DGL extracts a subgraph per relation and composes
    them as the resulting graph. Thus, the resulting graph has the same set of relations
    as the input one.

    Parameters
    ----------
    graph : DGLGraph
        The graph to extract the subgraph from.
    edges : edges or dict[(str, str, str), edges]
        The edges to form the subgraph. The allowed edges formats are:

        * Int Tensor: Each element is an edge ID. The tensor must have the same device type
          and ID data type as the graph's.
        * iterable[int]: Each element is an edge ID.
        * Bool Tensor: Each :math:`i^{th}` element is a bool flag indicating whether
          edge :math:`i` is in the subgraph.

        If the graph is homogeneous, one can directly pass the above formats.
        Otherwise, the argument must be a dictionary with keys being edge types
        and values being the edge IDs in the above formats.
    relabel_nodes : bool, optional
        If True, it will remove the isolated nodes and relabel the incident nodes in the
        extracted subgraph.
    store_ids : bool, optional
        If True, it will store the raw IDs of the extracted edges in the ``edata`` of the
        resulting graph under name ``dgl.EID``; if ``relabel_nodes`` is ``True``, it will
        also store the raw IDs of the incident nodes in the ``ndata`` of the resulting
        graph under name ``dgl.NID``.
    output_device : Framework-specific device context object, optional
        The output device.  Default is the same as the input graph.

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

    >>> sg = dgl.edge_subgraph(g, [0, 4], relabel_nodes=False)
    >>> sg
    Graph(num_nodes=5, num_edges=2,
          ndata_schemes={}
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
    Graph(num_nodes={'game': 1, user': 2},
          num_edges={('user', 'follows', 'user'): 2, ('user', 'plays', 'game'): 1},
          metagraph=[('user', 'user', 'follows'), ('user', 'game', 'plays')])

    See Also
    --------
    node_subgraph
    """
    if graph.is_block and relabel_nodes:
        raise DGLError("Extracting subgraph from a block graph is not allowed.")
    if not isinstance(edges, Mapping):
        assert (
            len(graph.canonical_etypes) == 1
        ), "need a dict of edge type and IDs for graph with multiple edge types"
        edges = {graph.canonical_etypes[0]: edges}

    def _process_edges(etype, e):
        if F.is_tensor(e) and F.dtype(e) == F.bool:
            return F.astype(
                F.nonzero_1d(F.copy_to(e, graph.device)), graph.idtype
            )
        else:
            return utils.prepare_tensor(graph, e, 'edges["{}"]'.format(etype))

    edges = {graph.to_canonical_etype(etype): e for etype, e in edges.items()}
    edges = {etype: _process_edges(etype, e) for etype, e in edges.items()}
    device = context_of(edges)
    induced_edges = [
        edges.get(cetype, F.copy_to(F.tensor([], graph.idtype), device))
        for cetype in graph.canonical_etypes
    ]

    sgi = graph._graph.edge_subgraph(induced_edges, not relabel_nodes)
    induced_nodes_or_device = sgi.induced_nodes if relabel_nodes else device
    subg = _create_hetero_subgraph(
        graph, sgi, induced_nodes_or_device, induced_edges, store_ids=store_ids
    )
    return subg if output_device is None else subg.to(output_device)


DGLGraph.edge_subgraph = utils.alias_func(edge_subgraph)


def in_subgraph(
    graph, nodes, *, relabel_nodes=False, store_ids=True, output_device=None
):
    """Return the subgraph induced on the inbound edges of all the edge types of the
    given nodes.

    An in subgraph is equivalent to creating a new graph using the incoming edges of the
    given nodes. In addition to extracting the subgraph, DGL also copies the features of
    the extracted nodes and edges to the resulting graph. The copy is *lazy* and incurs
    data movement only when needed.

    If the graph is heterogeneous, DGL extracts a subgraph per relation and composes
    them as the resulting graph. Thus, the resulting graph has the same set of relations
    as the input one.

    Parameters
    ----------
    graph : DGLGraph
        The input graph.
    nodes : nodes or dict[str, nodes]
        The nodes to form the subgraph, which cannot have any duplicate value. The result
        will be undefined otherwise. The allowed nodes formats are:

        * Int Tensor: Each element is a node ID. The tensor must have the same device type
          and ID data type as the graph's.
        * iterable[int]: Each element is a node ID.

        If the graph is homogeneous, one can directly pass the above formats.
        Otherwise, the argument must be a dictionary with keys being node types
        and values being the node IDs in the above formats.
    relabel_nodes : bool, optional
        If True, it will remove the isolated nodes and relabel the rest nodes in the
        extracted subgraph.
    store_ids : bool, optional
        If True, it will store the raw IDs of the extracted edges in the ``edata`` of the
        resulting graph under name ``dgl.EID``; if ``relabel_nodes`` is ``True``, it will
        also store the raw IDs of the extracted nodes in the ``ndata`` of the resulting
        graph under name ``dgl.NID``.
    output_device : Framework-specific device context object, optional
        The output device.  Default is the same as the input graph.

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

    Extract a subgraph with node labeling.

    >>> sg = dgl.in_subgraph(g, [2, 0], relabel_nodes=True)
    >>> sg
    Graph(num_nodes=4, num_edges=2,
          ndata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64}
          edata_schemes={'w': Scheme(shape=(2,), dtype=torch.int64),
                         '_ID': Scheme(shape=(), dtype=torch.int64)})
    >>> sg.edges()
    (tensor([1, 3]), tensor([2, 0]))
    >>> sg.edata[dgl.EID]  # original edge IDs
    tensor([1, 4])
    >>> sg.ndata[dgl.NID]  # original node IDs
    tensor([0, 1, 2, 4])

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
    if graph.is_block:
        raise DGLError("Extracting subgraph of a block graph is not allowed.")
    if not isinstance(nodes, dict):
        if len(graph.ntypes) > 1:
            raise DGLError(
                "Must specify node type when the graph is not homogeneous."
            )
        nodes = {graph.ntypes[0]: nodes}
    nodes = utils.prepare_tensor_dict(graph, nodes, "nodes")
    device = context_of(nodes)
    nodes_all_types = [
        F.to_dgl_nd(
            nodes.get(ntype, F.copy_to(F.tensor([], graph.idtype), device))
        )
        for ntype in graph.ntypes
    ]

    sgi = _CAPI_DGLInSubgraph(graph._graph, nodes_all_types, relabel_nodes)
    induced_nodes_or_device = sgi.induced_nodes if relabel_nodes else device
    induced_edges = sgi.induced_edges
    subg = _create_hetero_subgraph(
        graph, sgi, induced_nodes_or_device, induced_edges, store_ids=store_ids
    )
    return subg if output_device is None else subg.to(output_device)


DGLGraph.in_subgraph = utils.alias_func(in_subgraph)


def out_subgraph(
    graph, nodes, *, relabel_nodes=False, store_ids=True, output_device=None
):
    """Return the subgraph induced on the outbound edges of all the edge types of the
    given nodes.

    An out subgraph is equivalent to creating a new graph using the outcoming edges of
    the given nodes. In addition to extracting the subgraph, DGL also copies the features
    of the extracted nodes and edges to the resulting graph. The copy is *lazy* and incurs
    data movement only when needed.

    If the graph is heterogeneous, DGL extracts a subgraph per relation and composes
    them as the resulting graph. Thus, the resulting graph has the same set of relations
    as the input one.

    Parameters
    ----------
    graph : DGLGraph
        The input graph.
    nodes : nodes or dict[str, nodes]
        The nodes to form the subgraph, which cannot have any duplicate value. The result
        will be undefined otherwise. The allowed nodes formats are:

        * Int Tensor: Each element is a node ID. The tensor must have the same device type
          and ID data type as the graph's.
        * iterable[int]: Each element is a node ID.

        If the graph is homogeneous, one can directly pass the above formats.
        Otherwise, the argument must be a dictionary with keys being node types
        and values being the node IDs in the above formats.
    relabel_nodes : bool, optional
        If True, it will remove the isolated nodes and relabel the rest nodes in the
        extracted subgraph.
    store_ids : bool, optional
        If True, it will store the raw IDs of the extracted edges in the ``edata`` of the
        resulting graph under name ``dgl.EID``; if ``relabel_nodes`` is ``True``, it will
        also store the raw IDs of the extracted nodes in the ``ndata`` of the resulting
        graph under name ``dgl.NID``.
    output_device : Framework-specific device context object, optional
        The output device.  Default is the same as the input graph.

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

    Extract a subgraph with node labeling.

    >>> sg = dgl.out_subgraph(g, [2, 0], relabel_nodes=True)
    >>> sg
    Graph(num_nodes=4, num_edges=2,
          ndata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={'w': Scheme(shape=(2,), dtype=torch.int64),
                         '_ID': Scheme(shape=(), dtype=torch.int64)})
    >>> sg.edges()
    (tensor([2, 0]), tensor([3, 1]))
    >>> sg.edata[dgl.EID]  # original edge IDs
    tensor([2, 0])
    >>> sg.ndata[dgl.NID]  # original node IDs
    tensor([0, 1, 2, 3])

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
    if graph.is_block:
        raise DGLError("Extracting subgraph of a block graph is not allowed.")
    if not isinstance(nodes, dict):
        if len(graph.ntypes) > 1:
            raise DGLError(
                "Must specify node type when the graph is not homogeneous."
            )
        nodes = {graph.ntypes[0]: nodes}
    nodes = utils.prepare_tensor_dict(graph, nodes, "nodes")
    device = context_of(nodes)
    nodes_all_types = [
        F.to_dgl_nd(
            nodes.get(ntype, F.copy_to(F.tensor([], graph.idtype), device))
        )
        for ntype in graph.ntypes
    ]

    sgi = _CAPI_DGLOutSubgraph(graph._graph, nodes_all_types, relabel_nodes)
    induced_nodes_or_device = sgi.induced_nodes if relabel_nodes else device
    induced_edges = sgi.induced_edges
    subg = _create_hetero_subgraph(
        graph, sgi, induced_nodes_or_device, induced_edges, store_ids=store_ids
    )
    return subg if output_device is None else subg.to(output_device)


DGLGraph.out_subgraph = utils.alias_func(out_subgraph)


def khop_in_subgraph(
    graph, nodes, k, *, relabel_nodes=True, store_ids=True, output_device=None
):
    """Return the subgraph induced by k-hop in-neighborhood of the specified node(s).

    We can expand a set of nodes by including the predecessors of them. From a
    specified node set, a k-hop in subgraph is obtained by first repeating the node set
    expansion for k times and then creating a node induced subgraph. In addition to
    extracting the subgraph, DGL also copies the features of the extracted nodes and
    edges to the resulting graph. The copy is *lazy* and incurs data movement only
    when needed.

    If the graph is heterogeneous, DGL extracts a subgraph per relation and composes
    them as the resulting graph. Thus the resulting graph has the same set of relations
    as the input one.

    Parameters
    ----------
    graph : DGLGraph
        The input graph.
    nodes : nodes or dict[str, nodes]
        The starting node(s) to expand, which cannot have any duplicate value. The result
        will be undefined otherwise. The allowed formats are:

        * Int: ID of a single node.
        * Int Tensor: Each element is a node ID. The tensor must have the same device
          type and ID data type as the graph's.
        * iterable[int]: Each element is a node ID.

        If the graph is homogeneous, one can directly pass the above formats.
        Otherwise, the argument must be a dictionary with keys being node types
        and values being the node IDs in the above formats.
    k : int
        The number of hops.
    relabel_nodes : bool, optional
        If True, it will remove the isolated nodes and relabel the rest nodes in the
        extracted subgraph.
    store_ids : bool, optional
        If True, it will store the raw IDs of the extracted edges in the ``edata`` of the
        resulting graph under name ``dgl.EID``; if ``relabel_nodes`` is ``True``, it will
        also store the raw IDs of the extracted nodes in the ``ndata`` of the resulting
        graph under name ``dgl.NID``.
    output_device : Framework-specific device context object, optional
        The output device.  Default is the same as the input graph.

    Returns
    -------
    DGLGraph
        The subgraph.
    Tensor or dict[str, Tensor], optional
        The new IDs of the input :attr:`nodes` after node relabeling. This is returned
        only when :attr:`relabel_nodes` is True. It is in the same form as :attr:`nodes`.

    Notes
    -----

    When k is 1, the result subgraph is different from the one obtained by
    :func:`dgl.in_subgraph`. The 1-hop in subgraph also includes the edges
    among the neighborhood.

    Examples
    --------
    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch

    Extract a two-hop subgraph from a homogeneous graph.

    >>> g = dgl.graph(([1, 1, 2, 3, 4], [0, 2, 0, 4, 2]))
    >>> g.edata['w'] = torch.arange(10).view(5, 2)
    >>> sg, inverse_indices = dgl.khop_in_subgraph(g, 0, k=2)
    >>> sg
    Graph(num_nodes=4, num_edges=4,
          ndata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={'w': Scheme(shape=(2,), dtype=torch.int64),
                         '_ID': Scheme(shape=(), dtype=torch.int64)})
    >>> sg.edges()
    (tensor([1, 1, 2, 3]), tensor([0, 2, 0, 2]))
    >>> sg.edata[dgl.EID]  # original edge IDs
    tensor([0, 1, 2, 4])
    >>> sg.edata['w']  # also extract the features
    tensor([[0, 1],
            [2, 3],
            [4, 5],
            [8, 9]])
    >>> inverse_indices
    tensor([0])

    Extract a subgraph from a heterogeneous graph.

    >>> g = dgl.heterograph({
    ...     ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 2, 1]),
    ...     ('user', 'follows', 'user'): ([0, 1, 1], [1, 2, 2])})
    >>> sg, inverse_indices = dgl.khop_in_subgraph(g, {'game': 0}, k=2)
    >>> sg
    Graph(num_nodes={'game': 1, 'user': 2},
          num_edges={('user', 'follows', 'user'): 1, ('user', 'plays', 'game'): 2},
          metagraph=[('user', 'user', 'follows'), ('user', 'game', 'plays')])
    >>> inverse_indices
    {'game': tensor([0])}

    See also
    --------
    khop_out_subgraph
    """
    if graph.is_block:
        raise DGLError("Extracting subgraph of a block graph is not allowed.")

    is_mapping = isinstance(nodes, Mapping)
    if not is_mapping:
        assert (
            len(graph.ntypes) == 1
        ), "need a dict of node type and IDs for graph with multiple node types"
        nodes = {graph.ntypes[0]: nodes}

    for nty, nty_nodes in nodes.items():
        nodes[nty] = utils.prepare_tensor(
            graph, nty_nodes, 'nodes["{}"]'.format(nty)
        )

    last_hop_nodes = nodes
    k_hop_nodes_ = [last_hop_nodes]
    device = context_of(nodes)
    place_holder = F.copy_to(F.tensor([], dtype=graph.idtype), device)
    for _ in range(k):
        current_hop_nodes = {nty: [] for nty in graph.ntypes}
        for cetype in graph.canonical_etypes:
            srctype, _, dsttype = cetype
            in_nbrs, _ = graph.in_edges(
                last_hop_nodes.get(dsttype, place_holder), etype=cetype
            )
            current_hop_nodes[srctype].append(in_nbrs)
        for nty in graph.ntypes:
            if len(current_hop_nodes[nty]) == 0:
                current_hop_nodes[nty] = place_holder
                continue
            current_hop_nodes[nty] = F.unique(
                F.cat(current_hop_nodes[nty], dim=0)
            )
        k_hop_nodes_.append(current_hop_nodes)
        last_hop_nodes = current_hop_nodes

    k_hop_nodes = dict()
    inverse_indices = dict()
    for nty in graph.ntypes:
        k_hop_nodes[nty], inverse_indices[nty] = F.unique(
            F.cat(
                [
                    hop_nodes.get(nty, place_holder)
                    for hop_nodes in k_hop_nodes_
                ],
                dim=0,
            ),
            return_inverse=True,
        )

    sub_g = node_subgraph(
        graph, k_hop_nodes, relabel_nodes=relabel_nodes, store_ids=store_ids
    )
    if output_device is not None:
        sub_g = sub_g.to(output_device)
    if relabel_nodes:
        if is_mapping:
            seed_inverse_indices = dict()
            for nty in nodes:
                seed_inverse_indices[nty] = F.slice_axis(
                    inverse_indices[nty], axis=0, begin=0, end=len(nodes[nty])
                )
        else:
            seed_inverse_indices = F.slice_axis(
                inverse_indices[nty], axis=0, begin=0, end=len(nodes[nty])
            )
        if output_device is not None:
            seed_inverse_indices = recursive_apply(
                seed_inverse_indices, lambda x: F.copy_to(x, output_device)
            )
        return sub_g, seed_inverse_indices
    else:
        return sub_g


DGLGraph.khop_in_subgraph = utils.alias_func(khop_in_subgraph)


def khop_out_subgraph(
    graph, nodes, k, *, relabel_nodes=True, store_ids=True, output_device=None
):
    """Return the subgraph induced by k-hop out-neighborhood of the specified node(s).

    We can expand a set of nodes by including the successors of them. From a
    specified node set, a k-hop out subgraph is obtained by first repeating the node set
    expansion for k times and then creating a node induced subgraph. In addition to
    extracting the subgraph, DGL also copies the features of the extracted nodes and
    edges to the resulting graph. The copy is *lazy* and incurs data movement only
    when needed.

    If the graph is heterogeneous, DGL extracts a subgraph per relation and composes
    them as the resulting graph. Thus the resulting graph has the same set of relations
    as the input one.

    Parameters
    ----------
    graph : DGLGraph
        The input graph.
    nodes : nodes or dict[str, nodes]
        The starting node(s) to expand, which cannot have any duplicate value. The result
        will be undefined otherwise. The allowed formats are:

        * Int: ID of a single node.
        * Int Tensor: Each element is a node ID. The tensor must have the same device
          type and ID data type as the graph's.
        * iterable[int]: Each element is a node ID.

        If the graph is homogeneous, one can directly pass the above formats.
        Otherwise, the argument must be a dictionary with keys being node types
        and values being the node IDs in the above formats.
    k : int
        The number of hops.
    relabel_nodes : bool, optional
        If True, it will remove the isolated nodes and relabel the rest nodes in the
        extracted subgraph.
    store_ids : bool, optional
        If True, it will store the raw IDs of the extracted edges in the ``edata`` of the
        resulting graph under name ``dgl.EID``; if ``relabel_nodes`` is ``True``, it will
        also store the raw IDs of the extracted nodes in the ``ndata`` of the resulting
        graph under name ``dgl.NID``.
    output_device : Framework-specific device context object, optional
        The output device.  Default is the same as the input graph.

    Returns
    -------
    DGLGraph
        The subgraph.
    Tensor or dict[str, Tensor], optional
        The new IDs of the input :attr:`nodes` after node relabeling. This is returned
        only when :attr:`relabel_nodes` is True. It is in the same form as :attr:`nodes`.

    Notes
    -----

    When k is 1, the result subgraph is different from the one obtained by
    :func:`dgl.out_subgraph`. The 1-hop out subgraph also includes the edges
    among the neighborhood.

    Examples
    --------
    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch

    Extract a two-hop subgraph from a homogeneous graph.

    >>> g = dgl.graph(([0, 2, 0, 4, 2], [1, 1, 2, 3, 4]))
    >>> g.edata['w'] = torch.arange(10).view(5, 2)
    >>> sg, inverse_indices = dgl.khop_out_subgraph(g, 0, k=2)
    >>> sg
    Graph(num_nodes=4, num_edges=4,
          ndata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={'w': Scheme(shape=(2,), dtype=torch.int64),
                         '_ID': Scheme(shape=(), dtype=torch.int64)})
    >>> sg.edges()
    (tensor([0, 0, 2, 2]), tensor([1, 2, 1, 3]))
    >>> sg.edata[dgl.EID]  # original edge IDs
    tensor([0, 2, 1, 4])
    >>> sg.edata['w']  # also extract the features
    tensor([[0, 1],
            [4, 5],
            [2, 3],
            [8, 9]])
    >>> inverse_indices
    tensor([0])

    Extract a subgraph from a heterogeneous graph.

    >>> g = dgl.heterograph({
    ...     ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 2, 1]),
    ...     ('user', 'follows', 'user'): ([0, 1], [1, 3])})
    >>> sg, inverse_indices = dgl.khop_out_subgraph(g, {'user': 0}, k=2)
    >>> sg
    Graph(num_nodes={'game': 2, 'user': 3},
          num_edges={('user', 'follows', 'user'): 2, ('user', 'plays', 'game'): 2},
          metagraph=[('user', 'user', 'follows'), ('user', 'game', 'plays')])
    >>> inverse_indices
    {'user': tensor([0])}

    See also
    --------
    khop_in_subgraph
    """
    if graph.is_block:
        raise DGLError("Extracting subgraph of a block graph is not allowed.")

    is_mapping = isinstance(nodes, Mapping)
    if not is_mapping:
        assert (
            len(graph.ntypes) == 1
        ), "need a dict of node type and IDs for graph with multiple node types"
        nodes = {graph.ntypes[0]: nodes}

    for nty, nty_nodes in nodes.items():
        nodes[nty] = utils.prepare_tensor(
            graph, nty_nodes, 'nodes["{}"]'.format(nty)
        )

    last_hop_nodes = nodes
    k_hop_nodes_ = [last_hop_nodes]
    device = context_of(nodes)
    place_holder = F.copy_to(F.tensor([], dtype=graph.idtype), device)
    for _ in range(k):
        current_hop_nodes = {nty: [] for nty in graph.ntypes}
        for cetype in graph.canonical_etypes:
            srctype, _, dsttype = cetype
            _, out_nbrs = graph.out_edges(
                last_hop_nodes.get(srctype, place_holder), etype=cetype
            )
            current_hop_nodes[dsttype].append(out_nbrs)
        for nty in graph.ntypes:
            if len(current_hop_nodes[nty]) == 0:
                current_hop_nodes[nty] = place_holder
                continue
            current_hop_nodes[nty] = F.unique(
                F.cat(current_hop_nodes[nty], dim=0)
            )
        k_hop_nodes_.append(current_hop_nodes)
        last_hop_nodes = current_hop_nodes

    k_hop_nodes = dict()
    inverse_indices = dict()
    for nty in graph.ntypes:
        k_hop_nodes[nty], inverse_indices[nty] = F.unique(
            F.cat(
                [
                    hop_nodes.get(nty, place_holder)
                    for hop_nodes in k_hop_nodes_
                ],
                dim=0,
            ),
            return_inverse=True,
        )

    sub_g = node_subgraph(
        graph, k_hop_nodes, relabel_nodes=relabel_nodes, store_ids=store_ids
    )
    if output_device is not None:
        sub_g = sub_g.to(output_device)
    if relabel_nodes:
        if is_mapping:
            seed_inverse_indices = dict()
            for nty in nodes:
                seed_inverse_indices[nty] = F.slice_axis(
                    inverse_indices[nty], axis=0, begin=0, end=len(nodes[nty])
                )
        else:
            seed_inverse_indices = F.slice_axis(
                inverse_indices[nty], axis=0, begin=0, end=len(nodes[nty])
            )
        if output_device is not None:
            seed_inverse_indices = recursive_apply(
                seed_inverse_indices, lambda x: F.copy_to(x, output_device)
            )
        return sub_g, seed_inverse_indices
    else:
        return sub_g


DGLGraph.khop_out_subgraph = utils.alias_func(khop_out_subgraph)


def node_type_subgraph(graph, ntypes, output_device=None):
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
    output_device : Framework-specific device context object, optional
        The output device.  Default is the same as the input graph.

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
    stids, dtids, etids = graph._graph.metagraph.edges("eid")
    stids, dtids, etids = stids.tonumpy(), dtids.tonumpy(), etids.tonumpy()
    etypes = []
    for stid, dtid, etid in zip(stids, dtids, etids):
        if stid in ntid and dtid in ntid:
            etypes.append(graph.canonical_etypes[etid])
    if len(etypes) == 0:
        raise DGLError("There are no edges among nodes of the specified types.")
    return edge_type_subgraph(graph, etypes, output_device=output_device)


DGLGraph.node_type_subgraph = utils.alias_func(node_type_subgraph)


def edge_type_subgraph(graph, etypes, output_device=None):
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
    output_device : Framework-specific device context object, optional
        The output device.  Default is the same as the input graph.

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
    meta_src, meta_dst, _ = graph._graph.metagraph.find_edges(
        utils.toindex(etype_ids, "int64")
    )
    rel_graphs = [graph._graph.get_relation_graph(i) for i in etype_ids]
    meta_src = meta_src.tonumpy()
    meta_dst = meta_dst.tonumpy()
    ntypes_invmap = {n: i for i, n in enumerate(set(meta_src) | set(meta_dst))}
    mapped_meta_src = [ntypes_invmap[v] for v in meta_src]
    mapped_meta_dst = [ntypes_invmap[v] for v in meta_dst]
    node_frames = [graph._node_frames[i] for i in ntypes_invmap]
    edge_frames = [graph._edge_frames[i] for i in etype_ids]
    induced_ntypes = [graph._ntypes[i] for i in ntypes_invmap]
    induced_etypes = [
        graph._etypes[i] for i in etype_ids
    ]  # get the "name" of edge type
    num_nodes_per_induced_type = [
        graph.num_nodes(ntype) for ntype in induced_ntypes
    ]

    metagraph = graph_index.from_edge_list(
        (mapped_meta_src, mapped_meta_dst), True
    )
    # num_nodes_per_type should be int64
    hgidx = heterograph_index.create_heterograph_from_relations(
        metagraph,
        rel_graphs,
        utils.toindex(num_nodes_per_induced_type, "int64"),
    )
    hg = DGLGraph(
        hgidx, induced_ntypes, induced_etypes, node_frames, edge_frames
    )
    return hg if output_device is None else hg.to(output_device)


DGLGraph.edge_type_subgraph = utils.alias_func(edge_type_subgraph)

#################### Internal functions ####################


def _create_hetero_subgraph(
    parent,
    sgi,
    induced_nodes_or_device,
    induced_edges_or_device,
    store_ids=True,
):
    """Internal function to create a subgraph.

    Parameters
    ----------
    parent : DGLGraph
        The parent DGLGraph.
    sgi : HeteroSubgraphIndex
        Subgraph object returned by CAPI.
    induced_nodes_or_device : list[Tensor] or device or None
        Induced node IDs or the device. Will store it as the dgl.NID ndata unless it
        is None, which means the induced node IDs are the same as the parent node IDs.
        If a device is given, the features will be copied to the given device.
    induced_edges_or_device : list[Tensor] or device or None
        Induced edge IDs. Will store it as the dgl.EID ndata unless it
        is None, which means the induced edge IDs are the same as the parent edge IDs.
        If a device is given, the features will be copied to the given device.
    store_ids : bool
        If True and induced_nodes is not None, it will store the raw IDs of the extracted
        nodes in the ``ndata`` of the resulting graph under name ``dgl.NID``.
        If True and induced_edges is not None, it will store the raw IDs of the extracted
        edges in the ``edata`` of the resulting graph under name ``dgl.EID``.

    Returns
    -------
    DGLGraph
        Graph
    """
    # (BarclayII) Giving a device argument to induced_nodes_or_device is necessary for
    # UVA subgraphing, where the node features are not sliced but the device changed.
    # Not having this will give us a subgraph on GPU but node features on CPU if we don't
    # relabel the nodes.
    node_frames = utils.extract_node_subframes(
        parent, induced_nodes_or_device, store_ids
    )
    edge_frames = utils.extract_edge_subframes(
        parent, induced_edges_or_device, store_ids
    )
    hsg = DGLGraph(sgi.graph, parent.ntypes, parent.etypes)
    utils.set_new_frames(hsg, node_frames=node_frames, edge_frames=edge_frames)
    return hsg


_init_api("dgl.subgraph")
