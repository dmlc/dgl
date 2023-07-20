"""Neighbor sampling APIs"""

import os

import torch

from .. import backend as F, ndarray as nd, utils
from .._ffi.function import _init_api
from ..base import DGLError, EID
from ..heterograph import DGLBlock, DGLGraph
from .utils import EidExcluder

__all__ = [
    "sample_etype_neighbors",
    "sample_neighbors",
    "sample_neighbors_fused",
    "sample_neighbors_biased",
    "select_topk",
]


def _prepare_edge_arrays(g, arg):
    """Converts the argument into a list of NDArrays.

    If the argument is already a list of array-like objects, directly do the
    conversion.

    If the argument is a string, converts g.edata[arg] into a list of NDArrays
    ordered by the edge types.
    """
    if isinstance(arg, list) and len(arg) > 0:
        if isinstance(arg[0], nd.NDArray):
            return arg
        else:
            # The list can have None as placeholders for empty arrays with
            # undetermined data type.
            dtype = None
            ctx = None
            result = []
            for entry in arg:
                if F.is_tensor(entry):
                    result.append(F.to_dgl_nd(entry))
                    dtype = F.dtype(entry)
                    ctx = F.context(entry)
                else:
                    result.append(None)

            result = [
                F.to_dgl_nd(F.copy_to(F.tensor([], dtype=dtype), ctx))
                if x is None
                else x
                for x in result
            ]
            return result
    elif arg is None:
        return [nd.array([], ctx=nd.cpu())] * len(g.etypes)
    else:
        arrays = []
        for etype in g.canonical_etypes:
            if arg in g.edges[etype].data:
                arrays.append(F.to_dgl_nd(g.edges[etype].data[arg]))
            else:
                arrays.append(nd.array([], ctx=nd.cpu()))
        return arrays


def sample_etype_neighbors(
    g,
    nodes,
    etype_offset,
    fanout,
    edge_dir="in",
    prob=None,
    replace=False,
    copy_ndata=True,
    copy_edata=True,
    etype_sorted=False,
    _dist_training=False,
    output_device=None,
):
    """Sample neighboring edges of the given nodes and return the induced subgraph.

    For each node, a number of inbound (or outbound when ``edge_dir == 'out'``) edges
    will be randomly chosen.  The graph returned will then contain all the nodes in the
    original graph, but only the sampled edges.

    Node/edge features are not preserved. The original IDs of
    the sampled edges are stored as the `dgl.EID` feature in the returned graph.

    Parameters
    ----------
    g : DGLGraph
        The graph.  Can only be in CPU. Should only have one node type and one edge type.
    nodes : tensor or dict
        Node IDs to sample neighbors from.

        This argument can take a single ID tensor or a dictionary of node types and ID tensors.
        If a single tensor is given, the graph must only have one type of nodes.
    etype_offset : list[int]
        The offset of each edge type ID.
    fanout : Tensor
        The number of edges to be sampled for each node per edge type.  Must be a
        1D tensor with the number of elements same as the number of edge types.

        If -1 is given, all of the neighbors with non-zero probability will be selected.
    edge_dir : str, optional
        Determines whether to sample inbound or outbound edges.

        Can take either ``in`` for inbound edges or ``out`` for outbound edges.
    prob : list[Tensor], optional
        The (unnormalized) probabilities associated with each neighboring edge of
        a node.

        The features must be non-negative floats or boolean.  Otherwise, the
        result will be undefined.
    replace : bool, optional
        If True, sample with replacement.
    copy_ndata: bool, optional
        If True, the node features of the new graph are copied from
        the original graph. If False, the new graph will not have any
        node features.

        (Default: True)
    copy_edata: bool, optional
        If True, the edge features of the new graph are copied from
        the original graph.  If False, the new graph will not have any
        edge features.

        (Default: True)
    _dist_training : bool, optional
        Internal argument.  Do not use.

        (Default: False)
    etype_sorted: bool, optional
        A hint telling whether the etypes are already sorted.

        (Default: False)
    output_device : Framework-specific device context object, optional
        The output device.  Default is the same as the input graph.

    Returns
    -------
    DGLGraph
        A sampled subgraph containing only the sampled neighboring edges, with the
        same device as the input graph.

    Notes
    -----
    If :attr:`copy_ndata` or :attr:`copy_edata` is True, same tensors are used as
    the node or edge features of the original graph and the new graph.
    As a result, users should avoid performing in-place operations
    on the node features of the new graph to avoid feature corruption.
    """
    if g.device != F.cpu():
        raise DGLError("The graph should be in cpu.")
    # (BarclayII) because the homogenized graph no longer contains the *name* of edge
    # types, the fanout argument can no longer be a dict of etypes and ints, as opposed
    # to sample_neighbors.
    if not F.is_tensor(fanout):
        raise DGLError("The fanout should be a tensor")
    if isinstance(nodes, dict):
        assert len(nodes) == 1, "The input graph should not have node types"
        nodes = list(nodes.values())[0]

    nodes = utils.prepare_tensor(g, nodes, "nodes")
    device = utils.context_of(nodes)
    nodes = F.to_dgl_nd(nodes)
    # treat etypes as int32, it is much cheaper than int64
    # TODO(xiangsx): int8 can be a better choice.
    fanout = F.to_dgl_nd(fanout)

    prob_array = _prepare_edge_arrays(g, prob)

    subgidx = _CAPI_DGLSampleNeighborsEType(
        g._graph,
        nodes,
        etype_offset,
        fanout,
        edge_dir,
        prob_array,
        replace,
        etype_sorted,
    )
    induced_edges = subgidx.induced_edges
    ret = DGLGraph(subgidx.graph, g.ntypes, g.etypes)

    # handle features
    # (TODO) (BarclayII) DGL distributed fails with bus error, freezes, or other
    # incomprehensible errors with lazy feature copy.
    # So in distributed training context, we fall back to old behavior where we
    # only set the edge IDs.
    if not _dist_training:
        if copy_ndata:
            node_frames = utils.extract_node_subframes(g, device)
            utils.set_new_frames(ret, node_frames=node_frames)

        if copy_edata:
            edge_frames = utils.extract_edge_subframes(g, induced_edges)
            utils.set_new_frames(ret, edge_frames=edge_frames)
    else:
        for i, etype in enumerate(ret.canonical_etypes):
            ret.edges[etype].data[EID] = induced_edges[i]

    return ret if output_device is None else ret.to(output_device)


DGLGraph.sample_etype_neighbors = utils.alias_func(sample_etype_neighbors)


def sample_neighbors(
    g,
    nodes,
    fanout,
    edge_dir="in",
    prob=None,
    replace=False,
    copy_ndata=True,
    copy_edata=True,
    _dist_training=False,
    exclude_edges=None,
    output_device=None,
):
    """Sample neighboring edges of the given nodes and return the induced subgraph.

    For each node, a number of inbound (or outbound when ``edge_dir == 'out'``) edges
    will be randomly chosen.  The graph returned will then contain all the nodes in the
    original graph, but only the sampled edges.

    Node/edge features are not preserved. The original IDs of
    the sampled edges are stored as the `dgl.EID` feature in the returned graph.

    GPU sampling is supported for this function. Refer to :ref:`guide-minibatch-gpu-sampling`
    for more details.

    Parameters
    ----------
    g : DGLGraph
        The graph.  Can be either on CPU or GPU.
    nodes : tensor or dict
        Node IDs to sample neighbors from.

        This argument can take a single ID tensor or a dictionary of node types and ID tensors.
        If a single tensor is given, the graph must only have one type of nodes.
    fanout : int or dict[etype, int]
        The number of edges to be sampled for each node on each edge type.

        This argument can take a single int or a dictionary of edge types and ints.
        If a single int is given, DGL will sample this number of edges for each node for
        every edge type.

        If -1 is given for a single edge type, all the neighboring edges with that edge
        type and non-zero probability will be selected.
    edge_dir : str, optional
        Determines whether to sample inbound or outbound edges.

        Can take either ``in`` for inbound edges or ``out`` for outbound edges.
    prob : str, optional
        Feature name used as the (unnormalized) probabilities associated with each
        neighboring edge of a node.  The feature must have only one element for each
        edge.

        The features must be non-negative floats or boolean.  Otherwise, the result
        will be undefined.
    exclude_edges: tensor or dict
        Edge IDs to exclude during sampling neighbors for the seed nodes.

        This argument can take a single ID tensor or a dictionary of edge types and ID tensors.
        If a single tensor is given, the graph must only have one type of nodes.
    replace : bool, optional
        If True, sample with replacement.
    copy_ndata: bool, optional
        If True, the node features of the new graph are copied from
        the original graph. If False, the new graph will not have any
        node features.

        (Default: True)
    copy_edata: bool, optional
        If True, the edge features of the new graph are copied from
        the original graph.  If False, the new graph will not have any
        edge features.

        (Default: True)
    _dist_training : bool, optional
        Internal argument.  Do not use.

        (Default: False)
    output_device : Framework-specific device context object, optional
        The output device.  Default is the same as the input graph.

    Returns
    -------
    DGLGraph
        A sampled subgraph containing only the sampled neighboring edges.

    Notes
    -----
    If :attr:`copy_ndata` or :attr:`copy_edata` is True, same tensors are used as
    the node or edge features of the original graph and the new graph.
    As a result, users should avoid performing in-place operations
    on the node features of the new graph to avoid feature corruption.

    Examples
    --------
    Assume that you have the following graph

    >>> g = dgl.graph(([0, 0, 1, 1, 2, 2], [1, 2, 0, 1, 2, 0]))

    And the weights

    >>> g.edata['prob'] = torch.FloatTensor([0., 1., 0., 1., 0., 1.])

    To sample one inbound edge for node 0 and node 1:

    >>> sg = dgl.sampling.sample_neighbors(g, [0, 1], 1)
    >>> sg.edges(order='eid')
    (tensor([1, 0]), tensor([0, 1]))
    >>> sg.edata[dgl.EID]
    tensor([2, 0])

    To sample one inbound edge for node 0 and node 1 with probability in edge feature
    ``prob``:

    >>> sg = dgl.sampling.sample_neighbors(g, [0, 1], 1, prob='prob')
    >>> sg.edges(order='eid')
    (tensor([2, 1]), tensor([0, 1]))

    With ``fanout`` greater than the number of actual neighbors and without replacement,
    DGL will take all neighbors instead:

    >>> sg = dgl.sampling.sample_neighbors(g, [0, 1], 3)
    >>> sg.edges(order='eid')
    (tensor([1, 2, 0, 1]), tensor([0, 0, 1, 1]))

    To exclude certain EID's during sampling for the seed nodes:

    >>> g = dgl.graph(([0, 0, 1, 1, 2, 2], [1, 2, 0, 1, 2, 0]))
    >>> g_edges = g.all_edges(form='all')``
    (tensor([0, 0, 1, 1, 2, 2]), tensor([1, 2, 0, 1, 2, 0]), tensor([0, 1, 2, 3, 4, 5]))
    >>> sg = dgl.sampling.sample_neighbors(g, [0, 1], 3, exclude_edges=[0, 1, 2])
    >>> sg.all_edges(form='all')
    (tensor([2, 1]), tensor([0, 1]), tensor([0, 1]))
    >>> sg.has_edges_between(g_edges[0][:3],g_edges[1][:3])
    tensor([False, False, False])
    >>> g = dgl.heterograph({
    ...   ('drug', 'interacts', 'drug'): ([0, 0, 1, 1, 3, 2], [1, 2, 0, 1, 2, 0]),
    ...   ('drug', 'interacts', 'gene'): ([0, 0, 1, 1, 2, 2], [1, 2, 0, 1, 2, 0]),
    ...   ('drug', 'treats', 'disease'): ([0, 0, 1, 1, 2, 2], [1, 2, 0, 1, 2, 0])})
    >>> g_edges = g.all_edges(form='all', etype=('drug', 'interacts', 'drug'))
    (tensor([0, 0, 1, 1, 3, 2]), tensor([1, 2, 0, 1, 2, 0]), tensor([0, 1, 2, 3, 4, 5]))
    >>> excluded_edges  = {('drug', 'interacts', 'drug'): g_edges[2][:3]}
    >>> sg = dgl.sampling.sample_neighbors(g, {'drug':[0, 1]}, 3, exclude_edges=excluded_edges)
    >>> sg.all_edges(form='all', etype=('drug', 'interacts', 'drug'))
    (tensor([2, 1]), tensor([0, 1]), tensor([0, 1]))
    >>> sg.has_edges_between(g_edges[0][:3],g_edges[1][:3],etype=('drug', 'interacts', 'drug'))
    tensor([False, False, False])

    """
    if F.device_type(g.device) == "cpu" and not g.is_pinned():
        frontier = _sample_neighbors(
            g,
            nodes,
            fanout,
            edge_dir=edge_dir,
            prob=prob,
            replace=replace,
            copy_ndata=copy_ndata,
            copy_edata=copy_edata,
            exclude_edges=exclude_edges,
        )
    else:
        frontier = _sample_neighbors(
            g,
            nodes,
            fanout,
            edge_dir=edge_dir,
            prob=prob,
            replace=replace,
            copy_ndata=copy_ndata,
            copy_edata=copy_edata,
        )
        if exclude_edges is not None:
            eid_excluder = EidExcluder(exclude_edges)
            frontier = eid_excluder(frontier)
    return frontier if output_device is None else frontier.to(output_device)


def sample_neighbors_fused(
    g,
    nodes,
    fanout,
    edge_dir="in",
    prob=None,
    replace=False,
    copy_ndata=True,
    copy_edata=True,
    exclude_edges=None,
    mapping=None,
):
    """Sample neighboring edges of the given nodes and return the induced subgraph.

    For each node, a number of inbound (or outbound when ``edge_dir == 'out'``) edges
    will be randomly chosen.  The graph returned will then contain all the nodes in the
    original graph, but only the sampled edges. Nodes will be renumbered starting from id 0,
    which would be new node id of first seed node.

    Parameters
    ----------
    g : DGLGraph
        The graph.  Can be either on CPU or GPU.
    nodes : tensor or dict
        Node IDs to sample neighbors from.

        This argument can take a single ID tensor or a dictionary of node types and ID tensors.
        If a single tensor is given, the graph must only have one type of nodes.
    fanout : int or dict[etype, int]
        The number of edges to be sampled for each node on each edge type.

        This argument can take a single int or a dictionary of edge types and ints.
        If a single int is given, DGL will sample this number of edges for each node for
        every edge type.

        If -1 is given for a single edge type, all the neighboring edges with that edge
        type and non-zero probability will be selected.
    edge_dir : str, optional
        Determines whether to sample inbound or outbound edges.

        Can take either ``in`` for inbound edges or ``out`` for outbound edges.
    prob : str, optional
        Feature name used as the (unnormalized) probabilities associated with each
        neighboring edge of a node.  The feature must have only one element for each
        edge.

        The features must be non-negative floats or boolean.  Otherwise, the result
        will be undefined.
    exclude_edges: tensor or dict
        Edge IDs to exclude during sampling neighbors for the seed nodes.

        This argument can take a single ID tensor or a dictionary of edge types and ID tensors.
        If a single tensor is given, the graph must only have one type of nodes.
    replace : bool, optional
        If True, sample with replacement.
    copy_ndata: bool, optional
        If True, the node features of the new graph are copied from
        the original graph. If False, the new graph will not have any
        node features.

        (Default: True)
    copy_edata: bool, optional
        If True, the edge features of the new graph are copied from
        the original graph.  If False, the new graph will not have any
        edge features.

        (Default: False)

    mapping : dictionary, optional
        Used by fused version of NeighborSampler. To avoid constant data allocation
        provide empty dictionary ({}) that will be allocated once with proper data and reused
        by each function call

        (Default: None)
    Returns
    -------
    DGLGraph
        A sampled subgraph containing only the sampled neighboring edges.

    Notes
    -----
    If :attr:`copy_ndata` or :attr:`copy_edata` is True, same tensors are used as
    the node or edge features of the original graph and the new graph.
    As a result, users should avoid performing in-place operations
    on the node features of the new graph to avoid feature corruption.

    """
    if not g.is_pinned():
        frontier = _sample_neighbors(
            g,
            nodes,
            fanout,
            edge_dir=edge_dir,
            prob=prob,
            replace=replace,
            copy_ndata=copy_ndata,
            copy_edata=copy_edata,
            exclude_edges=exclude_edges,
            fused=True,
            mapping=mapping,
        )
    else:
        frontier = _sample_neighbors(
            g,
            nodes,
            fanout,
            edge_dir=edge_dir,
            prob=prob,
            replace=replace,
            copy_ndata=copy_ndata,
            copy_edata=copy_edata,
            fused=True,
            mapping=mapping,
        )
        if exclude_edges is not None:
            eid_excluder = EidExcluder(exclude_edges)
            frontier = eid_excluder(frontier)
    return frontier


def _sample_neighbors(
    g,
    nodes,
    fanout,
    edge_dir="in",
    prob=None,
    replace=False,
    copy_ndata=True,
    copy_edata=True,
    _dist_training=False,
    exclude_edges=None,
    fused=False,
    mapping=None,
):
    if not isinstance(nodes, dict):
        if len(g.ntypes) > 1:
            raise DGLError(
                "Must specify node type when the graph is not homogeneous."
            )
        nodes = {g.ntypes[0]: nodes}

    nodes = utils.prepare_tensor_dict(g, nodes, "nodes")
    if len(nodes) == 0:
        raise ValueError(
            "Got an empty dictionary in the nodes argument. "
            "Please pass in a dictionary with empty tensors as values instead."
        )
    device = utils.context_of(nodes)
    ctx = utils.to_dgl_context(device)
    nodes_all_types = []
    for ntype in g.ntypes:
        if ntype in nodes:
            nodes_all_types.append(F.to_dgl_nd(nodes[ntype]))
        else:
            nodes_all_types.append(nd.array([], ctx=ctx))

    if isinstance(fanout, nd.NDArray):
        fanout_array = fanout
    else:
        if not isinstance(fanout, dict):
            fanout_array = [int(fanout)] * len(g.etypes)
        else:
            if len(fanout) != len(g.etypes):
                raise DGLError(
                    "Fan-out must be specified for each edge type "
                    "if a dict is provided."
                )
            fanout_array = [None] * len(g.etypes)
            for etype, value in fanout.items():
                fanout_array[g.get_etype_id(etype)] = value
        fanout_array = F.to_dgl_nd(F.tensor(fanout_array, dtype=F.int64))

    prob_arrays = _prepare_edge_arrays(g, prob)

    excluded_edges_all_t = []
    if exclude_edges is not None:
        if not isinstance(exclude_edges, dict):
            if len(g.etypes) > 1:
                raise DGLError(
                    "Must specify etype when the graph is not homogeneous."
                )
            exclude_edges = {g.canonical_etypes[0]: exclude_edges}
        exclude_edges = utils.prepare_tensor_dict(g, exclude_edges, "edges")
        for etype in g.canonical_etypes:
            if etype in exclude_edges:
                excluded_edges_all_t.append(F.to_dgl_nd(exclude_edges[etype]))
            else:
                excluded_edges_all_t.append(nd.array([], ctx=ctx))

    if fused:
        if _dist_training:
            raise DGLError(
                "distributed training not supported in fused sampling"
            )
        cpu = F.device_type(g.device) == "cpu"
        if isinstance(nodes, dict):
            for ntype in list(nodes.keys()):
                if not cpu:
                    break
                cpu = cpu and F.device_type(nodes[ntype].device) == "cpu"
        else:
            cpu = cpu and F.device_type(nodes.device) == "cpu"
        if not cpu or F.backend_name != "pytorch":
            raise DGLError(
                "Only PyTorch backend and cpu is supported in fused sampling"
            )

        if mapping is None:
            mapping = {}
        mapping_name = "__mapping" + str(os.getpid())
        if mapping_name not in mapping.keys():
            mapping[mapping_name] = [
                torch.LongTensor(g.num_nodes(ntype)).fill_(-1)
                for ntype in g.ntypes
            ]

        subgidx, induced_nodes, induced_edges = _CAPI_DGLSampleNeighborsFused(
            g._graph,
            nodes_all_types,
            [F.to_dgl_nd(m) for m in mapping[mapping_name]],
            fanout_array,
            edge_dir,
            prob_arrays,
            excluded_edges_all_t,
            replace,
        )
        for mapping_vector, src_nodes in zip(
            mapping[mapping_name], induced_nodes
        ):
            mapping_vector[F.from_dgl_nd(src_nodes).type(F.int64)] = -1

        new_ntypes = (g.ntypes, g.ntypes)
        ret = DGLBlock(subgidx, new_ntypes, g.etypes)
        assert ret.is_unibipartite

    else:
        subgidx = _CAPI_DGLSampleNeighbors(
            g._graph,
            nodes_all_types,
            fanout_array,
            edge_dir,
            prob_arrays,
            excluded_edges_all_t,
            replace,
        )
        ret = DGLGraph(subgidx.graph, g.ntypes, g.etypes)
        induced_edges = subgidx.induced_edges

    # handle features
    # (TODO) (BarclayII) DGL distributed fails with bus error, freezes, or other
    # incomprehensible errors with lazy feature copy.
    # So in distributed training context, we fall back to old behavior where we
    # only set the edge IDs.
    if not _dist_training:
        if copy_ndata:
            if fused:
                src_node_ids = [F.from_dgl_nd(src) for src in induced_nodes]
                dst_node_ids = [
                    utils.toindex(
                        nodes.get(ntype, []), g._idtype_str
                    ).tousertensor(ctx=F.to_backend_ctx(g._graph.ctx))
                    for ntype in g.ntypes
                ]
                node_frames = utils.extract_node_subframes_for_block(
                    g, src_node_ids, dst_node_ids
                )
                utils.set_new_frames(ret, node_frames=node_frames)
            else:
                node_frames = utils.extract_node_subframes(g, device)
                utils.set_new_frames(ret, node_frames=node_frames)

        if copy_edata:
            if fused:
                edge_ids = [F.from_dgl_nd(eid) for eid in induced_edges]
                edge_frames = utils.extract_edge_subframes(g, edge_ids)
                utils.set_new_frames(ret, edge_frames=edge_frames)
            else:
                edge_frames = utils.extract_edge_subframes(g, induced_edges)
                utils.set_new_frames(ret, edge_frames=edge_frames)

    else:
        for i, etype in enumerate(ret.canonical_etypes):
            ret.edges[etype].data[EID] = induced_edges[i]

    return ret


DGLGraph.sample_neighbors = utils.alias_func(sample_neighbors)
DGLGraph.sample_neighbors_fused = utils.alias_func(sample_neighbors_fused)


def sample_neighbors_biased(
    g,
    nodes,
    fanout,
    bias,
    edge_dir="in",
    tag_offset_name="_TAG_OFFSET",
    replace=False,
    copy_ndata=True,
    copy_edata=True,
    output_device=None,
):
    r"""Sample neighboring edges of the given nodes and return the induced subgraph, where each
    neighbor's probability to be picked is determined by its tag.

    For each node, a number of inbound (or outbound when ``edge_dir == 'out'``) edges
    will be randomly chosen.  The graph returned will then contain all the nodes in the
    original graph, but only the sampled edges.

    This version of neighbor sampling can support the scenario where adjacent nodes with different
    types have different sampling probability. Each node is assigned an integer (called a *tag*)
    which represents its type. Tag is an analogue of node type under the framework of homogeneous
    graphs. Nodes with the same tag share the same probability.

    For example, assume a node has :math:`N+M` neighbors, and :math:`N` of them
    have tag 0 while :math:`M` of them have tag 1. Assume a node of tag 0 has
    an unnormalized probability :math:`p` to be picked while a node of tag 1
    has :math:`q`. This function first chooses a tag according to the
    unnormalized probability distribution
    :math:`\frac{P(tag=0)}{P(tag=1)}=\frac{Np}{Mq}`, and then run a uniform
    sampling to get a node of the chosen tag.

    In order to make sampling more efficient, the input graph must have its
    CSC matrix (or CSR matrix if ``edge_dir='out'``) sorted according to the tag. The API
    :func:`~dgl.sort_csc_by_tag` and
    :func:`~dgl.sort_csr_by_tag` are designed for this purpose, which
    will internally reorder the neighbors by tags so that neighbors of the same tags are
    stored in a consecutive range. The two APIs will also store the offsets of these ranges
    in a node feature with :attr:`tag_offset_name` as its name.

    **Please make sure that the CSR (or CSC) matrix of the graph has been sorted before
    calling this function.**  This function itself will not check whether the
    input graph is sorted. Note that the input :attr:`tag_offset_name` should
    be consistent with that in the sorting function.

    Only homogeneous or bipartite graphs are supported. For bipartite graphs,
    the tag offsets of the source nodes when ``edge_dir='in'`` (or the destination
    nodes when ``edge_dir='out'``) will be used in sampling.

    Node/edge features are not preserved. The original IDs of
    the sampled edges are stored as the ``dgl.EID`` feature in the returned graph.

    Parameters
    ----------
    g : DGLGraph
        The graph. Must be homogeneous or bipartite (only one edge type). Must be on CPU.
    nodes : tensor or list
        Node IDs to sample neighbors from.
    fanout : int
        The number of edges to be sampled for each node on each edge type.

        If -1 is given, all the neighboring edges with non-zero probability will be selected.
    bias : tensor or list
        The (unnormalized) probabilities associated with each tag. Its length should be equal
        to the number of tags.

        Entries of this array must be non-negative floats. Otherwise, the result will be
        undefined.
    edge_dir : str, optional
        Determines whether to sample inbound or outbound edges.

        Can take either ``in`` for inbound edges or ``out`` for outbound edges.
    tag_offset_name : str, optional
        The name of the node feature storing tag offsets.

        (Default: "_TAG_OFFSET")
    replace : bool, optional
        If True, sample with replacement.
    copy_ndata: bool, optional
        If True, the node features of the new graph are copied from
        the original graph. If False, the new graph will not have any
        node features.

        (Default: True)
    copy_edata: bool, optional
        If True, the edge features of the new graph are copied from
        the original graph.  If False, the new graph will not have any
        edge features.

        (Default: True)
    output_device : Framework-specific device context object, optional
        The output device.  Default is the same as the input graph.

    Returns
    -------
    DGLGraph
        A sampled subgraph containing only the sampled neighboring edges.  It is on CPU.

    Notes
    -----
    If :attr:`copy_ndata` or :attr:`copy_edata` is True, same tensors are used as
    the node or edge features of the original graph and the new graph.
    As a result, users should avoid performing in-place operations
    on the node features of the new graph to avoid feature corruption.

    See Also
    --------
    dgl.sort_csc_by_tag
    dgl.sort_csr_by_tag

    Examples
    --------
    Assume that you have the following graph

    >>> g = dgl.graph(([0, 0, 1, 1, 2, 2], [1, 2, 0, 1, 2, 0]))

    And the tags

    >>> tag = torch.IntTensor([0, 0, 1])

    Sort the graph (necessary!)

    >>> g_sorted = dgl.transforms.sort_csr_by_tag(g, tag)
    >>> g_sorted.ndata['_TAG_OFFSET']
    tensor([[0, 1, 2],
            [0, 2, 2],
            [0, 1, 2]])

    Set the probability of each tag:

    >>> bias = torch.tensor([1.0, 0.001])
    >>> # node 2 is almost impossible to be sampled because it has tag 1.

    To sample one out bound edge for node 0 and node 2:

    >>> sg = dgl.sampling.sample_neighbors_biased(g_sorted, [0, 2], 1, bias, edge_dir='out')
    >>> sg.edges(order='eid')
    (tensor([0, 2]), tensor([1, 0]))
    >>> sg.edata[dgl.EID]
    tensor([0, 5])

    With ``fanout`` greater than the number of actual neighbors and without replacement,
    DGL will take all neighbors instead:

    >>> sg = dgl.sampling.sample_neighbors_biased(g_sorted, [0, 2], 3, bias, edge_dir='out')
    >>> sg.edges(order='eid')
    (tensor([0, 0, 2, 2]), tensor([1, 2, 0, 2]))
    """
    if isinstance(nodes, list):
        nodes = F.tensor(nodes)
    if isinstance(bias, list):
        bias = F.tensor(bias)
    device = utils.context_of(nodes)

    nodes_array = F.to_dgl_nd(nodes)
    bias_array = F.to_dgl_nd(bias)
    if edge_dir == "in":
        tag_offset_array = F.to_dgl_nd(g.dstdata[tag_offset_name])
    elif edge_dir == "out":
        tag_offset_array = F.to_dgl_nd(g.srcdata[tag_offset_name])
    else:
        raise DGLError("edge_dir can only be 'in' or 'out'")

    subgidx = _CAPI_DGLSampleNeighborsBiased(
        g._graph,
        nodes_array,
        fanout,
        bias_array,
        tag_offset_array,
        edge_dir,
        replace,
    )
    induced_edges = subgidx.induced_edges
    ret = DGLGraph(subgidx.graph, g.ntypes, g.etypes)

    if copy_ndata:
        node_frames = utils.extract_node_subframes(g, device)
        utils.set_new_frames(ret, node_frames=node_frames)

    if copy_edata:
        edge_frames = utils.extract_edge_subframes(g, induced_edges)
        utils.set_new_frames(ret, edge_frames=edge_frames)

    ret.edata[EID] = induced_edges[0]
    return ret if output_device is None else ret.to(output_device)


DGLGraph.sample_neighbors_biased = utils.alias_func(sample_neighbors_biased)


def select_topk(
    g,
    k,
    weight,
    nodes=None,
    edge_dir="in",
    ascending=False,
    copy_ndata=True,
    copy_edata=True,
    output_device=None,
):
    """Select the neighboring edges with k-largest (or k-smallest) weights of the given
    nodes and return the induced subgraph.

    For each node, a number of inbound (or outbound when ``edge_dir == 'out'``) edges
    with the largest (or smallest when ``ascending == True``) weights will be chosen.
    The graph returned will then contain all the nodes in the original graph, but only
    the sampled edges.

    Node/edge features are not preserved. The original IDs of
    the sampled edges are stored as the `dgl.EID` feature in the returned graph.

    Parameters
    ----------
    g : DGLGraph
        The graph.  Must be on CPU.
    k : int or dict[etype, int]
        The number of edges to be selected for each node on each edge type.

        This argument can take a single int or a dictionary of edge types and ints.
        If a single int is given, DGL will select this number of edges for each node for
        every edge type.

        If -1 is given for a single edge type, all the neighboring edges with that edge
        type will be selected.
    weight : str
        Feature name of the weights associated with each edge.  The feature should have only
        one element for each edge.  The feature can be either int32/64 or float32/64.
    nodes : tensor or dict, optional
        Node IDs to sample neighbors from.

        This argument can take a single ID tensor or a dictionary of node types and ID tensors.
        If a single tensor is given, the graph must only have one type of nodes.

        If None, DGL will select the edges for all nodes.
    edge_dir : str, optional
        Determines whether to sample inbound or outbound edges.

        Can take either ``in`` for inbound edges or ``out`` for outbound edges.
    ascending : bool, optional
        If True, DGL will return edges with k-smallest weights instead of
        k-largest weights.
    copy_ndata: bool, optional
        If True, the node features of the new graph are copied from
        the original graph. If False, the new graph will not have any
        node features.

        (Default: True)
    copy_edata: bool, optional
        If True, the edge features of the new graph are copied from
        the original graph.  If False, the new graph will not have any
        edge features.

        (Default: True)
    output_device : Framework-specific device context object, optional
        The output device.  Default is the same as the input graph.

    Returns
    -------
    DGLGraph
        A sampled subgraph containing only the sampled neighboring edges.  It is on CPU.

    Notes
    -----
    If :attr:`copy_ndata` or :attr:`copy_edata` is True, same tensors are used as
    the node or edge features of the original graph and the new graph.
    As a result, users should avoid performing in-place operations
    on the node features of the new graph to avoid feature corruption.

    Examples
    --------
    >>> g = dgl.graph(([0, 0, 1, 1, 2, 2], [1, 2, 0, 1, 2, 0]))
    >>> g.edata['weight'] = torch.FloatTensor([0, 1, 0, 1, 0, 1])
    >>> sg = dgl.sampling.select_topk(g, 1, 'weight')
    >>> sg.edges(order='eid')
    (tensor([2, 1, 0]), tensor([0, 1, 2]))
    """
    # Rectify nodes to a dictionary
    if nodes is None:
        nodes = {
            ntype: F.astype(F.arange(0, g.num_nodes(ntype)), g.idtype)
            for ntype in g.ntypes
        }
    elif not isinstance(nodes, dict):
        if len(g.ntypes) > 1:
            raise DGLError(
                "Must specify node type when the graph is not homogeneous."
            )
        nodes = {g.ntypes[0]: nodes}
    assert g.device == F.cpu(), "Graph must be on CPU."

    # Parse nodes into a list of NDArrays.
    nodes = utils.prepare_tensor_dict(g, nodes, "nodes")
    device = utils.context_of(nodes)
    nodes_all_types = []
    for ntype in g.ntypes:
        if ntype in nodes:
            nodes_all_types.append(F.to_dgl_nd(nodes[ntype]))
        else:
            nodes_all_types.append(nd.array([], ctx=nd.cpu()))

    if not isinstance(k, dict):
        k_array = [int(k)] * len(g.etypes)
    else:
        if len(k) != len(g.etypes):
            raise DGLError(
                "K value must be specified for each edge type "
                "if a dict is provided."
            )
        k_array = [None] * len(g.etypes)
        for etype, value in k.items():
            k_array[g.get_etype_id(etype)] = value
    k_array = F.to_dgl_nd(F.tensor(k_array, dtype=F.int64))

    weight_arrays = []
    for etype in g.canonical_etypes:
        if weight in g.edges[etype].data:
            weight_arrays.append(F.to_dgl_nd(g.edges[etype].data[weight]))
        else:
            raise DGLError(
                'Edge weights "{}" do not exist for relation graph "{}".'.format(
                    weight, etype
                )
            )

    subgidx = _CAPI_DGLSampleNeighborsTopk(
        g._graph,
        nodes_all_types,
        k_array,
        edge_dir,
        weight_arrays,
        bool(ascending),
    )
    induced_edges = subgidx.induced_edges
    ret = DGLGraph(subgidx.graph, g.ntypes, g.etypes)

    # handle features
    if copy_ndata:
        node_frames = utils.extract_node_subframes(g, device)
        utils.set_new_frames(ret, node_frames=node_frames)

    if copy_edata:
        edge_frames = utils.extract_edge_subframes(g, induced_edges)
        utils.set_new_frames(ret, edge_frames=edge_frames)
    return ret if output_device is None else ret.to(output_device)


DGLGraph.select_topk = utils.alias_func(select_topk)

_init_api("dgl.sampling.neighbor", __name__)
