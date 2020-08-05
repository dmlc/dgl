"""Neighbor sampling APIs"""

from .._ffi.function import _init_api
from .. import backend as F
from ..base import DGLError, EID
from ..heterograph import DGLHeteroGraph
from .. import ndarray as nd
from .. import utils
from .. import transform
from .dataloader import BlockSampler, assign_block_eids

__all__ = [
    'sample_neighbors',
    'select_topk',
    'MultiLayerNeighborSampler']

def sample_neighbors(g, nodes, fanout, edge_dir='in', prob=None, replace=False):
    """Sample neighboring edges of the given nodes and return the induced subgraph.

    For each node, a number of inbound (or outbound when ``edge_dir == 'out'``) edges
    will be randomly chosen.  The graph returned will then contain all the nodes in the
    original graph, but only the sampled edges.

    Node/edge features are not preserved. The original IDs of
    the sampled edges are stored as the `dgl.EID` feature in the returned graph.

    Parameters
    ----------
    g : DGLGraph
        The graph
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
        type will be selected.
    edge_dir : str, optional
        Determines whether to sample inbound or outbound edges.

        Can take either ``in`` for inbound edges or ``out`` for outbound edges.
    prob : str, optional
        Feature name used as the (unnormalized) probabilities associated with each
        neighboring edge of a node.  The feature must have only one element for each
        edge.

        The features must be non-negative floats, and the sum of the features of
        inbound/outbound edges for every node must be positive (though they don't have
        to sum up to one).  Otherwise, the result will be undefined.
    replace : bool, optional
        If True, sample with replacement.

    Returns
    -------
    DGLGraph
        A sampled subgraph containing only the sampled neighboring edges.

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
    """
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
            nodes_all_types.append(nd.array([], ctx=nd.cpu()))

    if not isinstance(fanout, dict):
        fanout_array = [int(fanout)] * len(g.etypes)
    else:
        if len(fanout) != len(g.etypes):
            raise DGLError('Fan-out must be specified for each edge type '
                           'if a dict is provided.')
        fanout_array = [None] * len(g.etypes)
        for etype, value in fanout.items():
            fanout_array[g.get_etype_id(etype)] = value
    fanout_array = F.to_dgl_nd(F.tensor(fanout_array, dtype=F.int64))

    if prob is None:
        prob_arrays = [nd.array([], ctx=nd.cpu())] * len(g.etypes)
    else:
        prob_arrays = []
        for etype in g.canonical_etypes:
            if prob in g.edges[etype].data:
                prob_arrays.append(F.to_dgl_nd(g.edges[etype].data[prob]))
            else:
                prob_arrays.append(nd.array([], ctx=nd.cpu()))

    subgidx = _CAPI_DGLSampleNeighbors(g._graph, nodes_all_types, fanout_array,
                                       edge_dir, prob_arrays, replace)
    induced_edges = subgidx.induced_edges
    ret = DGLHeteroGraph(subgidx.graph, g.ntypes, g.etypes)
    for i, etype in enumerate(ret.canonical_etypes):
        ret.edges[etype].data[EID] = induced_edges[i]
    return ret

def select_topk(g, k, weight, nodes=None, edge_dir='in', ascending=False):
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
        The graph
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

    Returns
    -------
    DGLGraph
        A sampled subgraph containing only the sampled neighboring edges.

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
        nodes = {ntype: F.arange(0, g.number_of_nodes(ntype)) for ntype in g.ntypes}
    elif not isinstance(nodes, dict):
        if len(g.ntypes) > 1:
            raise DGLError("Must specify node type when the graph is not homogeneous.")
        nodes = {g.ntypes[0] : nodes}

    # Parse nodes into a list of NDArrays.
    nodes = utils.prepare_tensor_dict(g, nodes, 'nodes')
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
            raise DGLError('K value must be specified for each edge type '
                           'if a dict is provided.')
        k_array = [None] * len(g.etypes)
        for etype, value in k.items():
            k_array[g.get_etype_id(etype)] = value
    k_array = F.to_dgl_nd(F.tensor(k_array, dtype=F.int64))

    weight_arrays = []
    for etype in g.canonical_etypes:
        if weight in g.edges[etype].data:
            weight_arrays.append(F.to_dgl_nd(g.edges[etype].data[weight]))
        else:
            raise DGLError('Edge weights "{}" do not exist for relation graph "{}".'.format(
                weight, etype))

    subgidx = _CAPI_DGLSampleNeighborsTopk(
        g._graph, nodes_all_types, k_array, edge_dir, weight_arrays, bool(ascending))
    induced_edges = subgidx.induced_edges
    ret = DGLHeteroGraph(subgidx.graph, g.ntypes, g.etypes)
    for i, etype in enumerate(ret.canonical_etypes):
        ret.edges[etype].data[EID] = induced_edges[i]
    return ret


class MultiLayerNeighborSampler(BlockSampler):
    """Sampler that builds computational dependency of node representations via
    neighbor sampling for multilayer GNN.

    This sampler will make every node gather messages from a fixed number of neighbors
    per edge type.  The neighbors are picked uniformly.

    Parameters
    ----------
    fanouts : list[int] or list[dict[etype, int] or None]
        List of neighbors to sample per edge type for each GNN layer, starting from the
        first layer.

        If the graph is homogeneous, only an integer is needed for each layer.

        If None is provided for one layer, all neighbors will be included regardless of
        edge types.

        If -1 is provided for one edge type on one layer, then all inbound edges
        of that edge type will be included.
    replace : bool, default True
        Whether to sample with replacement
    return_eids : bool, default False
        Whether to return edge IDs of the original graph in the sampled blocks.

        If True, the edge IDs will be stored as ``dgl.EID`` feature for each edge type.

    Examples
    --------
    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from all neighbors (assume
    the backend is PyTorch):
    >>> sampler = dgl.sampling.NeighborSampler([None, None, None])
    >>> collator = dgl.sampling.NodeCollator(g, train_nid, sampler)
    >>> dataloader = torch.utils.data.DataLoader(
    ...     collator.dataset, collate_fn=collator.collate,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for blocks in dataloader:
    ...     train_on(blocks)

    If we wish to gather from 5 neighbors on the first layer, 10 neighbors on the second,
    and 15 layers on the third:
    >>> sampler = dgl.sampling.NeighborSampler([5, 10, 15])

    If training on a heterogeneous graph and you want different number of neighbors for each
    edge type, one should instead provide a list of dicts.  Each dict would specify the
    number of neighbors to pick per edge type.
    >>> sampler = dgl.sampling.NeighborSampler([
    ...     {('user', 'follows', 'user'): 5,
    ...      ('user', 'plays', 'game'): 4,
    ...      ('game', 'played-by', 'user'): 3}] * 3)
    """
    def __init__(self, fanouts, replace=False, return_eids=False):
        super().__init__(len(fanouts))

        self.fanouts = fanouts
        self.replace = replace
        self.return_eids = return_eids
        if return_eids:
            self.set_block_postprocessor(assign_block_eids)

    def sample_frontier(self, block_id, g, seed_nodes, *args, **kwargs):
        fanout = self.fanouts[block_id]
        if fanout is None:
            frontier = transform.in_subgraph(g, seed_nodes)
        else:
            frontier = sample_neighbors(g, seed_nodes, fanout, replace=self.replace)
        return frontier

_init_api('dgl.sampling.neighbor', __name__)
