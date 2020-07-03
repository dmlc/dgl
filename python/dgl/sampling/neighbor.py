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
    """Sample from the neighbors of the given nodes and return the induced subgraph.

    When sampling with replacement, the sampled subgraph could have parallel edges.

    For sampling without replace, if fanout > the number of neighbors, all the
    neighbors are sampled.

    Node/edge features are not preserved. The original IDs of
    the sampled edges are stored as the `dgl.EID` feature in the returned graph.

    Parameters
    ----------
    g : DGLHeteroGraph
        Full graph structure.
    nodes : tensor or dict
        Node ids to sample neighbors from. The allowed types
        are dictionary of node types to node id tensors, or simply node id tensor if
        the given graph g has only one type of nodes.
    fanout : int or dict[etype, int]
        The number of sampled neighbors for each node on each edge type. Provide a dict
        to specify different fanout values for each edge type.

        If -1 is given, select all the neighbors.  ``prob`` and ``replace`` will be
        ignored in this case.
    edge_dir : str, optional
        Edge direction ('in' or 'out'). If is 'in', sample from in edges. Otherwise,
        sample from out edges.
    prob : str, optional
        Feature name used as the probabilities associated with each neighbor of a node.
        Its shape should be compatible with a scalar edge feature tensor.
    replace : bool, optional
        If True, sample with replacement.

    Returns
    -------
    DGLHeteroGraph
        A sampled subgraph containing only the sampled neighbor edges from
        ``nodes``. The sampled subgraph has the same metagraph as the original
        one.
    """
    if not isinstance(nodes, dict):
        if len(g.ntypes) > 1:
            raise DGLError("Must specify node type when the graph is not homogeneous.")
        nodes = {g.ntypes[0] : nodes}
    nodes_all_types = []
    for ntype in g.ntypes:
        if ntype in nodes:
            nodes_all_types.append(utils.toindex(nodes[ntype], g._idtype_str).todgltensor())
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
    fanout_array = utils.toindex(fanout_array).todgltensor()

    if prob is None:
        prob_arrays = [nd.array([], ctx=nd.cpu())] * len(g.etypes)
    else:
        prob_arrays = []
        for etype in g.canonical_etypes:
            if prob in g.edges[etype].data:
                prob_arrays.append(F.zerocopy_to_dgl_ndarray(g.edges[etype].data[prob]))
            else:
                prob_arrays.append(nd.array([], ctx=nd.cpu()))

    subgidx = _CAPI_DGLSampleNeighbors(g._graph, nodes_all_types, fanout_array,
                                       edge_dir, prob_arrays, replace)
    induced_edges = subgidx.induced_edges
    ret = DGLHeteroGraph(subgidx.graph, g.ntypes, g.etypes)
    for i, etype in enumerate(ret.canonical_etypes):
        ret.edges[etype].data[EID] = induced_edges[i].tousertensor()
    return ret

def select_topk(g, k, weight, nodes=None, edge_dir='in', ascending=False):
    """Select the neighbors with k-largest weights on the connecting edges for each given node.

    If k > the number of neighbors, all the neighbors are sampled.

    Node/edge features are not preserved. The original IDs of
    the sampled edges are stored as the `dgl.EID` feature in the returned graph.

    Parameters
    ----------
    g : DGLHeteroGraph
        Full graph structure.
    k : int or dict[etype, int]
        The K value.

        If -1 is given, select all the neighbors.
    weight : str
        Feature name of the weights associated with each edge. Its shape should be
        compatible with a scalar edge feature tensor.
    nodes : tensor or dict, optional
        Node ids to sample neighbors from. The allowed types
        are dictionary of node types to node id tensors, or simply node id
        tensor if the given graph g has only one type of nodes.
    edge_dir : str, optional
        Edge direction ('in' or 'out'). If is 'in', sample from in edges.
        Otherwise, sample from out edges.
    ascending : bool, optional
        If true, elements are sorted by ascending order, equivalent to find
        the K smallest values. Otherwise, find K largest values.

    Returns
    -------
    DGLHeteroGraph
        A sampled subgraph by top k criterion. The sampled subgraph has the same
        metagraph as the original one.
    """
    # Rectify nodes to a dictionary
    if nodes is None:
        nodes = {ntype: F.arange(0, g.number_of_nodes(ntype)) for ntype in g.ntypes}
    elif not isinstance(nodes, dict):
        if len(g.ntypes) > 1:
            raise DGLError("Must specify node type when the graph is not homogeneous.")
        nodes = {g.ntypes[0] : nodes}

    # Parse nodes into a list of NDArrays.
    nodes_all_types = []
    for ntype in g.ntypes:
        if ntype in nodes:
            nodes_all_types.append(utils.toindex(nodes[ntype], g._idtype_str).todgltensor())
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
    k_array = utils.toindex(k_array).todgltensor()

    weight_arrays = []
    for etype in g.canonical_etypes:
        if weight in g.edges[etype].data:
            weight_arrays.append(F.zerocopy_to_dgl_ndarray(g.edges[etype].data[weight]))
        else:
            raise DGLError('Edge weights "{}" do not exist for relation graph "{}".'.format(
                weight, etype))

    subgidx = _CAPI_DGLSampleNeighborsTopk(
        g._graph, nodes_all_types, k_array, edge_dir, weight_arrays, bool(ascending))
    induced_edges = subgidx.induced_edges
    ret = DGLHeteroGraph(subgidx.graph, g.ntypes, g.etypes)
    for i, etype in enumerate(ret.canonical_etypes):
        ret.edges[etype].data[EID] = induced_edges[i].tousertensor()
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
