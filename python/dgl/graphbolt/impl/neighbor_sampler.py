"""Neighbor subgraph samplers for GraphBolt."""

import torch
from torch.utils.data import functional_datapipe

from ..internal import compact_csc_format, unique_and_compact_csc_formats

from ..subgraph_sampler import SubgraphSampler
from .sampled_subgraph_impl import SampledSubgraphImpl


__all__ = ["NeighborSampler", "LayerNeighborSampler"]


@functional_datapipe("sample_neighbor")
class NeighborSampler(SubgraphSampler):
    """Sample neighbor edges from a graph and return a subgraph.

    Functional name: :obj:`sample_neighbor`.

    Neighbor sampler is responsible for sampling a subgraph from given data. It
    returns an induced subgraph along with compacted information. In the
    context of a node classification task, the neighbor sampler directly
    utilizes the nodes provided as seed nodes. However, in scenarios involving
    link prediction, the process needs another pre-peocess operation. That is,
    gathering unique nodes from the given node pairs, encompassing both
    positive and negative node pairs, and employs these nodes as the seed nodes
    for subsequent steps.

    Parameters
    ----------
    datapipe : DataPipe
        The datapipe.
    graph : FusedCSCSamplingGraph
        The graph on which to perform subgraph sampling.
    fanouts: list[torch.Tensor] or list[int]
        The number of edges to be sampled for each node with or without
        considering edge types. The length of this parameter implicitly
        signifies the layer of sampling being conducted.
        Note: The fanout order is from the outermost layer to innermost layer.
        For example, the fanout '[15, 10, 5]' means that 15 to the outermost
        layer, 10 to the intermediate layer and 5 corresponds to the innermost
        layer.
    replace: bool
        Boolean indicating whether the sample is preformed with or
        without replacement. If True, a value can be selected multiple
        times. Otherwise, each value can be selected only once.
    prob_name: str, optional
        The name of an edge attribute used as the weights of sampling for
        each node. This attribute tensor should contain (unnormalized)
        probabilities corresponding to each neighboring edge of a node.
        It must be a 1D floating-point or boolean tensor, with the number
        of elements equalling the total number of edges.
    deduplicate: bool
        Boolean indicating whether seeds between hops will be deduplicated.
        If True, the same elements in seeds will be deleted to only one.
        Otherwise, the same elements will be remained.

    Examples
    -------
    >>> import torch
    >>> import dgl.graphbolt as gb
    >>> indptr = torch.LongTensor([0, 2, 4, 5, 6, 7 ,8])
    >>> indices = torch.LongTensor([1, 2, 0, 3, 5, 4, 3, 5])
    >>> graph = gb.fused_csc_sampling_graph(indptr, indices)
    >>> node_pairs = torch.LongTensor([[0, 1], [1, 2]])
    >>> item_set = gb.ItemSet(node_pairs, names="node_pairs")
    >>> datapipe = gb.ItemSampler(item_set, batch_size=1)
    >>> datapipe = datapipe.sample_uniform_negative(graph, 2)
    >>> datapipe = datapipe.sample_neighbor(graph, [5, 10, 15])
    >>> next(iter(datapipe)).sampled_subgraphs
    [SampledSubgraphImpl(sampled_csc=CSCFormatBase(
            indptr=tensor([0, 2, 4, 5, 6, 7, 8]),
            indices=tensor([1, 4, 0, 5, 5, 3, 3, 2]),
        ),
        original_row_node_ids=tensor([0, 1, 4, 5, 2, 3]),
        original_edge_ids=None,
        original_column_node_ids=tensor([0, 1, 4, 5, 2, 3]),
    ),
    SampledSubgraphImpl(sampled_csc=CSCFormatBase(
            indptr=tensor([0, 2, 4, 5, 6, 7, 8]),
            indices=tensor([1, 4, 0, 5, 5, 3, 3, 2]),
        ),
        original_row_node_ids=tensor([0, 1, 4, 5, 2, 3]),
        original_edge_ids=None,
        original_column_node_ids=tensor([0, 1, 4, 5, 2, 3]),
    ),
    SampledSubgraphImpl(sampled_csc=CSCFormatBase(
            indptr=tensor([0, 2, 4, 5, 6]),
            indices=tensor([1, 4, 0, 5, 5, 3]),
        ),
        original_row_node_ids=tensor([0, 1, 4, 5, 2, 3]),
        original_edge_ids=None,
        original_column_node_ids=tensor([0, 1, 4, 5]),
    )]
    """

    def __init__(
        self,
        datapipe,
        graph,
        fanouts,
        replace=False,
        prob_name=None,
        deduplicate=True,
    ):
        super().__init__(datapipe)
        self.graph = graph
        # Convert fanouts to a list of tensors.
        self.fanouts = []
        for fanout in fanouts:
            if not isinstance(fanout, torch.Tensor):
                fanout = torch.LongTensor([int(fanout)])
            self.fanouts.insert(0, fanout)
        self.replace = replace
        self.prob_name = prob_name
        self.deduplicate = deduplicate
        self.sampler = graph.sample_neighbors

    def sample_subgraphs(self, seeds, seeds_timestamp=None):
        subgraphs = []
        num_layers = len(self.fanouts)
        # Enrich seeds with all node types.
        if isinstance(seeds, dict):
            ntypes = list(self.graph.node_type_to_id.keys())
            # Loop over different seeds to extract the device they are on.
            device = None
            dtype = None
            for _, seed in seeds.items():
                device = seed.device
                dtype = seed.dtype
                break
            default_tensor = torch.tensor([], dtype=dtype, device=device)
            seeds = {
                ntype: seeds.get(ntype, default_tensor) for ntype in ntypes
            }
        for hop in range(num_layers):
            subgraph = self.sampler(
                seeds,
                self.fanouts[hop],
                self.replace,
                self.prob_name,
            )
            if self.deduplicate:
                (
                    original_row_node_ids,
                    compacted_csc_format,
                ) = unique_and_compact_csc_formats(subgraph.sampled_csc, seeds)
                subgraph = SampledSubgraphImpl(
                    sampled_csc=compacted_csc_format,
                    original_column_node_ids=seeds,
                    original_row_node_ids=original_row_node_ids,
                    original_edge_ids=subgraph.original_edge_ids,
                )
            else:
                (
                    original_row_node_ids,
                    compacted_csc_format,
                ) = compact_csc_format(subgraph.sampled_csc, seeds)
                subgraph = SampledSubgraphImpl(
                    sampled_csc=compacted_csc_format,
                    original_column_node_ids=seeds,
                    original_row_node_ids=original_row_node_ids,
                    original_edge_ids=subgraph.original_edge_ids,
                )
            subgraphs.insert(0, subgraph)
            seeds = original_row_node_ids
        return seeds, subgraphs


@functional_datapipe("sample_layer_neighbor")
class LayerNeighborSampler(NeighborSampler):
    """Sample layer neighbor edges from a graph and return a subgraph.

    Functional name: :obj:`sample_layer_neighbor`.

    Sampler that builds computational dependency of node representations via
    labor sampling for multilayer GNN from the NeurIPS 2023 paper
    `Layer-Neighbor Sampling -- Defusing Neighborhood Explosion in GNNs
    <https://arxiv.org/abs/2210.13339>`__

    Layer-Neighbor sampler is responsible for sampling a subgraph from given
    data. It returns an induced subgraph along with compacted information. In
    the context of a node classification task, the neighbor sampler directly
    utilizes the nodes provided as seed nodes. However, in scenarios involving
    link prediction, the process needs another pre-process operation. That is,
    gathering unique nodes from the given node pairs, encompassing both
    positive and negative node pairs, and employs these nodes as the seed nodes
    for subsequent steps.

    Implements the approach described in Appendix A.3 of the paper. Similar to
    dgl.dataloading.LaborSampler but this uses sequential poisson sampling
    instead of poisson sampling to keep the count of sampled edges per vertex
    deterministic like NeighborSampler. Thus, it is a drop-in replacement for
    NeighborSampler. However, unlike NeighborSampler, it samples fewer vertices
    and edges for multilayer GNN scenario without harming convergence speed with
    respect to training iterations.

    Parameters
    ----------
    datapipe : DataPipe
        The datapipe.
    graph : FusedCSCSamplingGraph
        The graph on which to perform subgraph sampling.
    fanouts: list[torch.Tensor]
        The number of edges to be sampled for each node with or without
        considering edge types. The length of this parameter implicitly
        signifies the layer of sampling being conducted.
    replace: bool
        Boolean indicating whether the sample is preformed with or
        without replacement. If True, a value can be selected multiple
        times. Otherwise, each value can be selected only once.
    prob_name: str, optional
        The name of an edge attribute used as the weights of sampling for
        each node. This attribute tensor should contain (unnormalized)
        probabilities corresponding to each neighboring edge of a node.
        It must be a 1D floating-point or boolean tensor, with the number
        of elements equalling the total number of edges.
    deduplicate: bool
        Boolean indicating whether seeds between hops will be deduplicated.
        If True, the same elements in seeds will be deleted to only one.
        Otherwise, the same elements will be remained.

    Examples
    -------
    >>> import dgl.graphbolt as gb
    >>> import torch
    >>> indptr = torch.LongTensor([0, 2, 4, 5, 6, 7 ,8])
    >>> indices = torch.LongTensor([1, 2, 0, 3, 5, 4, 3, 5])
    >>> graph = gb.fused_csc_sampling_graph(indptr, indices)
    >>> node_pairs = torch.LongTensor([[0, 1], [1, 2]])
    >>> item_set = gb.ItemSet(node_pairs, names="node_pairs")
    >>> item_sampler = gb.ItemSampler(item_set, batch_size=1,)
    >>> neg_sampler = gb.UniformNegativeSampler(item_sampler, graph, 2)
    >>> fanouts = [torch.LongTensor([5]),
    ...     torch.LongTensor([10]),torch.LongTensor([15])]
    >>> subgraph_sampler = gb.LayerNeighborSampler(neg_sampler, graph, fanouts)
    >>> next(iter(subgraph_sampler)).sampled_subgraphs
    [SampledSubgraphImpl(sampled_csc=CSCFormatBase(
            indptr=tensor([0, 2, 4, 5, 6, 7, 8]),
            indices=tensor([1, 3, 0, 4, 2, 2, 5, 4]),
        ),
        original_row_node_ids=tensor([0, 1, 5, 2, 3, 4]),
        original_edge_ids=None,
        original_column_node_ids=tensor([0, 1, 5, 2, 3, 4]),
    ),
    SampledSubgraphImpl(sampled_csc=CSCFormatBase(
            indptr=tensor([0, 2, 4, 5, 6, 7]),
            indices=tensor([1, 3, 0, 4, 2, 2, 5]),
        ),
        original_row_node_ids=tensor([0, 1, 5, 2, 3, 4]),
        original_edge_ids=None,
        original_column_node_ids=tensor([0, 1, 5, 2, 3]),
    ),
    SampledSubgraphImpl(sampled_csc=CSCFormatBase(
            indptr=tensor([0, 2, 4, 5, 6]),
            indices=tensor([1, 3, 0, 4, 2, 2]),
        ),
        original_row_node_ids=tensor([0, 1, 5, 2, 3]),
        original_edge_ids=None,
        original_column_node_ids=tensor([0, 1, 5, 2]),
    )]
    >>> next(iter(subgraph_sampler)).compacted_node_pairs
    (tensor([0]), tensor([1]))
    """

    def __init__(
        self,
        datapipe,
        graph,
        fanouts,
        replace=False,
        prob_name=None,
        deduplicate=True,
    ):
        super().__init__(
            datapipe,
            graph,
            fanouts,
            replace,
            prob_name,
            deduplicate,
        )
        self.sampler = graph.sample_layer_neighbors
