"""In-subgraph sampler for GraphBolt."""

from torch.utils.data import functional_datapipe

from ..internal import unique_and_compact_node_pairs

from ..subgraph_sampler import SubgraphSampler
from .sampled_subgraph_impl import FusedSampledSubgraphImpl


__all__ = ["InSubgraphSampler"]


@functional_datapipe("sample_in_subgraph")
class InSubgraphSampler(SubgraphSampler):
    """Sample the subgraph induced on the inbound edges of the given nodes.

    In-subgraph sampler is responsible for sampling a subgraph from given data,
    returning an induced subgraph along with compacted information.

    Parameters
    ----------
    datapipe : DataPipe
        The datapipe.
    graph : FusedCSCSamplingGraph
        The graph on which to perform in_subgraph sampling.

    Examples
    -------
    >>> import dgl.graphbolt as gb
    >>> import torch
    >>> indptr = torch.LongTensor([0, 3, 5, 7, 9, 12, 14])
    >>> indices = torch.LongTensor([0, 1, 4, 2, 3, 0, 5, 1, 2, 0, 3, 5, 1, 4])
    >>> graph = gb.from_fused_csc(indptr, indices)
    >>> item_set = gb.ItemSet(len(indptr) - 1, names="seed_nodes")
    >>> item_sampler = gb.ItemSampler(item_set, batch_size=2)
    >>> insubgraph_sampler = gb.InSubgraphSampler(item_sampler, graph)
    >>> for _, data in enumerate(insubgraph_sampler):
    ...     print(data.sampled_subgraphs[0].node_pairs)
    ...     print(data.sampled_subgraphs[0].original_row_node_ids)
    ...     print(data.sampled_subgraphs[0].original_column_node_ids)
    (tensor([0, 1, 2, 3, 4]), tensor([0, 0, 0, 1, 1]))
    tensor([0, 1, 4, 2, 3])
    tensor([0, 1])
    (tensor([2, 3, 4, 0]), tensor([0, 0, 1, 1]))
    tensor([2, 3, 0, 5, 1])
    tensor([2, 3])
    (tensor([2, 3, 1, 4, 0]), tensor([0, 0, 0, 1, 1]))
    tensor([4, 5, 0, 3, 1])
    tensor([4, 5])
    """

    def __init__(
        self,
        datapipe,
        graph,
    ):
        super().__init__(datapipe)
        self.graph = graph
        self.sampler = graph.in_subgraph

    def _sample_subgraphs(self, seeds):
        subgraph = self.sampler(seeds)
        (
            original_row_node_ids,
            compacted_node_pairs,
        ) = unique_and_compact_node_pairs(subgraph.node_pairs, seeds)
        subgraph = FusedSampledSubgraphImpl(
            node_pairs=compacted_node_pairs,
            original_column_node_ids=seeds,
            original_row_node_ids=original_row_node_ids,
            original_edge_ids=subgraph.original_edge_ids,
        )
        seeds = original_row_node_ids
        return (seeds, [subgraph])
