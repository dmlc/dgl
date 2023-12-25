"""In-subgraph sampler for GraphBolt."""

from torch.utils.data import functional_datapipe

from ..internal import (
    unique_and_compact_csc_formats,
    unique_and_compact_node_pairs,
)

from ..subgraph_sampler import SubgraphSampler
from .sampled_subgraph_impl import FusedSampledSubgraphImpl, SampledSubgraphImpl


__all__ = ["InSubgraphSampler"]


@functional_datapipe("sample_in_subgraph")
class InSubgraphSampler(SubgraphSampler):
    """Sample the subgraph induced on the inbound edges of the given nodes.

    Functional name: :obj:`sample_in_subgraph`.

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
    >>> graph = gb.fused_csc_sampling_graph(indptr, indices)
    >>> item_set = gb.ItemSet(len(indptr) - 1, names="seed_nodes")
    >>> item_sampler = gb.ItemSampler(item_set, batch_size=2)
    >>> insubgraph_sampler = gb.InSubgraphSampler(item_sampler, graph)
    >>> for _, data in enumerate(insubgraph_sampler):
    ...     print(data.sampled_subgraphs[0].node_pairs)
    ...     print(data.sampled_subgraphs[0].original_row_node_ids)
    ...     print(data.sampled_subgraphs[0].original_column_node_ids)
    CSCFormatBase(indptr=tensor([0, 3, 5]),
                indices=tensor([0, 1, 2, 3, 4]),
    )
    tensor([0, 1, 4, 2, 3])
    tensor([0, 1])
    CSCFormatBase(indptr=tensor([0, 2, 4]),
                indices=tensor([2, 3, 4, 0]),
    )
    tensor([2, 3, 0, 5, 1])
    tensor([2, 3])
    CSCFormatBase(indptr=tensor([0, 3, 5]),
                indices=tensor([2, 3, 1, 4, 0]),
    )
    tensor([4, 5, 0, 3, 1])
    tensor([4, 5])
    """

    def __init__(
        self,
        datapipe,
        graph,
        # TODO: clean up once the migration is done.
        output_cscformat=True,
    ):
        super().__init__(datapipe)
        self.graph = graph
        self.output_cscformat = output_cscformat
        self.sampler = graph.in_subgraph

    def sample_subgraphs(self, seeds):
        subgraph = self.sampler(seeds, self.output_cscformat)
        if not self.output_cscformat:
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
        else:
            (
                original_row_node_ids,
                compacted_csc_formats,
            ) = unique_and_compact_csc_formats(subgraph.node_pairs, seeds)
            subgraph = SampledSubgraphImpl(
                node_pairs=compacted_csc_formats,
                original_column_node_ids=seeds,
                original_row_node_ids=original_row_node_ids,
                original_edge_ids=subgraph.original_edge_ids,
            )
        seeds = original_row_node_ids
        return (seeds, [subgraph])
