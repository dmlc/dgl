"""In-subgraph sampler for GraphBolt."""

from torch.utils.data import functional_datapipe

from ..internal import unique_and_compact_csc_formats

from ..subgraph_sampler import SubgraphSampler
from .sampled_subgraph_impl import SampledSubgraphImpl


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
    >>> item_set = gb.ItemSet(len(indptr) - 1, names="seeds")
    >>> item_sampler = gb.ItemSampler(item_set, batch_size=2)
    >>> insubgraph_sampler = gb.InSubgraphSampler(item_sampler, graph)
    >>> for _, data in enumerate(insubgraph_sampler):
    ...     print(data.sampled_subgraphs[0].sampled_csc)
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
    ):
        super().__init__(datapipe)
        self.graph = graph
        self.sampler = graph.in_subgraph

    def sample_subgraphs(
        self, seeds, seeds_timestamp, seeds_pre_time_window=None
    ):
        subgraph = self.sampler(seeds)
        (
            original_row_node_ids,
            compacted_csc_formats,
            _,
        ) = unique_and_compact_csc_formats(subgraph.sampled_csc, seeds)
        subgraph = SampledSubgraphImpl(
            sampled_csc=compacted_csc_formats,
            original_column_node_ids=seeds,
            original_row_node_ids=original_row_node_ids,
            original_edge_ids=subgraph.original_edge_ids,
        )
        seeds = original_row_node_ids
        return (seeds, [subgraph])
