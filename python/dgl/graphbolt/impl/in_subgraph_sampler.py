"""In-subgraph sampler for GraphBolt."""

from torch.utils.data import functional_datapipe

from ..subgraph_sampler import SubgraphSampler
from ..utils import unique_and_compact_node_pairs
from .sampled_subgraph_impl import FusedSampledSubgraphImpl


__all__ = ["InSubgraphSampler"]


@functional_datapipe("sample_in_subgraph")
class InSubgraphSampler(SubgraphSampler):
    """Sample neighbor edges from a graph and return a subgraph.

    In subgraph sampler is responsible for sampling a subgraph from given data,
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
    >>> from dgl import graphbolt as gb
    >>> indptr = torch.LongTensor([0, 2, 4, 5, 6, 7 ,8])
    >>> indices = torch.LongTensor([1, 2, 0, 3, 5, 4, 3, 5])
    >>> graph = gb.from_fused_csc(indptr, indices)
    >>> node_pairs = torch.LongTensor([[0, 1], [1, 2]])
    >>> item_set = gb.ItemSet(node_pairs, names="node_pairs")
    >>> item_sampler = gb.ItemSampler(
        ...item_set, batch_size=1,
        ...)
    >>> neg_sampler = gb.UniformNegativeSampler(
        ...item_sampler, graph, 2)
    >>> subgraph_sampler = gb.NeighborSampler(
        ...neg_sampler, graph, [5, 10, 15])
    >>> for data in subgraph_sampler:
        ... print(data.compacted_node_pairs)
        ... print(len(data.sampled_subgraphs))
    (tensor([0, 0, 0]), tensor([1, 0, 2]))
    3
    (tensor([0, 0, 0]), tensor([1, 1, 1]))
    3
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
