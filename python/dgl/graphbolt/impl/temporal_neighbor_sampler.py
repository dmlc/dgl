"""Temporal neighbor subgraph samplers for GraphBolt."""
import torch
from torch.utils.data import functional_datapipe

from ..internal import compact_csc_format

from ..subgraph_sampler import SubgraphSampler
from .sampled_subgraph_impl import SampledSubgraphImpl


__all__ = ["TemporalNeighborSampler"]


@functional_datapipe("temporal_sample_neighbor")
class TemporalNeighborSampler(SubgraphSampler):
    """Temporally sample neighbor edges from a graph and return sampled
    subgraphs.

    Functional name: :obj:`temporal_sample_neighbor`.

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
    node_timestamp_attr_name: str, optional
        The name of an node attribute used as the timestamps of nodes.
        It must be a 1D integer tensor, with the number of elements
        equalling the total number of nodes.
    edge_timestamp_attr_name: str, optional
        The name of an edge attribute used as the timestamps of edges.
        It must be a 1D integer tensor, with the number of elements
        equalling the total number of edges.

    Examples
    -------
    TODO(zhenkun) : Add an example after the API to pass timestamps is finalized.
    """

    def __init__(
        self,
        datapipe,
        graph,
        fanouts,
        replace=False,
        prob_name=None,
        node_timestamp_attr_name=None,
        edge_timestamp_attr_name=None,
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
        self.node_timestamp_attr_name = node_timestamp_attr_name
        self.edge_timestamp_attr_name = edge_timestamp_attr_name
        self.sampler = graph.temporal_sample_neighbors

    def sample_subgraphs(self, seeds, seeds_timestamp):
        assert (
            seeds_timestamp is not None
        ), "seeds_timestamp must be provided for temporal neighbor sampling."
        subgraphs = []
        num_layers = len(self.fanouts)
        # Enrich seeds with all node types.
        if isinstance(seeds, dict):
            ntypes = list(self.graph.node_type_to_id.keys())
            seeds = {
                ntype: seeds.get(ntype, torch.LongTensor([]))
                for ntype in ntypes
            }
            seeds_timestamp = {
                ntype: seeds_timestamp.get(ntype, torch.LongTensor([]))
                for ntype in ntypes
            }
        for hop in range(num_layers):
            subgraph = self.sampler(
                seeds,
                seeds_timestamp,
                self.fanouts[hop],
                self.replace,
                self.prob_name,
                self.node_timestamp_attr_name,
                self.edge_timestamp_attr_name,
            )
            (
                original_row_node_ids,
                compacted_csc_formats,
                row_timestamps,
            ) = compact_csc_format(subgraph.sampled_csc, seeds, seeds_timestamp)

            subgraph = SampledSubgraphImpl(
                sampled_csc=compacted_csc_formats,
                original_column_node_ids=seeds,
                original_row_node_ids=original_row_node_ids,
                original_edge_ids=subgraph.original_edge_ids,
            )

            subgraphs.insert(0, subgraph)
            seeds = original_row_node_ids
            seeds_timestamp = row_timestamps
        return seeds, subgraphs
