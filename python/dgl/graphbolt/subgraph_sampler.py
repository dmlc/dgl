"""Subgraph samplers"""

import torch
from torchdata.datapipes.iter import Mapper

from .impl import SampledSubgraphImpl

from .link_unified_data_struct import LinkUnifiedDataStruct
from .node_unified_data_struct import NodeUnifiedDataStruct
from .utils import unique_and_compact_node_pairs


class SubgraphSampler(Mapper):
    """A subgraph sampler.

    It is an iterator equivalent to the following:

    .. code:: python

       for data in datapipe:
           yield _sample(data)

    Parameters
    ----------
    datapipe : DataPipe
        The datapipe.
    fn : callable
        The subgraph sampling function.
    """

    def __init__(
        self,
        datapipe,
        fanouts,
        replace=False,
        prob_name=None,
    ):
        """
        Initlization for a subgraph sampler.

        Parameters
        ----------
        datapipe : DataPipe
            The datapipe.
        fanouts: list[list[int]]
            The number of edges to be sampled for each node with or without
            considering edge types.
        replace: bool
            Boolean indicating whether the sample is preformed with or
            without replacement. If True, a value can be selected multiple
            times. Otherwise, each value can be selected only once.
        prob_name: str, optional
            The name of an edge attribute used a. This
            attribute tensor should contain (unnormalized) probabilities
            corresponding to each neighboring edge of a node. It must be a 1D
            floating-point or boolean tensor, with the number of elements
            equalling the total number of edges.
        """
        super().__init__(datapipe, self._sample)
        self.fanouts = fanouts
        self.replace = replace
        self.prob_name = prob_name

    def _sample(self, data):
        subgraphs = []
        num_layers = len(self.fanouts)
        data = self._preprocess(data)
        seeds = data.seed_node
        for hop in range(num_layers):
            subgraph = self._sample_sub_graph(
                seeds,
                hop,
            )
            seeds, compacted_node_pairs = unique_and_compact_node_pairs(
                subgraph.node_pairs, seeds
            )
            subgraph = SampledSubgraphImpl(
                node_pairs=compacted_node_pairs,
                reverse_column_node_ids=seeds,
                reverse_row_node_ids=seeds,
            )
            subgraphs.insert(0, subgraph)
        data.input_nodes = seeds
        data.sampled_subgraphs = subgraphs
        return data

    def _preprocess(self, data):
        if isinstance(data, LinkUnifiedDataStruct):
            src, dst = data.node_pair
            neg_src, neg_dst = data.negative_head, data.negative_tail

            def combine_pos_and_neg(pos, neg):
                if isinstance(pos, torch.Tensor):
                    return torch.cat((pos, neg.view(-1)))
                else:
                    return {
                        etype: torch.cat((nodes, neg[etype].view(-1)))
                        for etype, nodes in pos.items()
                    }

            src = (
                combine_pos_and_neg(src, neg_src)
                if neg_src is not None
                else src
            )
            dst = (
                combine_pos_and_neg(dst, neg_dst)
                if neg_dst is not None
                else dst
            )
            seeds, compacted_pairs = unique_and_compact_node_pairs((src, dst))
            data.seed_node = seeds
            data.node_pair = compacted_pairs
            return data
        elif isinstance(data, NodeUnifiedDataStruct):
            pass
        else:
            raise TypeError(f"Unkown input data {data}.")

    def _sample_sub_graph(self, seeds, hop):
        raise NotImplemented
