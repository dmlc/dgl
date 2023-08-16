"""Subgraph samplers"""

import torch
from torchdata.datapipes.iter import Mapper

from .impl import SampledSubgraphImpl

from .link_prediction_block import LinkPredictionBlock
from .node_classification_block import NodeClassificationBlock
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
        if isinstance(data, LinkPredictionBlock):
            (
                seeds,
                compacted_pairs,
                compacted_negative_head,
                compacted_negative_tail,
            ) = self._link_prediction_preprocess(data)
            data.seed_node = seeds
            data.compacted_node_pair = compacted_pairs
            data.compacted_negative_head = compacted_negative_head
            data.compacted_negative_tail = compacted_negative_tail
        elif isinstance(data, NodeClassificationBlock):
            seeds = data.seed_node
        else:
            raise TypeError(f"Unsupported data {data}.")
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

    def _link_prediction_preprocess(self, data):
        src, dst = data.node_pair
        neg_src, neg_dst = data.negative_head, data.negative_tail

        is_homogeneous = isinstance(src, torch.Tensor)

        def combine_pos_and_neg(pos, neg):
            if is_homogeneous:
                return torch.cat((pos, neg.view(-1)))
            else:
                return {
                    etype: torch.cat((nodes, neg[etype].view(-1)))
                    for etype, nodes in pos.items()
                }

        src = combine_pos_and_neg(src, neg_src) if neg_src is not None else src
        dst = combine_pos_and_neg(dst, neg_dst) if neg_dst is not None else dst
        seeds, compacted_node_pair = unique_and_compact_node_pairs((src, dst))

        def split_pos_and_neg(nodes, num_pos, neg_shape):
            return nodes[:num_pos], nodes[num_pos:].reshape(neg_shape)

        compacted_negative_head = {} if neg_src is not None else None
        compacted_negative_tail = {} if neg_dst is not None else None

        if is_homogeneous:
            if neg_src is not None:
                num_pos = src.size(0)
                (
                    compacted_node_pair[0],
                    compacted_negative_head,
                ) = split_pos_and_neg(
                    compacted_node_pair[0], num_pos, neg_src.shape
                )
            if neg_dst is not None:
                num_pos = dst.size(0)
                (
                    compacted_node_pair[1],
                    compacted_negative_tail,
                ) = split_pos_and_neg(
                    compacted_node_pair[1], num_pos, neg_dst.shape
                )
        else:
            for etype, (src, dst) in compacted_node_pair.items():
                num_pos = src.size(0)

                if neg_src is not None:
                    shape = neg_src[etype].shape
                    (
                        compacted_node_pair[etype][0],
                        compacted_negative_head[etype],
                    ) = split_pos_and_neg(src, num_pos, shape)

                if neg_dst is not None:
                    shape = neg_dst[etype].shape
                    (
                        compacted_node_pair[etype][1],
                        compacted_negative_tail[etype],
                    ) = split_pos_and_neg(dst, num_pos, shape)
        return (
            seeds,
            compacted_node_pair,
            compacted_negative_head,
            compacted_negative_tail,
        )

    def _sample_sub_graph(self, seeds, hop):
        raise NotImplemented
