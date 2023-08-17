"""Subgraph samplers"""

from collections import defaultdict
from typing import Dict

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
                data.compacted_node_pair,
                data.compacted_negative_head,
                data.compacted_negative_tail,
            ) = self._link_prediction_preprocess(data)
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
        node_pair = data.node_pair
        neg_src, neg_dst = data.negative_head, data.negative_tail
        is_heterogeneous = isinstance(node_pair, Dict)
        has_neg_src = neg_src is not None
        has_neg_dst = neg_dst is not None

        def merge_node_pairs(src, dst, neg_src, neg_dst):
            if has_neg_src:
                src = torch.cat([src, neg_src.view(-1)])
            if has_neg_dst:
                dst = torch.cat([dst, neg_dst.view(-1)])
            return (src, dst)

        # Merge postive graph and negative node pairs.
        if is_heterogeneous:
            merged_node_pair = {}
            for etype, pair in node_pair.items():
                neg_src_etype = neg_src[etype] if has_neg_src else None
                neg_dst_etype = neg_dst[etype] if has_neg_dst else None
                merged_node_pair[etype] = merge_node_pairs(
                    pair[0], pair[1], neg_src_etype, neg_dst_etype
                )
        else:
            merged_node_pair = merge_node_pairs(
                node_pair[0], node_pair[1], neg_src, neg_dst
            )

        # Compacct merged and get seed nodes for sampling.
        seeds, compacted_merged_node_pair = unique_and_compact_node_pairs(
            merged_node_pair
        )

        def split_node_pairs(src, dst, num_pos_src, num_pos_dst):
            pos_src, neg_src = src[:num_pos_src], src[num_pos_dst:]
            pos_dst, neg_dst = dst[:num_pos_dst], dst[num_pos_dst:]
            return (pos_src, pos_dst), neg_src, neg_dst

        compacted_node_pair = {}
        compacted_negative_head = {}
        compacted_negative_tail = {}

        # Split positive and negative node pairs.
        if is_heterogeneous:
            for etype, (src, dst) in compacted_merged_node_pair.items():
                num_pos_src = node_pair[etype][0].size(0)
                num_pos_dst = node_pair[etype][1].size(0)
                (
                    compacted_node_pair[etype],
                    compacted_negative_head[etype],
                    compacted_negative_tail[etype],
                ) = split_node_pairs(src, dst, num_pos_src, num_pos_dst)
        else:
            src, dst = compacted_merged_node_pair
            num_pos_src = node_pair[0].size(0)
            num_pos_dst = node_pair[1].size(0)
            (
                compacted_node_pair,
                compacted_negative_head,
                compacted_negative_tail,
            ) = split_node_pairs(src, dst, num_pos_src, num_pos_dst)

        return (
            seeds,
            compacted_node_pair,
            compacted_negative_head if has_neg_src else None,
            compacted_negative_tail if has_neg_dst else None,
        )

    def _sample_sub_graph(self, seeds, hop):
        raise NotImplemented
