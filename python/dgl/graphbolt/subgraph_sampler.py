"""Subgraph samplers"""

from collections import defaultdict
from typing import Dict

from torchdata.datapipes.iter import Mapper

from .link_prediction_block import LinkPredictionBlock
from .node_classification_block import NodeClassificationBlock
from .utils import unique_and_compact


class SubgraphSampler(Mapper):
    """A subgraph sampler used to sample a subgraph from a given set of nodes
    from a larger graph."""

    def __init__(
        self,
        datapipe,
    ):
        """
        Initlization for a subgraph sampler.

        Parameters
        ----------
        datapipe : DataPipe
            The datapipe.
        """
        super().__init__(datapipe, self._sample)

    def _sample(self, data):
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
            raise TypeError(f"Unsupported type of data {data}.")
        data.input_nodes, data.sampled_subgraphs = self._sample_subgraphs(
            seeds
        )
        return data

    def _link_prediction_preprocess(self, data):
        node_pair = data.node_pair
        neg_src, neg_dst = data.negative_head, data.negative_tail
        has_neg_src = neg_src is not None
        has_neg_dst = neg_dst is not None
        is_heterogeneous = isinstance(node_pair, Dict)
        if is_heterogeneous:
            # Collect nodes from all types of input.
            nodes = defaultdict(list)
            for (src_type, _, dst_type), (src, dst) in node_pair.items():
                nodes[src_type].append(src)
                nodes[dst_type].append(dst)
            if has_neg_src:
                for (src_type, _, _), src in neg_src.items():
                    nodes[src_type].append(src.view(-1))
            if has_neg_dst:
                for (_, _, dst_type), dst in neg_dst.items():
                    nodes[dst_type].append(dst.view(-1))
            # Unique and compact the collected nodes.
            seeds, compacted = unique_and_compact(nodes)
            (
                compacted_node_pair,
                compacted_negative_head,
                compacted_negative_tail,
            ) = ({}, {}, {})
            # Map back in same order as collect.
            for etype, _ in node_pair.items():
                src_type, _, dst_type = etype
                src = compacted[src_type].pop(0)
                dst = compacted[dst_type].pop(0)
                compacted_node_pair[etype] = (src, dst)
            if has_neg_src:
                for etype, _ in neg_src.items():
                    compacted_negative_head[etype] = compacted[etype[0]].pop(0)
            if has_neg_dst:
                for etype, _ in neg_dst.items():
                    compacted_negative_tail[etype] = compacted[etype[2]].pop(0)
        else:
            # Collect nodes from all types of input.
            nodes = list(node_pair)
            if has_neg_src:
                nodes.append(neg_src.view(-1))
            if has_neg_dst:
                nodes.append(neg_dst.view(-1))
            # Unique and compact the collected nodes.
            seeds, compacted = unique_and_compact(nodes)
            # Map back in same order as collect.
            compacted_node_pair = tuple(compacted[:2])
            compacted = compacted[2:]
            if has_neg_src:
                compacted_negative_head = compacted.pop(0)
            if has_neg_dst:
                compacted_negative_tail = compacted.pop(0)
        return (
            seeds,
            compacted_node_pair,
            compacted_negative_head if has_neg_src else None,
            compacted_negative_tail if has_neg_dst else None,
        )

    def _sample_subgraphs(self, seeds):
        raise NotImplementedError
