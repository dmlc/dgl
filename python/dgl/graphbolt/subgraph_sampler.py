"""Subgraph samplers"""

from collections import defaultdict
from typing import Dict

from torch.utils.data import functional_datapipe

from .base import etype_str_to_tuple
from .internal import unique_and_compact
from .minibatch_transformer import MiniBatchTransformer

__all__ = [
    "SubgraphSampler",
]


@functional_datapipe("sample_subgraph")
class SubgraphSampler(MiniBatchTransformer):
    """A subgraph sampler used to sample a subgraph from a given set of nodes
    from a larger graph.

    Functional name: :obj:`sample_subgraph`.

    This class is the base class of all subgraph samplers. Any subclass of
    SubgraphSampler should implement the :meth:`sample_subgraphs` method.

    Parameters
    ----------
    datapipe : DataPipe
        The datapipe.
    """

    def __init__(
        self,
        datapipe,
    ):
        super().__init__(datapipe, self._sample)

    def _sample(self, minibatch):
        if minibatch.node_pairs is not None:
            (
                seeds,
                minibatch.compacted_node_pairs,
                minibatch.compacted_negative_srcs,
                minibatch.compacted_negative_dsts,
            ) = self._node_pairs_preprocess(minibatch)
        elif minibatch.seed_nodes is not None:
            seeds = minibatch.seed_nodes
        else:
            raise ValueError(
                f"Invalid minibatch {minibatch}: Either `node_pairs` or "
                "`seed_nodes` should have a value."
            )
        (
            minibatch.input_nodes,
            minibatch.sampled_subgraphs,
        ) = self.sample_subgraphs(seeds)
        return minibatch

    def _node_pairs_preprocess(self, minibatch):
        node_pairs = minibatch.node_pairs
        neg_src, neg_dst = minibatch.negative_srcs, minibatch.negative_dsts
        has_neg_src = neg_src is not None
        has_neg_dst = neg_dst is not None
        is_heterogeneous = isinstance(node_pairs, Dict)
        if is_heterogeneous:
            has_neg_src = has_neg_src and all(
                item is not None for item in neg_src.values()
            )
            has_neg_dst = has_neg_dst and all(
                item is not None for item in neg_dst.values()
            )
            # Collect nodes from all types of input.
            nodes = defaultdict(list)
            for etype, (src, dst) in node_pairs.items():
                src_type, _, dst_type = etype_str_to_tuple(etype)
                nodes[src_type].append(src)
                nodes[dst_type].append(dst)
            if has_neg_src:
                for etype, src in neg_src.items():
                    src_type, _, _ = etype_str_to_tuple(etype)
                    nodes[src_type].append(src.view(-1))
            if has_neg_dst:
                for etype, dst in neg_dst.items():
                    _, _, dst_type = etype_str_to_tuple(etype)
                    nodes[dst_type].append(dst.view(-1))
            # Unique and compact the collected nodes.
            seeds, compacted = unique_and_compact(nodes)
            (
                compacted_node_pairs,
                compacted_negative_srcs,
                compacted_negative_dsts,
            ) = ({}, {}, {})
            # Map back in same order as collect.
            for etype, _ in node_pairs.items():
                src_type, _, dst_type = etype_str_to_tuple(etype)
                src = compacted[src_type].pop(0)
                dst = compacted[dst_type].pop(0)
                compacted_node_pairs[etype] = (src, dst)
            if has_neg_src:
                for etype, _ in neg_src.items():
                    src_type, _, _ = etype_str_to_tuple(etype)
                    compacted_negative_srcs[etype] = compacted[src_type].pop(0)
            if has_neg_dst:
                for etype, _ in neg_dst.items():
                    _, _, dst_type = etype_str_to_tuple(etype)
                    compacted_negative_dsts[etype] = compacted[dst_type].pop(0)
        else:
            # Collect nodes from all types of input.
            nodes = list(node_pairs)
            if has_neg_src:
                nodes.append(neg_src.view(-1))
            if has_neg_dst:
                nodes.append(neg_dst.view(-1))
            # Unique and compact the collected nodes.
            seeds, compacted = unique_and_compact(nodes)
            # Map back in same order as collect.
            compacted_node_pairs = tuple(compacted[:2])
            compacted = compacted[2:]
            if has_neg_src:
                compacted_negative_srcs = compacted.pop(0)
                # Since we need to calculate the neg_ratio according to the
                # compacted_negatvie_srcs shape, we need to reshape it back.
                compacted_negative_srcs = compacted_negative_srcs.view(
                    neg_src.shape
                )
            if has_neg_dst:
                compacted_negative_dsts = compacted.pop(0)
                # Same as above.
                compacted_negative_dsts = compacted_negative_dsts.view(
                    neg_dst.shape
                )
        return (
            seeds,
            compacted_node_pairs,
            compacted_negative_srcs if has_neg_src else None,
            compacted_negative_dsts if has_neg_dst else None,
        )

    def sample_subgraphs(self, seeds, seeds_timestamp=None):
        """Sample subgraphs from the given seeds.

        Any subclass of SubgraphSampler should implement this method.

        Parameters
        ----------
        seeds : Union[torch.Tensor, Dict[str, torch.Tensor]]
            The seed nodes.

        Returns
        -------
        Union[torch.Tensor, Dict[str, torch.Tensor]]
            The input nodes.
        List[SampledSubgraph]
            The sampled subgraphs.

        Examples
        --------
        >>> @functional_datapipe("my_sample_subgraph")
        >>> class MySubgraphSampler(SubgraphSampler):
        >>>     def __init__(self, datapipe, graph, fanouts):
        >>>         super().__init__(datapipe)
        >>>         self.graph = graph
        >>>         self.fanouts = fanouts
        >>>     def sample_subgraphs(self, seeds):
        >>>         # Sample subgraphs from the given seeds.
        >>>         subgraphs = []
        >>>         subgraphs_nodes = []
        >>>         for fanout in reversed(self.fanouts):
        >>>             subgraph = self.graph.sample_neighbors(seeds, fanout)
        >>>             subgraphs.insert(0, subgraph)
        >>>             subgraphs_nodes.append(subgraph.nodes)
        >>>             seeds = subgraph.nodes
        >>>         subgraphs_nodes = torch.unique(torch.cat(subgraphs_nodes))
        >>>         return subgraphs_nodes, subgraphs
        """
        raise NotImplementedError
