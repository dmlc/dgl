"""Subgraph samplers"""

from collections import defaultdict
from typing import Dict

import torch
from torch.utils.data import functional_datapipe

from .base import etype_str_to_tuple
from .internal import compact_temporal_nodes, unique_and_compact
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
    SubgraphSampler should implement either the :meth:`sample_subgraphs` method
    or the :meth:`sampling_stages` method to define the fine-grained sampling
    stages to take advantage of optimizations provided by the GraphBolt
    DataLoader.

    Parameters
    ----------
    datapipe : DataPipe
        The datapipe.
    args : Non-Keyword Arguments
        Arguments to be passed into sampling_stages.
    kwargs : Keyword Arguments
        Arguments to be passed into sampling_stages.
    """

    def __init__(
        self,
        datapipe,
        *args,
        **kwargs,
    ):
        datapipe = datapipe.transform(self._preprocess)
        datapipe = self.sampling_stages(datapipe, *args, **kwargs)
        datapipe = datapipe.transform(self._postprocess)
        super().__init__(datapipe)

    @staticmethod
    def _postprocess(minibatch):
        delattr(minibatch, "_seed_nodes")
        delattr(minibatch, "_seeds_timestamp")
        return minibatch

    @staticmethod
    def _preprocess(minibatch):
        if minibatch.node_pairs is not None:
            (
                seeds,
                seeds_timestamp,
                minibatch.compacted_node_pairs,
                minibatch.compacted_negative_srcs,
                minibatch.compacted_negative_dsts,
            ) = SubgraphSampler._node_pairs_preprocess(minibatch)
        elif minibatch.seed_nodes is not None:
            seeds = minibatch.seed_nodes
            seeds_timestamp = (
                minibatch.timestamp if hasattr(minibatch, "timestamp") else None
            )
        elif minibatch.seeds is not None:
            (
                seeds,
                seeds_timestamp,
                minibatch.compacted_seeds,
            ) = SubgraphSampler._seeds_preprocess(minibatch)
        else:
            raise ValueError(
                f"Invalid minibatch {minibatch}: One of `node_pairs`, "
                "`seed_nodes` and `seeds` should have a value."
            )
        minibatch._seed_nodes = seeds
        minibatch._seeds_timestamp = seeds_timestamp
        return minibatch

    @staticmethod
    def _node_pairs_preprocess(minibatch):
        use_timestamp = hasattr(minibatch, "timestamp")
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
            nodes_timestamp = None
            if use_timestamp:
                nodes_timestamp = defaultdict(list)
            for etype, (src, dst) in node_pairs.items():
                src_type, _, dst_type = etype_str_to_tuple(etype)
                nodes[src_type].append(src)
                nodes[dst_type].append(dst)
                if use_timestamp:
                    nodes_timestamp[src_type].append(minibatch.timestamp[etype])
                    nodes_timestamp[dst_type].append(minibatch.timestamp[etype])
            if has_neg_src:
                for etype, src in neg_src.items():
                    src_type, _, _ = etype_str_to_tuple(etype)
                    nodes[src_type].append(src.view(-1))
                    if use_timestamp:
                        nodes_timestamp[src_type].append(
                            minibatch.timestamp[etype].repeat_interleave(
                                src.shape[-1]
                            )
                        )
            if has_neg_dst:
                for etype, dst in neg_dst.items():
                    _, _, dst_type = etype_str_to_tuple(etype)
                    nodes[dst_type].append(dst.view(-1))
                    if use_timestamp:
                        nodes_timestamp[dst_type].append(
                            minibatch.timestamp[etype].repeat_interleave(
                                dst.shape[-1]
                            )
                        )
            # Unique and compact the collected nodes.
            if use_timestamp:
                seeds, nodes_timestamp, compacted = compact_temporal_nodes(
                    nodes, nodes_timestamp
                )
            else:
                seeds, compacted = unique_and_compact(nodes)
                nodes_timestamp = None
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
                    compacted_negative_srcs[etype] = compacted_negative_srcs[
                        etype
                    ].view(neg_src[etype].shape)
            if has_neg_dst:
                for etype, _ in neg_dst.items():
                    _, _, dst_type = etype_str_to_tuple(etype)
                    compacted_negative_dsts[etype] = compacted[dst_type].pop(0)
                    compacted_negative_dsts[etype] = compacted_negative_dsts[
                        etype
                    ].view(neg_dst[etype].shape)
        else:
            # Collect nodes from all types of input.
            nodes = list(node_pairs)
            nodes_timestamp = None
            if use_timestamp:
                # Timestamp for source and destination nodes are the same.
                nodes_timestamp = [minibatch.timestamp, minibatch.timestamp]
            if has_neg_src:
                nodes.append(neg_src.view(-1))
                if use_timestamp:
                    nodes_timestamp.append(
                        minibatch.timestamp.repeat_interleave(neg_src.shape[-1])
                    )
            if has_neg_dst:
                nodes.append(neg_dst.view(-1))
                if use_timestamp:
                    nodes_timestamp.append(
                        minibatch.timestamp.repeat_interleave(neg_dst.shape[-1])
                    )
            # Unique and compact the collected nodes.
            if use_timestamp:
                seeds, nodes_timestamp, compacted = compact_temporal_nodes(
                    nodes, nodes_timestamp
                )
            else:
                seeds, compacted = unique_and_compact(nodes)
                nodes_timestamp = None
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
            nodes_timestamp,
            compacted_node_pairs,
            compacted_negative_srcs if has_neg_src else None,
            compacted_negative_dsts if has_neg_dst else None,
        )

    def _sample(self, minibatch):
        (
            minibatch.input_nodes,
            minibatch.sampled_subgraphs,
        ) = self.sample_subgraphs(
            minibatch._seed_nodes, minibatch._seeds_timestamp
        )
        return minibatch

    def sampling_stages(self, datapipe):
        """The sampling stages are defined here by chaining to the datapipe. The
        default implementation expects :meth:`sample_subgraphs` to be
        implemented. To define fine-grained stages, this method should be
        overridden.
        """
        return datapipe.transform(self._sample)

    @staticmethod
    def _seeds_preprocess(minibatch):
        """Preprocess `seeds` in a minibatch to construct `unique_seeds`,
        `node_timestamp` and `compacted_seeds` for further sampling. It
        optionally incorporates timestamps for temporal graphs, organizing and
        compacting seeds based on their types and timestamps.

        Parameters
        ----------
        minibatch: MiniBatch
            The minibatch.

        Returns
        -------
        unique_seeds: torch.Tensor or Dict[str, torch.Tensor]
            A tensor or a dictionary of tensors representing the unique seeds.
            In heterogeneous graphs, seeds are returned for each node type.
        nodes_timestamp: None or a torch.Tensor or Dict[str, torch.Tensor]
            Containing timestamps for each seed. This is only returned if
            `minibatch` includes timestamps and the graph is temporal.
        compacted_seeds: torch.tensor or a Dict[str, torch.Tensor]
            Representation of compacted seeds corresponding to 'seeds', where
            all node ids inside are compacted.
        """
        use_timestamp = hasattr(minibatch, "timestamp")
        seeds = minibatch.seeds
        is_heterogeneous = isinstance(seeds, Dict)
        if is_heterogeneous:
            # Collect nodes from all types of input.
            nodes = defaultdict(list)
            nodes_timestamp = None
            if use_timestamp:
                nodes_timestamp = defaultdict(list)
            for etype, typed_seeds in seeds.items():
                # When typed_seeds is a one-dimensional tensor, it represents
                # seed nodes, which does not need to do unique and compact.
                if typed_seeds.ndim == 1:
                    nodes_timestamp = (
                        minibatch.timestamp
                        if hasattr(minibatch, "timestamp")
                        else None
                    )
                    return seeds, nodes_timestamp, None
                assert typed_seeds.ndim == 2 and typed_seeds.shape[1] == 2, (
                    "Only tensor with shape 1*N and N*2 is "
                    + f"supported now, but got {typed_seeds.shape}."
                )
                ntypes = etype[:].split(":")[::2]
                if use_timestamp:
                    negative_ratio = (
                        typed_seeds.shape[0]
                        // minibatch.timestamp[etype].shape[0]
                        - 1
                    )
                    neg_timestamp = minibatch.timestamp[
                        etype
                    ].repeat_interleave(negative_ratio)
                for i, ntype in enumerate(ntypes):
                    nodes[ntype].append(typed_seeds[:, i])
                    if use_timestamp:
                        nodes_timestamp[ntype].append(
                            minibatch.timestamp[etype]
                        )
                        nodes_timestamp[ntype].append(neg_timestamp)
            # Unique and compact the collected nodes.
            if use_timestamp:
                (
                    unique_seeds,
                    nodes_timestamp,
                    compacted,
                ) = compact_temporal_nodes(nodes, nodes_timestamp)
            else:
                unique_seeds, compacted = unique_and_compact(nodes)
                nodes_timestamp = None
            compacted_seeds = {}
            # Map back in same order as collect.
            for etype, typed_seeds in seeds.items():
                src_type, _, dst_type = etype_str_to_tuple(etype)
                src = compacted[src_type].pop(0)
                dst = compacted[dst_type].pop(0)
                compacted_seeds[etype] = torch.cat((src, dst)).view(2, -1).T
        else:
            # When seeds is a one-dimensional tensor, it represents seed nodes,
            # which does not need to do unique and compact.
            if seeds.ndim == 1:
                nodes_timestamp = (
                    minibatch.timestamp
                    if hasattr(minibatch, "timestamp")
                    else None
                )
                return seeds, nodes_timestamp, None
            # Collect nodes from all types of input.
            nodes = [seeds.view(-1)]
            nodes_timestamp = None
            if use_timestamp:
                # Timestamp for source and destination nodes are the same.
                negative_ratio = (
                    seeds.shape[0] // minibatch.timestamp.shape[0] - 1
                )
                neg_timestamp = minibatch.timestamp.repeat_interleave(
                    negative_ratio
                )
                seeds_timestamp = torch.cat(
                    (minibatch.timestamp, neg_timestamp)
                )
                nodes_timestamp = [seeds_timestamp for _ in range(seeds.ndim)]
            # Unique and compact the collected nodes.
            if use_timestamp:
                (
                    unique_seeds,
                    nodes_timestamp,
                    compacted,
                ) = compact_temporal_nodes(nodes, nodes_timestamp)
            else:
                unique_seeds, compacted = unique_and_compact(nodes)
                nodes_timestamp = None
            # Map back in same order as collect.
            compacted_seeds = compacted[0].view(seeds.shape)
        return (
            unique_seeds,
            nodes_timestamp,
            compacted_seeds,
        )

    def sample_subgraphs(self, seeds, seeds_timestamp):
        """Sample subgraphs from the given seeds, possibly with temporal constraints.

        Any subclass of SubgraphSampler should implement this method.

        Parameters
        ----------
        seeds : Union[torch.Tensor, Dict[str, torch.Tensor]]
            The seed nodes.

        seeds_timestamp : Union[torch.Tensor, Dict[str, torch.Tensor]]
            The timestamps of the seed nodes. If given, the sampled subgraphs
            should not contain any nodes or edges that are newer than the
            timestamps of the seed nodes. Default: None.

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
