"""Subgraph samplers"""

from collections import defaultdict
from typing import Dict

import torch
from torch.utils.data import functional_datapipe

from .base import seed_type_str_to_ntypes
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
        if minibatch.seeds is not None:
            (
                seeds,
                seeds_timestamp,
                minibatch.compacted_seeds,
            ) = SubgraphSampler._seeds_preprocess(minibatch)
        else:
            raise ValueError(
                f"Invalid minibatch {minibatch}: `seeds` should have a value."
            )
        minibatch._seed_nodes = seeds
        minibatch._seeds_timestamp = seeds_timestamp
        return minibatch

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
        compacting seeds based on their types and timestamps. In heterogeneous
        graph, `seeds` with same node type will be unqiued together.

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
            for seed_type, typed_seeds in seeds.items():
                # When typed_seeds is a one-dimensional tensor, it represents
                # seed nodes, which does not need to do unique and compact.
                if typed_seeds.ndim == 1:
                    nodes_timestamp = (
                        minibatch.timestamp
                        if hasattr(minibatch, "timestamp")
                        else None
                    )
                    return seeds, nodes_timestamp, None
                assert typed_seeds.ndim == 2, (
                    "Only tensor with shape 1*N and N*M is "
                    + f"supported now, but got {typed_seeds.shape}."
                )
                ntypes = seed_type_str_to_ntypes(
                    seed_type, typed_seeds.shape[1]
                )
                if use_timestamp:
                    negative_ratio = (
                        typed_seeds.shape[0]
                        // minibatch.timestamp[seed_type].shape[0]
                        - 1
                    )
                    neg_timestamp = minibatch.timestamp[
                        seed_type
                    ].repeat_interleave(negative_ratio)
                for i, ntype in enumerate(ntypes):
                    nodes[ntype].append(typed_seeds[:, i])
                    if use_timestamp:
                        nodes_timestamp[ntype].append(
                            minibatch.timestamp[seed_type]
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
            for seed_type, typed_seeds in seeds.items():
                ntypes = seed_type_str_to_ntypes(
                    seed_type, typed_seeds.shape[1]
                )
                compacted_seed = []
                for ntype in ntypes:
                    compacted_seed.append(compacted[ntype].pop(0))
                compacted_seeds[seed_type] = (
                    torch.cat(compacted_seed).view(len(ntypes), -1).T
                )
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
                nodes_timestamp = [
                    seeds_timestamp for _ in range(seeds.shape[1])
                ]
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
