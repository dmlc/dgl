"""Subgraph samplers"""

from collections import defaultdict
from functools import partial
from typing import Dict

import torch
import torch.distributed as thd
from torch.utils.data import functional_datapipe

from .base import seed_type_str_to_ntypes
from .internal import compact_temporal_nodes, unique_and_compact
from .minibatch import MiniBatch
from .minibatch_transformer import MiniBatchTransformer

__all__ = [
    "SubgraphSampler",
    "all_to_all",
    "convert_to_hetero",
    "revert_to_homo",
]


class _NoOpWaiter:
    def __init__(self, result):
        self.result = result

    def wait(self):
        """Returns the stored value when invoked."""
        result = self.result
        # Ensure there is no memory leak.
        self.result = None
        return result


def _shift(inputs: list, group=None):
    cutoff = len(inputs) - thd.get_rank(group)
    return inputs[cutoff:] + inputs[:cutoff]


def all_to_all(outputs, inputs, group=None, async_op=False):
    """Wrapper for thd.all_to_all that permuted outputs and inputs before
    calling it. The arguments have the permutation
    `rank, ..., world_size - 1, 0, ..., rank - 1` and we make it
    `0, world_size - 1` before calling `thd.all_to_all`."""
    shift_fn = partial(_shift, group=group)
    outputs = shift_fn(list(outputs))
    inputs = shift_fn(list(inputs))
    if outputs[0].is_cuda:
        return thd.all_to_all(outputs, inputs, group, async_op)
    # gloo backend will be used.
    outputs_single = torch.cat(outputs)
    output_split_sizes = [o.size(0) for o in outputs]
    handle = thd.all_to_all_single(
        outputs_single,
        torch.cat(inputs),
        output_split_sizes,
        [i.size(0) for i in inputs],
        group,
        async_op,
    )
    temp_outputs = outputs_single.split(output_split_sizes)

    class _Waiter:
        def __init__(self, handle, outputs, temp_outputs):
            self.handle = handle
            self.outputs = outputs
            self.temp_outputs = temp_outputs

        def wait(self):
            """Returns the stored value when invoked."""
            handle = self.handle
            outputs = self.outputs
            temp_outputs = self.temp_outputs
            # Ensure that there is no leak
            self.handle = self.outputs = self.temp_outputs = None

            if handle is not None:
                handle.wait()
            for output, temp_output in zip(outputs, temp_outputs):
                output.copy_(temp_output)

    post_processor = _Waiter(handle, outputs, temp_outputs)
    return post_processor if async_op else post_processor.wait()


def revert_to_homo(d: dict):
    """Utility function to convert a dictionary that stores homogenous data."""
    is_homogenous = len(d) == 1 and "_N" in d
    return list(d.values())[0] if is_homogenous else d


def convert_to_hetero(item):
    """Utility function to convert homogenous data to heterogenous with a single
    node type."""
    is_heterogenous = isinstance(item, dict)
    return item if is_heterogenous else {"_N": item}


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
        Arguments to be passed into sampling_stages. Preprocessing stage makes
        use of the `asynchronous` and `cooperative` parameters before they are
        passed to the sampling stages.
    """

    def __init__(
        self,
        datapipe,
        *args,
        **kwargs,
    ):
        async_op = kwargs.get("asynchronous", False)
        cooperative = kwargs.get("cooperative", False)
        preprocess_fn = partial(
            self._preprocess, cooperative=cooperative, async_op=async_op
        )
        datapipe = datapipe.transform(preprocess_fn)
        if async_op:
            fn = partial(self._wait_preprocess_future, cooperative=cooperative)
            datapipe = datapipe.buffer().transform(fn)
        if cooperative:
            datapipe = datapipe.transform(self._seeds_cooperative_exchange_1)
            datapipe = datapipe.buffer()
            datapipe = datapipe.transform(
                self._seeds_cooperative_exchange_1_wait_future
            ).buffer()
            datapipe = datapipe.transform(self._seeds_cooperative_exchange_2)
            datapipe = datapipe.buffer()
            datapipe = datapipe.transform(self._seeds_cooperative_exchange_3)
            datapipe = datapipe.buffer()
            datapipe = datapipe.transform(self._seeds_cooperative_exchange_4)
        datapipe = self.sampling_stages(datapipe, *args, **kwargs)
        datapipe = datapipe.transform(self._postprocess)
        super().__init__(datapipe)

    @staticmethod
    def _postprocess(minibatch):
        delattr(minibatch, "_seed_nodes")
        delattr(minibatch, "_seeds_timestamp")
        return minibatch

    @staticmethod
    def _preprocess(minibatch, cooperative: bool, async_op: bool):
        if minibatch.seeds is None:
            raise ValueError(
                f"Invalid minibatch {minibatch}: `seeds` should have a value."
            )
        rank = thd.get_rank() if cooperative else 0
        world_size = thd.get_world_size() if cooperative else 1
        results = SubgraphSampler._seeds_preprocess(
            minibatch, rank, world_size, async_op
        )
        if async_op:
            minibatch._preprocess_future = results
        else:
            (
                minibatch._seed_nodes,
                minibatch._seeds_timestamp,
                minibatch.compacted_seeds,
                offsets,
            ) = results
            if cooperative:
                minibatch._seeds_offsets = offsets
        return minibatch

    @staticmethod
    def _wait_preprocess_future(minibatch, cooperative: bool):
        (
            minibatch._seed_nodes,
            minibatch._seeds_timestamp,
            minibatch.compacted_seeds,
            offsets,
        ) = minibatch._preprocess_future.wait()
        delattr(minibatch, "_preprocess_future")
        if cooperative:
            minibatch._seeds_offsets = offsets
        return minibatch

    @staticmethod
    def _seeds_cooperative_exchange_1(minibatch):
        rank = thd.get_rank()
        world_size = thd.get_world_size()
        seeds = minibatch._seed_nodes
        is_homogeneous = not isinstance(seeds, dict)
        if is_homogeneous:
            seeds = {"_N": seeds}
        if minibatch._seeds_offsets is None:
            assert minibatch.compacted_seeds is None
            minibatch._rank_sort_future = torch.ops.graphbolt.rank_sort_async(
                list(seeds.values()), rank, world_size
            )
        return minibatch

    @staticmethod
    def _seeds_cooperative_exchange_1_wait_future(minibatch):
        world_size = thd.get_world_size()
        seeds = minibatch._seed_nodes
        is_homogeneous = not isinstance(seeds, dict)
        if is_homogeneous:
            seeds = {"_N": seeds}
        num_ntypes = len(seeds.keys())
        if minibatch._seeds_offsets is None:
            result = minibatch._rank_sort_future.wait()
            delattr(minibatch, "_rank_sort_future")
            sorted_seeds, sorted_compacted, sorted_offsets = {}, {}, {}
            for i, (
                seed_type,
                (typed_sorted_seeds, typed_index, typed_offsets),
            ) in enumerate(zip(seeds.keys(), result)):
                sorted_seeds[seed_type] = typed_sorted_seeds
                sorted_compacted[seed_type] = typed_index
                sorted_offsets[seed_type] = typed_offsets

            minibatch._seed_nodes = sorted_seeds
            minibatch.compacted_seeds = revert_to_homo(sorted_compacted)
            minibatch._seeds_offsets = sorted_offsets
        else:
            minibatch._seeds_offsets = {"_N": minibatch._seeds_offsets}
        counts_sent = torch.empty(world_size * num_ntypes, dtype=torch.int64)
        for i, offsets in enumerate(minibatch._seeds_offsets.values()):
            counts_sent[
                torch.arange(i, world_size * num_ntypes, num_ntypes)
            ] = offsets.diff()
        delattr(minibatch, "_seeds_offsets")
        counts_received = torch.empty_like(counts_sent)
        minibatch._counts_future = all_to_all(
            counts_received.split(num_ntypes),
            counts_sent.split(num_ntypes),
            async_op=True,
        )
        minibatch._counts_sent = counts_sent
        minibatch._counts_received = counts_received
        return minibatch

    @staticmethod
    def _seeds_cooperative_exchange_2(minibatch):
        world_size = thd.get_world_size()
        seeds = minibatch._seed_nodes
        minibatch._counts_future.wait()
        delattr(minibatch, "_counts_future")
        num_ntypes = len(seeds.keys())
        seeds_received = {}
        counts_sent = {}
        counts_received = {}
        for i, (ntype, typed_seeds) in enumerate(seeds.items()):
            idx = torch.arange(i, world_size * num_ntypes, num_ntypes)
            typed_counts_sent = minibatch._counts_sent[idx].tolist()
            typed_counts_received = minibatch._counts_received[idx].tolist()
            typed_seeds_received = typed_seeds.new_empty(
                sum(typed_counts_received)
            )
            all_to_all(
                typed_seeds_received.split(typed_counts_received),
                typed_seeds.split(typed_counts_sent),
            )
            seeds_received[ntype] = typed_seeds_received
            counts_sent[ntype] = typed_counts_sent
            counts_received[ntype] = typed_counts_received
        minibatch._seed_nodes = seeds_received
        minibatch._counts_sent = revert_to_homo(counts_sent)
        minibatch._counts_received = revert_to_homo(counts_received)
        return minibatch

    @staticmethod
    def _seeds_cooperative_exchange_3(minibatch):
        nodes = {
            ntype: [typed_seeds]
            for ntype, typed_seeds in minibatch._seed_nodes.items()
        }
        minibatch._unique_future = unique_and_compact(
            nodes, 0, 1, async_op=True
        )
        return minibatch

    @staticmethod
    def _seeds_cooperative_exchange_4(minibatch):
        unique_seeds, inverse_seeds, _ = minibatch._unique_future.wait()
        delattr(minibatch, "_unique_future")
        inverse_seeds = {
            ntype: typed_inv[0] for ntype, typed_inv in inverse_seeds.items()
        }
        minibatch._seed_nodes = revert_to_homo(unique_seeds)
        sizes = {
            ntype: typed_seeds.size(0)
            for ntype, typed_seeds in unique_seeds.items()
        }
        minibatch._seed_sizes = revert_to_homo(sizes)
        minibatch._seed_inverse_ids = revert_to_homo(inverse_seeds)
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
    def _seeds_preprocess(
        minibatch: MiniBatch,
        rank: int = 0,
        world_size: int = 1,
        async_op: bool = False,
    ):
        """Preprocess `seeds` in a minibatch to construct `unique_seeds`,
        `node_timestamp` and `compacted_seeds` for further sampling. It
        optionally incorporates timestamps for temporal graphs, organizing and
        compacting seeds based on their types and timestamps. In heterogeneous
        graph, `seeds` with same node type will be unqiued together.

        Parameters
        ----------
        minibatch: MiniBatch
            The minibatch.
        rank : int
            The rank of the current process among cooperating processes.
        world_size : int
            The number of cooperating
            (`arXiv:2210.13339<https://arxiv.org/abs/2310.12403>`__) processes.
        async_op: bool
            Boolean indicating whether the call is asynchronous. If so, the
            result can be obtained by calling wait on the returned future.

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
        offsets: None or torch.Tensor or Dict[src, torch.Tensor]
            The unique nodes offsets tensor partitions the unique_nodes tensor.
            Has size `world_size + 1` and
            `unique_nodes[offsets[i]: offsets[i + 1]]` belongs to the rank
            `(rank + i) % world_size`.
        """
        use_timestamp = hasattr(minibatch, "timestamp")
        assert (
            not use_timestamp or world_size == 1
        ), "Temporal code path does not currently support Cooperative Minibatching"
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
                    result = _NoOpWaiter((seeds, nodes_timestamp, None, None))
                    break
                result = None
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

            class _Waiter:
                def __init__(self, nodes, nodes_timestamp, seeds):
                    # Unique and compact the collected nodes.
                    if use_timestamp:
                        self.future = compact_temporal_nodes(
                            nodes, nodes_timestamp
                        )
                    else:
                        self.future = unique_and_compact(
                            nodes, rank, world_size, async_op
                        )
                    self.seeds = seeds

                def wait(self):
                    """Returns the stored value when invoked."""
                    if use_timestamp:
                        unique_seeds, nodes_timestamp, compacted = self.future
                        offsets = None
                    else:
                        unique_seeds, compacted, offsets = (
                            self.future.wait() if async_op else self.future
                        )
                        nodes_timestamp = None
                    seeds = self.seeds
                    # Ensure there is no memory leak.
                    self.future = self.seeds = None

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

                    return (
                        unique_seeds,
                        nodes_timestamp,
                        compacted_seeds,
                        offsets,
                    )

            # When typed_seeds is not a one-dimensional tensor
            if result is None:
                result = _Waiter(nodes, nodes_timestamp, seeds)
        else:
            # When seeds is a one-dimensional tensor, it represents seed nodes,
            # which does not need to do unique and compact.
            if seeds.ndim == 1:
                nodes_timestamp = (
                    minibatch.timestamp
                    if hasattr(minibatch, "timestamp")
                    else None
                )
                result = _NoOpWaiter((seeds, nodes_timestamp, None, None))
            else:
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

                class _Waiter:
                    def __init__(self, nodes, nodes_timestamp, seeds):
                        # Unique and compact the collected nodes.
                        if use_timestamp:
                            self.future = compact_temporal_nodes(
                                nodes, nodes_timestamp
                            )
                        else:
                            self.future = unique_and_compact(
                                nodes, async_op=async_op
                            )
                        self.seeds = seeds

                    def wait(self):
                        """Returns the stored value when invoked."""
                        if use_timestamp:
                            (
                                unique_seeds,
                                nodes_timestamp,
                                compacted,
                            ) = self.future
                            offsets = None
                        else:
                            unique_seeds, compacted, offsets = (
                                self.future.wait() if async_op else self.future
                            )
                            nodes_timestamp = None
                        seeds = self.seeds
                        # Ensure there is no memory leak.
                        self.future = self.seeds = None

                        # Map back in same order as collect.
                        compacted_seeds = compacted[0].view(seeds.shape)

                        return (
                            unique_seeds,
                            nodes_timestamp,
                            compacted_seeds,
                            offsets,
                        )

                result = _Waiter(nodes, nodes_timestamp, seeds)

        return result if async_op else result.wait()

    def sample_subgraphs(
        self, seeds, seeds_timestamp, seeds_pre_time_window=None
    ):
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

        seeds_pre_time_window : Union[torch.Tensor, Dict[str, torch.Tensor]]
            The time window of the nodes represents a period of time before
            `seeds_timestamp`. If provided, only neighbors and related edges
            whose timestamps fall within `[seeds_timestamp -
            seeds_pre_time_window, seeds_timestamp]` will be filtered.
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
