"""Neighbor subgraph samplers for GraphBolt."""

from concurrent.futures import ThreadPoolExecutor
from functools import partial

import torch
from torch.utils.data import functional_datapipe
from torchdata.datapipes.iter import Mapper

from ..base import ORIGINAL_EDGE_ID
from ..internal import compact_csc_format, unique_and_compact_csc_formats
from ..minibatch_transformer import MiniBatchTransformer

from ..subgraph_sampler import SubgraphSampler
from .fused_csc_sampling_graph import fused_csc_sampling_graph
from .sampled_subgraph_impl import SampledSubgraphImpl


__all__ = [
    "NeighborSampler",
    "LayerNeighborSampler",
    "SamplePerLayer",
    "SamplePerLayerFromFetchedSubgraph",
    "FetchInsubgraphData",
    "ConcatHeteroSeeds",
    "CombineCachedAndFetchedInSubgraph",
]


@functional_datapipe("fetch_cached_insubgraph_data")
class FetchCachedInsubgraphData(Mapper):
    """Queries the GPUGraphCache and returns the missing seeds and a lambda
    function that can be called with the fetched graph structure.
    """

    def __init__(self, datapipe, gpu_graph_cache):
        super().__init__(datapipe, self._fetch_per_layer)
        self.cache = gpu_graph_cache

    def _fetch_per_layer(self, minibatch):
        minibatch._seeds, minibatch._replace = self.cache.query(
            minibatch._seeds
        )

        return minibatch


@functional_datapipe("combine_cached_and_fetched_insubgraph")
class CombineCachedAndFetchedInSubgraph(Mapper):
    """Combined the fetched graph structure with the graph structure already
    found inside the GPUGraphCache.
    """

    def __init__(self, datapipe, sample_per_layer_obj):
        super().__init__(datapipe, self._combine_per_layer)
        self.prob_name = sample_per_layer_obj.prob_name

    def _combine_per_layer(self, minibatch):
        subgraph = minibatch._sliced_sampling_graph

        edge_tensors = [subgraph.indices]
        if subgraph.type_per_edge is not None:
            edge_tensors.append(subgraph.type_per_edge)
        probs_or_mask = subgraph.edge_attribute(self.prob_name)
        if probs_or_mask is not None:
            edge_tensors.append(probs_or_mask)
        edge_tensors.append(subgraph.edge_attribute(ORIGINAL_EDGE_ID))

        subgraph.csc_indptr, edge_tensors = minibatch._replace(
            subgraph.csc_indptr, edge_tensors
        )
        delattr(minibatch, "_replace")

        subgraph.indices = edge_tensors[0]
        edge_tensors = edge_tensors[1:]
        if subgraph.type_per_edge is not None:
            subgraph.type_per_edge = edge_tensors[0]
            edge_tensors = edge_tensors[1:]
        if probs_or_mask is not None:
            subgraph.add_edge_attribute(self.prob_name, edge_tensors[0])
            edge_tensors = edge_tensors[1:]
        subgraph.add_edge_attribute(ORIGINAL_EDGE_ID, edge_tensors[0])
        edge_tensors = edge_tensors[1:]
        assert len(edge_tensors) == 0

        return minibatch


@functional_datapipe("concat_hetero_seeds")
class ConcatHeteroSeeds(Mapper):
    """Concatenates the seeds into a single tensor in the hetero case."""

    def __init__(self, datapipe, sample_per_layer_obj):
        super().__init__(datapipe, self._concat)
        self.graph = sample_per_layer_obj.sampler.__self__

    def _concat(self, minibatch):
        seeds = minibatch._seed_nodes
        if isinstance(seeds, dict):
            (
                seeds,
                seed_offsets,
            ) = self.graph._convert_to_homogeneous_nodes(seeds)
        else:
            seed_offsets = None
        minibatch._seeds = seeds
        minibatch._seed_offsets = seed_offsets

        return minibatch


@functional_datapipe("fetch_insubgraph_data")
class FetchInsubgraphData(Mapper):
    """Fetches the insubgraph and wraps it in a FusedCSCSamplingGraph object. If
    the provided sample_per_layer_obj has a valid prob_name, then it reads the
    probabilies of all the fetched edges. Furthermore, if type_per_array tensor
    exists in the underlying graph, then the types of all the fetched edges are
    read as well."""

    def __init__(
        self,
        datapipe,
        sample_per_layer_obj,
        gpu_graph_cache,
        stream=None,
        executor=None,
    ):
        self.graph = sample_per_layer_obj.sampler.__self__
        datapipe = datapipe.concat_hetero_seeds(sample_per_layer_obj)
        if gpu_graph_cache is not None:
            datapipe = datapipe.fetch_cached_insubgraph_data(gpu_graph_cache)
        super().__init__(datapipe, self._fetch_per_layer)
        self.prob_name = sample_per_layer_obj.prob_name
        self.stream = stream
        if executor is None:
            self.executor = ThreadPoolExecutor(max_workers=1)
        else:
            self.executor = executor

    def _fetch_per_layer_impl(self, minibatch, stream):
        with torch.cuda.stream(self.stream):
            seeds = minibatch._seeds
            seed_offsets = minibatch._seed_offsets
            delattr(minibatch, "_seeds")
            delattr(minibatch, "_seed_offsets")

            def record_stream(tensor):
                if stream is not None and tensor.is_cuda:
                    tensor.record_stream(stream)
                return tensor

            # Packs tensors for batch slicing.
            tensors_to_be_sliced = [self.graph.indices]

            has_type_per_edge = False
            if self.graph.type_per_edge is not None:
                tensors_to_be_sliced.append(self.graph.type_per_edge)
                has_type_per_edge = True

            has_probs_or_mask = None
            if self.graph.edge_attributes is not None:
                probs_or_mask = self.graph.edge_attributes.get(
                    self.prob_name, None
                )
                if probs_or_mask is not None:
                    tensors_to_be_sliced.append(probs_or_mask)
                    has_probs_or_mask = True

            # Slices the batched tensors.
            (
                indptr,
                sliced_tensors,
            ) = torch.ops.graphbolt.index_select_csc_batched(
                self.graph.csc_indptr, tensors_to_be_sliced, seeds, True, None
            )
            for tensor in [indptr] + sliced_tensors:
                record_stream(tensor)

            # Unpacks the sliced tensors.
            indices = sliced_tensors[0]
            sliced_tensors = sliced_tensors[1:]

            type_per_edge = None
            if has_type_per_edge:
                type_per_edge = sliced_tensors[0]
                sliced_tensors = sliced_tensors[1:]

            probs_or_mask = None
            if has_probs_or_mask:
                probs_or_mask = sliced_tensors[0]
                sliced_tensors = sliced_tensors[1:]

            edge_ids = sliced_tensors[0]
            sliced_tensors = sliced_tensors[1:]
            assert len(sliced_tensors) == 0

            subgraph = fused_csc_sampling_graph(
                indptr,
                indices,
                node_type_offset=self.graph.node_type_offset,
                type_per_edge=type_per_edge,
                node_type_to_id=self.graph.node_type_to_id,
                edge_type_to_id=self.graph.edge_type_to_id,
            )
            if self.prob_name is not None and probs_or_mask is not None:
                subgraph.add_edge_attribute(self.prob_name, probs_or_mask)
            subgraph.add_edge_attribute(ORIGINAL_EDGE_ID, edge_ids)

            subgraph._indptr_node_type_offset_list = seed_offsets
            minibatch._sliced_sampling_graph = subgraph

            if self.stream is not None:
                minibatch.wait = torch.cuda.current_stream().record_event().wait

            return minibatch

    def _fetch_per_layer(self, minibatch):
        current_stream = None
        if self.stream is not None:
            current_stream = torch.cuda.current_stream()
            self.stream.wait_stream(current_stream)
        return self.executor.submit(
            self._fetch_per_layer_impl, minibatch, current_stream
        )


@functional_datapipe("sample_per_layer_from_fetched_subgraph")
class SamplePerLayerFromFetchedSubgraph(MiniBatchTransformer):
    """Sample neighbor edges from a graph for a single layer."""

    def __init__(self, datapipe, sample_per_layer_obj):
        super().__init__(datapipe, self._sample_per_layer_from_fetched_subgraph)
        self.sampler_name = sample_per_layer_obj.sampler.__name__
        self.fanout = sample_per_layer_obj.fanout
        self.replace = sample_per_layer_obj.replace
        self.prob_name = sample_per_layer_obj.prob_name

    def _sample_per_layer_from_fetched_subgraph(self, minibatch):
        subgraph = minibatch._sliced_sampling_graph
        delattr(minibatch, "_sliced_sampling_graph")
        kwargs = {
            key[1:]: getattr(minibatch, key)
            for key in ["_random_seed", "_seed2_contribution"]
            if hasattr(minibatch, key)
        }
        sampled_subgraph = getattr(subgraph, self.sampler_name)(
            None,
            self.fanout,
            self.replace,
            self.prob_name,
            **kwargs,
        )
        minibatch.sampled_subgraphs.insert(0, sampled_subgraph)

        return minibatch


@functional_datapipe("sample_per_layer")
class SamplePerLayer(MiniBatchTransformer):
    """Sample neighbor edges from a graph for a single layer."""

    def __init__(self, datapipe, sampler, fanout, replace, prob_name):
        super().__init__(datapipe, self._sample_per_layer)
        self.sampler = sampler
        self.fanout = fanout
        self.replace = replace
        self.prob_name = prob_name

    def _sample_per_layer(self, minibatch):
        kwargs = {
            key[1:]: getattr(minibatch, key)
            for key in ["_random_seed", "_seed2_contribution"]
            if hasattr(minibatch, key)
        }
        subgraph = self.sampler(
            minibatch._seed_nodes,
            self.fanout,
            self.replace,
            self.prob_name,
            **kwargs,
        )
        minibatch.sampled_subgraphs.insert(0, subgraph)
        return minibatch


@functional_datapipe("compact_per_layer")
class CompactPerLayer(MiniBatchTransformer):
    """Compact the sampled edges for a single layer."""

    def __init__(self, datapipe, deduplicate):
        super().__init__(datapipe, self._compact_per_layer)
        self.deduplicate = deduplicate

    def _compact_per_layer(self, minibatch):
        subgraph = minibatch.sampled_subgraphs[0]
        seeds = minibatch._seed_nodes
        if self.deduplicate:
            (
                original_row_node_ids,
                compacted_csc_format,
            ) = unique_and_compact_csc_formats(subgraph.sampled_csc, seeds)
            subgraph = SampledSubgraphImpl(
                sampled_csc=compacted_csc_format,
                original_column_node_ids=seeds,
                original_row_node_ids=original_row_node_ids,
                original_edge_ids=subgraph.original_edge_ids,
            )
        else:
            (
                original_row_node_ids,
                compacted_csc_format,
            ) = compact_csc_format(subgraph.sampled_csc, seeds)
            subgraph = SampledSubgraphImpl(
                sampled_csc=compacted_csc_format,
                original_column_node_ids=seeds,
                original_row_node_ids=original_row_node_ids,
                original_edge_ids=subgraph.original_edge_ids,
            )
        minibatch._seed_nodes = original_row_node_ids
        minibatch.sampled_subgraphs[0] = subgraph
        return minibatch


@functional_datapipe("fetch_and_sample")
class FetcherAndSampler(MiniBatchTransformer):
    """Overlapped graph sampling operation replacement."""

    def __init__(
        self,
        sampler,
        gpu_graph_cache,
        stream,
        executor,
        buffer_size,
    ):
        datapipe = sampler.datapipe.fetch_insubgraph_data(
            sampler, gpu_graph_cache, stream, executor
        )
        datapipe = datapipe.buffer(buffer_size).wait_future().wait()
        if gpu_graph_cache is not None:
            datapipe = datapipe.combine_cached_and_fetched_insubgraph(sampler)
        datapipe = datapipe.sample_per_layer_from_fetched_subgraph(sampler)
        super().__init__(datapipe)


class NeighborSamplerImpl(SubgraphSampler):
    # pylint: disable=abstract-method
    """Base class for NeighborSamplers."""

    # pylint: disable=useless-super-delegation
    def __init__(
        self,
        datapipe,
        graph,
        fanouts,
        replace,
        prob_name,
        deduplicate,
        sampler,
        layer_dependency=None,
        batch_dependency=None,
    ):
        if sampler.__name__ == "sample_layer_neighbors":
            self._init_seed(batch_dependency)
        super().__init__(
            datapipe,
            graph,
            fanouts,
            replace,
            prob_name,
            deduplicate,
            sampler,
            layer_dependency,
        )

    def _init_seed(self, batch_dependency):
        self.rng = torch.random.manual_seed(
            torch.randint(0, int(1e18), size=tuple())
        )
        self.cnt = [-1, int(batch_dependency)]
        self.random_seed = torch.empty(
            2 if self.cnt[1] > 1 else 1, dtype=torch.int64
        )
        self.random_seed.random_(generator=self.rng)

    def _set_seed(self, minibatch):
        self.cnt[0] += 1
        if self.cnt[1] > 0 and self.cnt[0] % self.cnt[1] == 0:
            self.random_seed[0] = self.random_seed[-1]
            self.random_seed[-1:].random_(generator=self.rng)
        minibatch._random_seed = self.random_seed.clone()
        minibatch._seed2_contribution = (
            0.0
            if self.cnt[1] <= 1
            else (self.cnt[0] % self.cnt[1]) / self.cnt[1]
        )
        minibatch._iter = self.cnt[0]
        return minibatch

    @staticmethod
    def _increment_seed(minibatch):
        minibatch._random_seed = 1 + minibatch._random_seed
        return minibatch

    @staticmethod
    def _delattr_dependency(minibatch):
        delattr(minibatch, "_random_seed")
        delattr(minibatch, "_seed2_contribution")
        return minibatch

    @staticmethod
    def _prepare(node_type_to_id, minibatch):
        seeds = minibatch._seed_nodes
        # Enrich seeds with all node types.
        if isinstance(seeds, dict):
            ntypes = list(node_type_to_id.keys())
            # Loop over different seeds to extract the device they are on.
            device = None
            dtype = None
            for _, seed in seeds.items():
                device = seed.device
                dtype = seed.dtype
                break
            default_tensor = torch.tensor([], dtype=dtype, device=device)
            seeds = {
                ntype: seeds.get(ntype, default_tensor) for ntype in ntypes
            }
        minibatch._seed_nodes = seeds
        minibatch.sampled_subgraphs = []
        return minibatch

    @staticmethod
    def _set_input_nodes(minibatch):
        minibatch.input_nodes = minibatch._seed_nodes
        return minibatch

    # pylint: disable=arguments-differ
    def sampling_stages(
        self,
        datapipe,
        graph,
        fanouts,
        replace,
        prob_name,
        deduplicate,
        sampler,
        layer_dependency,
    ):
        datapipe = datapipe.transform(
            partial(self._prepare, graph.node_type_to_id)
        )
        is_labor = sampler.__name__ == "sample_layer_neighbors"
        if is_labor:
            datapipe = datapipe.transform(self._set_seed)
        for fanout in reversed(fanouts):
            # Convert fanout to tensor.
            if not isinstance(fanout, torch.Tensor):
                fanout = torch.LongTensor([int(fanout)])
            datapipe = datapipe.sample_per_layer(
                sampler, fanout, replace, prob_name
            )
            datapipe = datapipe.compact_per_layer(deduplicate)
            if is_labor and not layer_dependency:
                datapipe = datapipe.transform(self._increment_seed)
        if is_labor:
            datapipe = datapipe.transform(self._delattr_dependency)
        return datapipe.transform(self._set_input_nodes)


@functional_datapipe("sample_neighbor")
class NeighborSampler(NeighborSamplerImpl):
    # pylint: disable=abstract-method
    """Sample neighbor edges from a graph and return a subgraph.

    Functional name: :obj:`sample_neighbor`.

    Neighbor sampler is responsible for sampling a subgraph from given data. It
    returns an induced subgraph along with compacted information. In the
    context of a node classification task, the neighbor sampler directly
    utilizes the nodes provided as seed nodes. However, in scenarios involving
    link prediction, the process needs another pre-peocess operation. That is,
    gathering unique nodes from the given node pairs, encompassing both
    positive and negative node pairs, and employs these nodes as the seed nodes
    for subsequent steps. When the graph is hetero, sampled subgraphs in
    minibatch will contain every edge type even though it is empty after
    sampling.

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
    deduplicate: bool
        Boolean indicating whether seeds between hops will be deduplicated.
        If True, the same elements in seeds will be deleted to only one.
        Otherwise, the same elements will be remained.

    Examples
    -------
    >>> import torch
    >>> import dgl.graphbolt as gb
    >>> indptr = torch.LongTensor([0, 2, 4, 5, 6, 7 ,8])
    >>> indices = torch.LongTensor([1, 2, 0, 3, 5, 4, 3, 5])
    >>> graph = gb.fused_csc_sampling_graph(indptr, indices)
    >>> seeds = torch.LongTensor([[0, 1], [1, 2]])
    >>> item_set = gb.ItemSet(seeds, names="seeds")
    >>> datapipe = gb.ItemSampler(item_set, batch_size=1)
    >>> datapipe = datapipe.sample_uniform_negative(graph, 2)
    >>> datapipe = datapipe.sample_neighbor(graph, [5, 10, 15])
    >>> next(iter(datapipe)).sampled_subgraphs
    [SampledSubgraphImpl(sampled_csc=CSCFormatBase(
            indptr=tensor([0, 2, 4, 5, 6, 7, 8]),
            indices=tensor([1, 4, 0, 5, 5, 3, 3, 2]),
        ),
        original_row_node_ids=tensor([0, 1, 4, 5, 2, 3]),
        original_edge_ids=None,
        original_column_node_ids=tensor([0, 1, 4, 5, 2, 3]),
    ),
    SampledSubgraphImpl(sampled_csc=CSCFormatBase(
            indptr=tensor([0, 2, 4, 5, 6, 7, 8]),
            indices=tensor([1, 4, 0, 5, 5, 3, 3, 2]),
        ),
        original_row_node_ids=tensor([0, 1, 4, 5, 2, 3]),
        original_edge_ids=None,
        original_column_node_ids=tensor([0, 1, 4, 5, 2, 3]),
    ),
    SampledSubgraphImpl(sampled_csc=CSCFormatBase(
            indptr=tensor([0, 2, 4, 5, 6]),
            indices=tensor([1, 4, 0, 5, 5, 3]),
        ),
        original_row_node_ids=tensor([0, 1, 4, 5, 2, 3]),
        original_edge_ids=None,
        original_column_node_ids=tensor([0, 1, 4, 5]),
    )]
    """

    # pylint: disable=useless-super-delegation
    def __init__(
        self,
        datapipe,
        graph,
        fanouts,
        replace=False,
        prob_name=None,
        deduplicate=True,
    ):
        super().__init__(
            datapipe,
            graph,
            fanouts,
            replace,
            prob_name,
            deduplicate,
            graph.sample_neighbors,
        )


@functional_datapipe("sample_layer_neighbor")
class LayerNeighborSampler(NeighborSamplerImpl):
    # pylint: disable=abstract-method
    """Sample layer neighbor edges from a graph and return a subgraph.

    Functional name: :obj:`sample_layer_neighbor`.

    Sampler that builds computational dependency of node representations via
    labor sampling for multilayer GNN from the NeurIPS 2023 paper
    `Layer-Neighbor Sampling -- Defusing Neighborhood Explosion in GNNs
    <https://proceedings.neurips.cc/paper_files/paper/2023/file/51f9036d5e7ae822da8f6d4adda1fb39-Paper-Conference.pdf>`__

    Layer-Neighbor sampler is responsible for sampling a subgraph from given
    data. It returns an induced subgraph along with compacted information. In
    the context of a node classification task, the neighbor sampler directly
    utilizes the nodes provided as seed nodes. However, in scenarios involving
    link prediction, the process needs another pre-process operation. That is,
    gathering unique nodes from the given node pairs, encompassing both
    positive and negative node pairs, and employs these nodes as the seed nodes
    for subsequent steps. When the graph is hetero, sampled subgraphs in
    minibatch will contain every edge type even though it is empty after
    sampling.

    Implements the approach described in Appendix A.3 of the paper. Similar to
    dgl.dataloading.LaborSampler but this uses sequential poisson sampling
    instead of poisson sampling to keep the count of sampled edges per vertex
    deterministic like NeighborSampler. Thus, it is a drop-in replacement for
    NeighborSampler. However, unlike NeighborSampler, it samples fewer vertices
    and edges for multilayer GNN scenario without harming convergence speed with
    respect to training iterations.

    Parameters
    ----------
    datapipe : DataPipe
        The datapipe.
    graph : FusedCSCSamplingGraph
        The graph on which to perform subgraph sampling.
    fanouts: list[torch.Tensor]
        The number of edges to be sampled for each node with or without
        considering edge types. The length of this parameter implicitly
        signifies the layer of sampling being conducted.
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
    deduplicate: bool
        Boolean indicating whether seeds between hops will be deduplicated.
        If True, the same elements in seeds will be deleted to only one.
        Otherwise, the same elements will be remained.
    layer_dependency: bool
        Boolean indicating whether different layers should use the same random
        variates. Results in a reduction in the number of nodes sampled and
        turns LayerNeighborSampler into a subgraph sampling method. Later layers
        will be guaranteed to sample overlapping neighbors as the previous
        layers.
    batch_dependency: int
        Specifies whether consecutive minibatches should use similar random
        variates. Results in a higher temporal access locality of sampled
        nodes and edges. Setting it to :math:`\\kappa` slows down the change in
        the random variates proportional to :math:`\\frac{1}{\\kappa}`. Implements
        the dependent minibatching approach in `arXiv:2310.12403
        <https://arxiv.org/abs/2310.12403>`__.

    Examples
    -------
    >>> import dgl.graphbolt as gb
    >>> import torch
    >>> indptr = torch.LongTensor([0, 2, 4, 5, 6, 7 ,8])
    >>> indices = torch.LongTensor([1, 2, 0, 3, 5, 4, 3, 5])
    >>> graph = gb.fused_csc_sampling_graph(indptr, indices)
    >>> seeds = torch.LongTensor([[0, 1], [1, 2]])
    >>> item_set = gb.ItemSet(seeds, names="seeds")
    >>> item_sampler = gb.ItemSampler(item_set, batch_size=1,)
    >>> neg_sampler = gb.UniformNegativeSampler(item_sampler, graph, 2)
    >>> fanouts = [torch.LongTensor([5]),
    ...     torch.LongTensor([10]),torch.LongTensor([15])]
    >>> subgraph_sampler = gb.LayerNeighborSampler(neg_sampler, graph, fanouts)
    >>> next(iter(subgraph_sampler)).sampled_subgraphs
    [SampledSubgraphImpl(sampled_csc=CSCFormatBase(
            indptr=tensor([0, 2, 4, 5, 6, 7, 8]),
            indices=tensor([1, 3, 0, 4, 2, 2, 5, 4]),
        ),
        original_row_node_ids=tensor([0, 1, 5, 2, 3, 4]),
        original_edge_ids=None,
        original_column_node_ids=tensor([0, 1, 5, 2, 3, 4]),
    ),
    SampledSubgraphImpl(sampled_csc=CSCFormatBase(
            indptr=tensor([0, 2, 4, 5, 6, 7]),
            indices=tensor([1, 3, 0, 4, 2, 2, 5]),
        ),
        original_row_node_ids=tensor([0, 1, 5, 2, 3, 4]),
        original_edge_ids=None,
        original_column_node_ids=tensor([0, 1, 5, 2, 3]),
    ),
    SampledSubgraphImpl(sampled_csc=CSCFormatBase(
            indptr=tensor([0, 2, 4, 5, 6]),
            indices=tensor([1, 3, 0, 4, 2, 2]),
        ),
        original_row_node_ids=tensor([0, 1, 5, 2, 3]),
        original_edge_ids=None,
        original_column_node_ids=tensor([0, 1, 5, 2]),
    )]
    >>> next(iter(subgraph_sampler)).compacted_seeds
    tensor([[0, 1], [0, 2], [0, 3]])
    >>> next(iter(subgraph_sampler)).labels
    tensor([1., 0., 0.])
    >>> next(iter(subgraph_sampler)).indexes
    tensor([0, 0, 0])
    """

    def __init__(
        self,
        datapipe,
        graph,
        fanouts,
        replace=False,
        prob_name=None,
        deduplicate=True,
        layer_dependency=False,
        batch_dependency=1,
    ):
        super().__init__(
            datapipe,
            graph,
            fanouts,
            replace,
            prob_name,
            deduplicate,
            graph.sample_layer_neighbors,
            layer_dependency,
            batch_dependency,
        )
