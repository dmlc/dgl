"""Graph Bolt DataLoaders"""

from collections import OrderedDict

import torch
import torch.utils.data as torch_data
import torchdata.dataloader2.graph as dp_utils

from .base import CopyTo, get_host_to_device_uva_stream
from .feature_fetcher import FeatureFetcher, FeatureFetcherStartMarker
from .impl.gpu_graph_cache import GPUGraphCache
from .impl.neighbor_sampler import SamplePerLayer

from .internal import datapipe_graph_to_adjlist
from .item_sampler import ItemSampler


__all__ = [
    "DataLoader",
    "construct_gpu_graph_cache",
]


def construct_gpu_graph_cache(
    sample_per_layer_obj, num_gpu_cached_edges, gpu_cache_threshold
):
    "Construct a GPUGraphCache given a sample_per_layer_obj and cache parameters."
    graph = sample_per_layer_obj.sampler.__self__
    num_gpu_cached_edges = min(num_gpu_cached_edges, graph.total_num_edges)
    dtypes = OrderedDict()
    dtypes["indices"] = graph.indices.dtype
    if graph.type_per_edge is not None:
        dtypes["type_per_edge"] = graph.type_per_edge.dtype
    if graph.edge_attributes is not None:
        probs_or_mask = graph.edge_attributes.get(
            sample_per_layer_obj.prob_name, None
        )
        if probs_or_mask is not None:
            dtypes["probs_or_mask"] = probs_or_mask.dtype
    return GPUGraphCache(
        num_gpu_cached_edges,
        gpu_cache_threshold,
        graph.csc_indptr.dtype,
        list(dtypes.values()),
    )


def _find_and_wrap_parent(datapipe_graph, target_datapipe, wrapper, **kwargs):
    """Find parent of target_datapipe and wrap it with ."""
    datapipes = dp_utils.find_dps(
        datapipe_graph,
        target_datapipe,
    )
    datapipe_adjlist = datapipe_graph_to_adjlist(datapipe_graph)
    for datapipe in datapipes:
        datapipe_id = id(datapipe)
        for parent_datapipe_id in datapipe_adjlist[datapipe_id][1]:
            parent_datapipe, _ = datapipe_adjlist[parent_datapipe_id]
            datapipe_graph = dp_utils.replace_dp(
                datapipe_graph,
                parent_datapipe,
                wrapper(parent_datapipe, **kwargs),
            )
    return datapipe_graph


def _set_worker_id(worked_id):
    torch.ops.graphbolt.set_worker_id(worked_id)


class MultiprocessingWrapper(torch_data.IterDataPipe):
    """Wraps a datapipe with multiprocessing.

    Parameters
    ----------
    datapipe : DataPipe
        The data pipeline.
    num_workers : int, optional
        The number of worker processes. Default is 0, meaning that there
        will be no multiprocessing.
    persistent_workers : bool, optional
        If True, the data loader will not shut down the worker processes after a
        dataset has been consumed once. This allows to maintain the workers
        instances alive.
    """

    def __init__(self, datapipe, num_workers=0, persistent_workers=True):
        self.datapipe = datapipe
        self.dataloader = torch_data.DataLoader(
            datapipe,
            batch_size=None,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
            worker_init_fn=_set_worker_id if num_workers > 0 else None,
        )

    def __iter__(self):
        yield from self.dataloader


class DataLoader(torch_data.DataLoader):
    """Multiprocessing DataLoader.

    Iterates over the data pipeline with everything before feature fetching
    (i.e. :class:`dgl.graphbolt.FeatureFetcher`) in subprocesses, and
    everything after feature fetching in the main process. The datapipe
    is modified in-place as a result.

    When the copy_to operation is placed earlier in the data pipeline, the
    num_workers argument is required to be 0 as utilizing CUDA in multiple
    worker processes is not supported.

    Parameters
    ----------
    datapipe : DataPipe
        The data pipeline.
    num_workers : int, optional
        Number of worker processes. Default is 0.
    persistent_workers : bool, optional
        If True, the data loader will not shut down the worker processes after a
        dataset has been consumed once. This allows to maintain the workers
        instances alive.
    overlap_graph_fetch : bool, optional
        If True, the data loader will overlap the UVA graph fetching operations
        with the rest of operations by using an alternative CUDA stream. Default
        is False.
    num_gpu_cached_edges : int, optional
        If positive and overlap_graph_fetch is True, then the GPU will cache
        frequently accessed vertex neighborhoods to reduce the PCI-e bandwidth
        demand due to pinned graph accesses.
    gpu_cache_threshold : int, optional
        Determines how many times a vertex needs to be accessed before its
        neighborhood ends up being cached on the GPU.
    max_uva_threads : int, optional
        Limits the number of CUDA threads used for UVA copies so that the rest
        of the computations can run simultaneously with it. Setting it to a too
        high value will limit the amount of overlap while setting it too low may
        cause the PCI-e bandwidth to not get fully utilized. Manually tuned
        default is 6144, meaning around 3-4 Streaming Multiprocessors.
    """

    def __init__(
        self,
        datapipe,
        num_workers=0,
        persistent_workers=True,
        overlap_graph_fetch=False,
        num_gpu_cached_edges=0,
        gpu_cache_threshold=1,
        max_uva_threads=6144,
    ):
        # Multiprocessing requires two modifications to the datapipe:
        #
        # 1. Insert a stage after ItemSampler to distribute the
        #    minibatches evenly across processes.
        # 2. Cut the datapipe at FeatureFetcher, and wrap the inner datapipe
        #    of the FeatureFetcher with a multiprocessing PyTorch DataLoader.

        datapipe = datapipe.mark_end()
        datapipe_graph = dp_utils.traverse_dps(datapipe)

        # (1) Insert minibatch distribution.
        # TODO(BarclayII): Currently I'm using sharding_filter() as a
        # concept demonstration. Later on minibatch distribution should be
        # merged into ItemSampler to maximize efficiency.
        item_samplers = dp_utils.find_dps(
            datapipe_graph,
            ItemSampler,
        )
        for item_sampler in item_samplers:
            datapipe_graph = dp_utils.replace_dp(
                datapipe_graph,
                item_sampler,
                item_sampler.sharding_filter(),
            )

        # (2) Cut datapipe at FeatureFetcher and wrap.
        datapipe_graph = _find_and_wrap_parent(
            datapipe_graph,
            FeatureFetcherStartMarker,
            MultiprocessingWrapper,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

        # (3) Limit the number of UVA threads used if the feature_fetcher has
        # overlapping optimization enabled.
        if num_workers == 0 and torch.cuda.is_available():
            feature_fetchers = dp_utils.find_dps(
                datapipe_graph,
                FeatureFetcher,
            )
            for feature_fetcher in feature_fetchers:
                if feature_fetcher.max_num_stages > 0:  # Overlap enabled.
                    torch.ops.graphbolt.set_max_uva_threads(max_uva_threads)

        if (
            overlap_graph_fetch
            and num_workers == 0
            and torch.cuda.is_available()
        ):
            torch.ops.graphbolt.set_max_uva_threads(max_uva_threads)
            samplers = dp_utils.find_dps(
                datapipe_graph,
                SamplePerLayer,
            )
            gpu_graph_cache = None
            for sampler in samplers:
                if num_gpu_cached_edges > 0 and gpu_graph_cache is None:
                    gpu_graph_cache = construct_gpu_graph_cache(
                        sampler, num_gpu_cached_edges, gpu_cache_threshold
                    )
                datapipe_graph = dp_utils.replace_dp(
                    datapipe_graph,
                    sampler,
                    sampler.fetch_and_sample(
                        gpu_graph_cache,
                        get_host_to_device_uva_stream(),
                        1,
                    ),
                )

        # (4) Cut datapipe at CopyTo and wrap with pinning and prefetching
        # before it. This enables enables non_blocking copies to the device.
        # Prefetching enables the data pipeline up to the CopyTo to run in a
        # separate thread.
        if torch.cuda.is_available():
            copiers = dp_utils.find_dps(datapipe_graph, CopyTo)
            for copier in copiers:
                if copier.device.type == "cuda":
                    datapipe_graph = dp_utils.replace_dp(
                        datapipe_graph,
                        copier,
                        # Add prefetch so that CPU and GPU can run concurrently.
                        copier.datapipe.prefetch(2).copy_to(
                            copier.device, non_blocking=True
                        ),
                    )

        # The stages after feature fetching is still done in the main process.
        # So we set num_workers to 0 here.
        super().__init__(datapipe, batch_size=None, num_workers=0)
