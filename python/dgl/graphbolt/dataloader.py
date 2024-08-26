"""Graph Bolt DataLoaders"""

import torch
import torch.utils.data as torch_data

from .base import CopyTo
from .datapipes import (
    datapipe_graph_to_adjlist,
    find_dps,
    replace_dp,
    traverse_dps,
)
from .feature_fetcher import FeatureFetcher, FeatureFetcherStartMarker
from .impl.neighbor_sampler import SamplePerLayer
from .internal_utils import gb_warning
from .item_sampler import ItemSampler
from .minibatch_transformer import MiniBatchTransformer


__all__ = [
    "DataLoader",
]


def _find_and_wrap_parent(datapipe_graph, target_datapipe, wrapper, **kwargs):
    """Find parent of target_datapipe and wrap it with ."""
    datapipes = find_dps(
        datapipe_graph,
        target_datapipe,
    )
    datapipe_adjlist = datapipe_graph_to_adjlist(datapipe_graph)
    for datapipe in datapipes:
        datapipe_id = id(datapipe)
        for parent_datapipe_id in datapipe_adjlist[datapipe_id][1]:
            parent_datapipe, _ = datapipe_adjlist[parent_datapipe_id]
            datapipe_graph = replace_dp(
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


class DataLoader(MiniBatchTransformer):
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
    max_uva_threads : int, optional
        Limits the number of CUDA threads used for UVA copies so that the rest
        of the computations can run simultaneously with it. Setting it to a too
        high value will limit the amount of overlap while setting it too low may
        cause the PCI-e bandwidth to not get fully utilized. Manually tuned
        default is 10240, meaning around 5-7 Streaming Multiprocessors.
    """

    def __init__(
        self,
        datapipe,
        num_workers=0,
        persistent_workers=True,
        max_uva_threads=10240,
    ):
        # Multiprocessing requires two modifications to the datapipe:
        #
        # 1. Insert a stage after ItemSampler to distribute the
        #    minibatches evenly across processes.
        # 2. Cut the datapipe at FeatureFetcher, and wrap the inner datapipe
        #    of the FeatureFetcher with a multiprocessing PyTorch DataLoader.

        datapipe = datapipe.mark_end()
        datapipe_graph = traverse_dps(datapipe)

        if num_workers > 0:
            # (1) Insert minibatch distribution.
            # TODO(BarclayII): Currently I'm using sharding_filter() as a
            # concept demonstration. Later on minibatch distribution should be
            # merged into ItemSampler to maximize efficiency.
            item_samplers = find_dps(
                datapipe_graph,
                ItemSampler,
            )
            for item_sampler in item_samplers:
                datapipe_graph = replace_dp(
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

        # (3) Limit the number of UVA threads used if the feature_fetcher
        # or any of the samplers have overlapping optimization enabled.
        if num_workers == 0 and torch.cuda.is_available():
            feature_fetchers = find_dps(
                datapipe_graph,
                FeatureFetcher,
            )
            for feature_fetcher in feature_fetchers:
                if feature_fetcher.max_num_stages > 0:  # Overlap enabled.
                    torch.ops.graphbolt.set_max_uva_threads(max_uva_threads)

        if num_workers == 0 and torch.cuda.is_available():
            samplers = find_dps(
                datapipe_graph,
                SamplePerLayer,
            )
            for sampler in samplers:
                if sampler.overlap_fetch:
                    torch.ops.graphbolt.set_max_uva_threads(max_uva_threads)

        # (4) Cut datapipe at CopyTo and wrap with pinning and prefetching
        # before it. This enables enables non_blocking copies to the device.
        # Prefetching enables the data pipeline up to the CopyTo to run in a
        # separate thread.
        copiers = find_dps(datapipe_graph, CopyTo)
        if len(copiers) > 1:
            gb_warning(
                "Multiple CopyTo operations were found in the datapipe graph."
                " This case is not officially supported."
            )
        for copier in copiers:
            # We enable the prefetch at all times for good CPU only performance.
            datapipe_graph = replace_dp(
                datapipe_graph,
                copier,
                # Add prefetch so that CPU and GPU can run concurrently.
                copier.datapipe.prefetch(2).copy_to(
                    copier.device, non_blocking=True
                ),
            )

        super().__init__(datapipe)
