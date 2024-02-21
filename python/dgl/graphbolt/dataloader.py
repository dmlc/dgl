"""Graph Bolt DataLoaders"""

from concurrent.futures import ThreadPoolExecutor

import torch
import torch.utils.data
import torchdata.dataloader2.graph as dp_utils
import torchdata.datapipes as dp

from .base import CopyTo
from .feature_fetcher import FeatureFetcher
from .impl.neighbor_sampler import SamplePerLayer

from .internal import datapipe_graph_to_adjlist
from .item_sampler import ItemSampler


__all__ = [
    "DataLoader",
]


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


class MultiprocessingWrapper(dp.iter.IterDataPipe):
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
        self.dataloader = torch.utils.data.DataLoader(
            datapipe,
            batch_size=None,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
        )

    def __iter__(self):
        yield from self.dataloader


# There needs to be a single instance of the uva_stream, if it is created
# multiple times, it leads to multiple CUDA memory pools and memory leaks.
def _get_uva_stream():
    if not hasattr(_get_uva_stream, "stream"):
        _get_uva_stream.stream = torch.cuda.Stream(priority=-1)
    return _get_uva_stream.stream


class DataLoader(torch.utils.data.DataLoader):
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
    overlap_feature_fetch : bool, optional
        If True, the data loader will overlap the UVA feature fetcher operations
        with the rest of operations by using an alternative CUDA stream. Default
        is True.
    overlap_graph_fetch : bool, optional
        If True, the data loader will overlap the UVA graph fetching operations
        with the rest of operations by using an alternative CUDA stream. Default
        is False.
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
        overlap_feature_fetch=True,
        overlap_graph_fetch=False,
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
            FeatureFetcher,
            MultiprocessingWrapper,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

        # (3) Overlap UVA feature fetching by buffering and using an alternative
        # stream.
        if (
            overlap_feature_fetch
            and num_workers == 0
            and torch.cuda.is_available()
        ):
            torch.ops.graphbolt.set_max_uva_threads(max_uva_threads)
            feature_fetchers = dp_utils.find_dps(
                datapipe_graph,
                FeatureFetcher,
            )
            for feature_fetcher in feature_fetchers:
                feature_fetcher.stream = _get_uva_stream()
                datapipe_graph = dp_utils.replace_dp(
                    datapipe_graph,
                    feature_fetcher,
                    feature_fetcher.buffer(1).wait(),
                )

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
            executor = ThreadPoolExecutor(max_workers=1)
            for sampler in samplers:
                datapipe_graph = dp_utils.replace_dp(
                    datapipe_graph,
                    sampler,
                    sampler.fetch_and_sample(_get_uva_stream(), executor, 1),
                )

        # (4) Cut datapipe at CopyTo and wrap with prefetcher. This enables the
        # data pipeline up to the CopyTo operation to run in a separate thread.
        datapipe_graph = _find_and_wrap_parent(
            datapipe_graph,
            CopyTo,
            dp.iter.Prefetcher,
            buffer_size=2,
        )

        # The stages after feature fetching is still done in the main process.
        # So we set num_workers to 0 here.
        super().__init__(datapipe, batch_size=None, num_workers=0)
