"""Graph Bolt DataLoaders"""

import queue
import threading

import torch.utils.data
import torchdata.dataloader2.graph as dp_utils
import torchdata.datapipes as dp

from .feature_fetcher import FeatureFetcher
from .item_sampler import ItemSampler

from .utils import datapipe_graph_to_adjlist


__all__ = [
    "SingleProcessDataLoader",
    "MultiProcessDataLoader",
    "ThreadingWrapper",
]


class SingleProcessDataLoader(torch.utils.data.DataLoader):
    """Single process DataLoader.

    Iterates over the data pipeline in the main process.

    Parameters
    ----------
    datapipe : DataPipe
        The data pipeline.
    """

    # In the single process dataloader case, we don't need to do any
    # modifications to the datapipe, and we just PyTorch's native
    # dataloader as-is.
    #
    # The exception is that batch_size should be None, since we already
    # have minibatch sampling and collating in ItemSampler.
    def __init__(self, datapipe):
        super().__init__(datapipe, batch_size=None, num_workers=0)


class MultiprocessingWrapper(dp.iter.IterDataPipe):
    """Wraps a datapipe with multiprocessing.

    Parameters
    ----------
    datapipe : DataPipe
        The data pipeline.
    num_workers : int, optional
        The number of worker processes. Default is 0, meaning that there
        will be no multiprocessing.
    """

    def __init__(self, datapipe, num_workers=0):
        self.datapipe = datapipe
        self.dataloader = torch.utils.data.DataLoader(
            datapipe,
            batch_size=None,
            num_workers=num_workers,
        )

    def __iter__(self):
        yield from self.dataloader


class ThreadingWrapper(dp.iter.IterDataPipe):
    """Wraps a datapipe with a prefetch thread.

    Parameters
    ----------
    datapipe : DataPipe
        The data pipeline.
    """

    def __init__(self, datapipe, num_workers=0):
        self.datapipe = datapipe
        self.dataloader = torch.utils.data.DataLoader(
            datapipe,
            batch_size=None,
            num_workers=num_workers,
        )

    def __iter__(self):
        q = queue.Queue()

        def worker(q):
            for item in self.dataloader:
                q.put(item)
            q.put(None)

        prefetch_thread = threading.Thread(
            target=worker, args=(q,), daemon=True
        )
        prefetch_thread.start()

        while True:
            item = q.get()
            if item is not None:
                yield item
            else:
                break

        prefetch_thread.join()


class MultiProcessDataLoader(torch.utils.data.DataLoader):
    """Multiprocessing DataLoader.

    Iterates over the data pipeline with everything before feature fetching
    (i.e. :class:`dgl.graphbolt.FeatureFetcher`) in subprocesses, and
    everything after feature fetching in the main process. The datapipe
    is modified in-place as a result.

    Only works on single GPU.

    Parameters
    ----------
    datapipe : DataPipe
        The data pipeline.
    num_workers : int, optional
        Number of worker processes. Default is 0, which is identical to
        :class:`SingleProcessDataLoader`.
    """

    def __init__(self, datapipe, num_workers=0, use_prefetch_thread=False):
        # Multiprocessing requires two modifications to the datapipe:
        #
        # 1. Insert a stage after ItemSampler to distribute the
        #    minibatches evenly across processes.
        # 2. Cut the datapipe at FeatureFetcher, and wrap the inner datapipe
        #    of the FeatureFetcher with a multiprocessing PyTorch DataLoader.

        datapipe_graph = dp_utils.traverse_dps(datapipe)
        datapipe_adjlist = datapipe_graph_to_adjlist(datapipe_graph)

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
        feature_fetchers = dp_utils.find_dps(
            datapipe_graph,
            FeatureFetcher,
        )
        for feature_fetcher in feature_fetchers:
            feature_fetcher_id = id(feature_fetcher)
            for parent_datapipe_id in datapipe_adjlist[feature_fetcher_id][1]:
                parent_datapipe, _ = datapipe_adjlist[parent_datapipe_id]
                datapipe_graph = dp_utils.replace_dp(
                    datapipe_graph,
                    parent_datapipe,
                    MultiprocessingWrapper(parent_datapipe, num_workers)
                    if use_prefetch_thread == False or num_workers > 0
                    else ThreadingWrapper(parent_datapipe, num_workers),
                )

        # (3) Wrap the datapipe with a ThreadingWrapper to enable prefetching.
        if use_prefetch_thread:
            datapipe = ThreadingWrapper(datapipe, num_workers=0)

        # The stages after feature fetching is still done in the main process.
        # So we set num_workers to 0 here.
        super().__init__(datapipe, batch_size=None, num_workers=0)
