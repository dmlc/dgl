"""Graph Bolt DataLoaders"""

import os

import torch

from .cuda import *
from .datapipe_utils import *
from .feature_fetcher import *
from .thread_wrapper import *

prefetcher_timeout = int(os.environ.get("DGL_PREFETCHER_TIMEOUT", "30"))


class MultiprocessingDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        sampler_dp,
        feature_fetch_func,
        num_workers=0,
        device=None,
        stream=None,
    ):
        subgraph_sampler_dl = torch.utils.data.DataLoader(
            sampler_dp,
            batch_size=None,
            num_workers=num_workers,
        )

        fetcher_dp = FeatureFetcher(subgraph_sampler_dl, feature_fetch_func)
        if stream is not None:
            fetcher_dp = CopyToDevice(
                fetcher_dp,
                device,
                stream=stream,
            )
            fetcher_dp = ThreadWrapper(
                fetcher_dp,
                torch_num_threads=torch.get_num_threads(),
                timeout=prefetcher_timeout,
            )
            wait_stream_event = WaitStreamEvent(fetcher_dp)
            super().__init__(wait_stream_event, batch_size=None, num_workers=0)
        else:
            fetcher_dp = CopyToDevice(fetcher_dp, device)
            fetcher_dp = ThreadWrapper(
                fetcher_dp,
                torch_num_threads=torch.get_num_threads(),
                timeout=prefetcher_timeout,
            )
            super().__init__(fetcher_dp, batch_size=None, num_workers=0)


class MockDataLoader(torch.utils.data.DataLoader):
    def __new__(
        cls,
        sampler_dp,
        feature_fetch_func,
        num_workers=0,
        device=None,
        stream=None,
    ):
        return MultiprocessingDataLoader(
            sampler_dp,
            feature_fetch_func,
            num_workers=num_workers,
            device=device,
            stream=stream,
        )
