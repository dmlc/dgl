"""Feature fetchers"""

from torchdata.datapipes.iter import IterDataPipe


class FeatureFetcher(IterDataPipe):
    """A mock feature fetcher."""

    def __init__(self, dp, feature_fetch_func):
        super().__init__()
        self.dp = dp
        self.feature_fetch_func = feature_fetch_func

    def __iter__(self):
        for data in self.dp:
            yield self.feature_fetch_func(data)
