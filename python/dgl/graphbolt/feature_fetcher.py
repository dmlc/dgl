"""Feature fetchers"""

from torchdata.datapipes.iter import IterDataPipe


class FeatureFetcher(IterDataPipe):
    """Base feature fetcher.

    This is equivalent to the following iterator:

    .. code:: python

       for data in dp:
           yield feature_fetch_func(data)

    Parameters
    ----------
    dp : DataPipe
        The datapipe.
    feature_fetch_func : callable
        The function that performs feature fetching.
    """

    def __init__(self, dp, feature_fetch_func):
        super().__init__()
        self.dp = dp
        self.feature_fetch_func = feature_fetch_func

    def __iter__(self):
        for data in self.dp:
            yield self.feature_fetch_func(data)
