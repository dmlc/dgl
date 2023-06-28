"""Feature fetchers"""

from torchdata.datapipes.iter import Mapper


class FeatureFetcher(Mapper):
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
        super().__init__(dp, feature_fetch_func)
