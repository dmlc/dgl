"""Feature fetchers"""

from torchdata.datapipes.iter import Mapper


class FeatureFetcher(Mapper):
    """Base feature fetcher.

    This is equivalent to the following iterator:

    .. code:: python

       for data in datapipe:
           yield feature_fetch_func(data)

    Parameters
    ----------
    datapipe : DataPipe
        The datapipe.
    fn : callable
        The function that performs feature fetching.
    """
