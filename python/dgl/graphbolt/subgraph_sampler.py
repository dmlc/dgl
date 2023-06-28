"""Subgraph samplers"""

from torchdata.datapipes.iter import Mapper


class SubgraphSampler(Mapper):
    """A subgraph sampler.

    It is an iterator equivalent to the following:

    .. code:: python

       for data in dp:
           yield sampler_func(data)

    Parameters
    ----------
    dp : DataPipe
        The datapipe.
    sampler_func : callable
        The subgraph sampling function.
    """

    def __init__(self, dp, sampler_func):
        super().__init__(dp, sampler_func)
