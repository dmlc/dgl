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
    fn : callable
        The subgraph sampling function.
    """
    pass
