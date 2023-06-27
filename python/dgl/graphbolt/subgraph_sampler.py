"""Subgraph samplers"""

from torchdata.datapipes.iter import IterDataPipe


class SubgraphSampler(IterDataPipe):
    """A subgraph sampler.

    It is an iterator equivalent to the following:

    .. code:: python

       for data in dp:
           yield sampler_func(graph, data)

    Parameters
    ----------
    dp : DataPipe
        The datapipe.
    graph : GraphStorage
        The graph storage.
    sampler_func : callable
        The subgraph sampling function.
    """

    def __init__(self, dp, graph, sampler_func):
        super().__init__()
        self.graph = graph
        self.dp = dp
        self.sampler_func = sampler_func

    def __iter__(self):
        for data in self.dp:
            yield self.sampler_func(self.graph, data)
