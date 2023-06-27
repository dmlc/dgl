"""Negative samplers"""

from functools import partial

from torchdata.datapipes.iter import IterDataPipe

from .graph_storage import CSCSamplingGraph
from .linked_data_format import *

__all__ = ["NegativeSampler"]


class NegativeSampler(IterDataPipe):
    """
    A negative sampler.

    It is an iterator equivalent to the following:

    .. code:: python

    for data in dp:
        yield negative_sampler_func(data, graph,negative_ratio,
            linked_data_format)

    Parameters
    ----------
    dp : IterDataPipe
        The data pipe, which always refers to a minibatch sampler.
    negative_sampler_func : callable
        The function used to generate negative samples.
    graph : CSCSamplingGraph
        The graph for negative sampling.
    negative_ratio : int
        The ratio of the number of negative samples to positive samples.
    linked_data_format : LinkedDataFormat
        The format of the output data:
            - Conditioned format: The output data is organized as quadruples
                `[u, v, [negative heads], [negative tails]]`. In this format,
                'u' and 'v' represent the source and destination nodes of
                positive edges, respectively. 'Negative heads' and
                'negative tails' signify the source and destination nodes of
                negative edges, respectively.
            - Independent format: The output data is displayed as triples
                `[u, v, label]`. Here, 'u' and 'v' denote the source and
                destination nodes of edges. The 'label' is normally set to
                0 or1. A label of0 signifies a negative edge, while 1 indicates
                a positive edge.

    """

    def __init__(
        self,
        dp: IterDataPipe,
        negative_sampler_func: callable,
        graph: CSCSamplingGraph,
        negative_ratio: int,
        linked_data_format: LinkedDataFormat,
    ):
        super().__init__()
        self.dp = dp
        self.negative_sampler_func = partial(
            negative_sampler_func,
            graph=graph,
            negative_ratio=negative_ratio,
            linked_data_format=linked_data_format,
        )

    def __iter__(self):
        """
        Iterate over the data pipe and apply the negative sampling
        function to each data item.
        """
        for data in self.dp:
            yield self.negative_sampler_func(data)
