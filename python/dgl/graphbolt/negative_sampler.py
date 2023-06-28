"""Negative samplers"""

from functools import partial

from torchdata.datapipes.iter import IterDataPipe

from .graph_storage import CSCSamplingGraph
from .linked_data_format import LinkedDataFormat

__all__ = ["NegativeSampler"]


class NegativeSampler(IterDataPipe):
    """
    A negative sampler.

    It is an iterator equivalent to the following:

    .. code:: python

    for data in data_pipe:
        yield negative_sample_generator(data, graph,negative_ratio,
            linked_data_format)

    Parameters
    ----------
    data_pipe : IterDataPipe
        The data pipe, which always refers to a minibatch sampler.
    negative_sample_generator : callable
        A callable used to generate negative samples.
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
                0 or 1. A label of 0 signifies a negative edge, while 1
                indicates a positive edge.

    """

    def __init__(
        self,
        data_pipe: IterDataPipe,
        negative_sample_generator: callable,
        graph: CSCSamplingGraph,
        negative_ratio: int,
        linked_data_format: LinkedDataFormat,
    ):
        super().__init__()
        self.data_pipe = data_pipe
        self.negative_sample_generator = partial(
            negative_sample_generator,
            graph=graph,
            negative_ratio=negative_ratio,
            linked_data_format=linked_data_format,
        )

    def __iter__(self):
        """
        Iterate over the data pipe and apply the negative sampling
        function to each data item.
        """
        for data in self.data_pipe:
            yield self.negative_sample_generator(data)
