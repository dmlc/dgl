"""Negative samplers"""

from functools import partial

import torch
from torchdata.datapipes.iter import IterDataPipe

from .graph_storage import CSCSamplingGraph
from .linked_data_format import LinkedDataFormat

__all__ = ["NegativeSampler", "PerSourceUniformGenerator"]


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


class _BaseNegativeSampleGenerator:
    def _generate(self, pos_pairs, graph, negative_ratio):
        raise NotImplementedError

    def __call__(self, pos_pairs, graph, negative_ratio, linked_data_format):
        """
        Generates a mix of positive and negative samples, the format of which
        depends on the specified `linked_data_format`.

        Parameters
        ----------
        pos_pairs : Iterable[Tensor]
            Represents source-destination positive node pairs.
        graph : CSCSamplingGraph
            The graph utilized for negative sampling.
        negative_ratio : int
            The proportion of negative samples to positive samples.
        linked_data_format : LinkedDataFormat
            Determines the format of the output data:
                - Conditioned format: Outputs data as quadruples
                `[u, v, [negative heads], [negative tails]]`. Here, 'u' and 'v'
                are the source and destination nodes of positive edges,  while
                'negative heads' and 'negative tails' refer to the source and
                destination nodes of negative edges.
                - Independent format: Outputs data as triples `[u, v, label]`.
                In this case, 'u' and 'v' are the source and destination nodes
                of an edge, and 'label' indicates whether the edge is negative
                (0) or positive (1).

        Returns
        -------
        Iterable
            An iterable of edges, which includes both positive and negative
            samples. The format of it is determined by the provided
            `linked_data_format`.
        """

        neg_src, neg_dst = self._generate(pos_pairs, graph, negative_ratio)
        pos_src, pos_dst = pos_pairs
        if linked_data_format == LinkedDataFormat.INDEPENDENT:
            pos_labels = torch.ones_like(pos_src)
            neg_labels = torch.zeros_like(neg_src)
            src = torch.cat([pos_src, neg_src])
            dst = torch.cat([pos_dst, neg_dst])
            labels = torch.cat([pos_labels, neg_labels])
            return (src, dst, labels)
        elif linked_data_format == LinkedDataFormat.CONDITIONED:
            neg_src = neg_src.view(-1, negative_ratio)
            neg_dst = neg_dst.view(-1, negative_ratio)
            return (src, dst, neg_src, neg_dst)
        else:
            raise ValueError(f"Unsupported data format: {linked_data_format}")


class PerSourceUniformGenerator(_BaseNegativeSampleGenerator):
    """Negative samples generator that randomly chooses negative destination
    nodes for each source node according to a uniform distribution.

    For each edge ``(u, v)``, it is supposed to generate `negative_ratio` pairs
    of negative edges ``(u, v')``, where ``v'`` is chosen uniformly from all
    the nodes in the graph.

    Examples
    --------
    >>> from dgl import graphbolt as gb
    >>> indptr = torch.LongTensor([0, 2, 4, 5])
    >>> indices = torch.LongTensor([1, 2, 0, 2, 0])
    >>> graph = gb.from_csc(indptr, indices)
    >>> generator = gb.negative_sampler.PerSourceUniformGenerator()
    >>> pos_pairs = (torch.tensor([0, 1]), torch.tensor([1, 2]))
    >>> generator(g, pos_pairs, 2, LinkedDataFormat.INDEPENDENT)
    (tensor([0, 0, 1, 1]), tensor([1, 2, 0, 1]))
    """

    def _generate(self, pos_pairs, graph, negative_ratio):
        src, _ = pos_pairs
        shape, dtype = src.shape, src.dtype
        shape = (shape[0] * negative_ratio,)
        neg_src = src.repeat(negative_ratio)
        neg_dst = torch.randint(0, graph.num_nodes, shape, dtype=dtype)
        return (neg_src, neg_dst)
