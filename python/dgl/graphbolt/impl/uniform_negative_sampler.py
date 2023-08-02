"""Uniform negative sampler for GraphBolt."""

from ..negative_sampler import NegativeSampler


class UniformNegativeSampler(NegativeSampler):
    """
    Negative samplers randomly select negative destination nodes for each
    source node based on a uniform distribution. It's important to note that
    the term 'negative' refers to false negatives, indicating that the sampled
    pairs are not ensured to be absent in the graph.
    For each edge ``(u, v)``, it is supposed to generate `negative_ratio` pairs
    of negative edges ``(u, v')``, where ``v'`` is chosen uniformly from all
    the nodes in the graph.
    """

    def __init__(
        self,
        datapipe,
        negative_ratio,
        output_format,
        graph,
    ):
        """
        Initlization for a uniform negative sampler.

        Parameters
        ----------
        datapipe : DataPipe
            The datapipe.
        negative_ratio : int
            The proportion of negative samples to positive samples.
        output_format : LinkPredictionEdgeFormat
            Determines the format of the output data:
                - Conditioned format: Outputs data as quadruples
                `[u, v, [negative heads], [negative tails]]`. Here, 'u' and 'v'
                are the source and destination nodes of positive edges, while
                'negative heads' and 'negative tails' refer to the source and
                destination nodes of negative edges.
                - Independent format: Outputs data as triples `[u, v, label]`.
                In this case, 'u' and 'v' are the source and destination nodes
                of an edge, and 'label' indicates whether the edge is negative
                (0) or positive (1).
        graph : CSCSamplingGraph
            The graph on which to perform negative sampling.

        Examples
        --------
        >>> from dgl import graphbolt as gb
        >>> indptr = torch.LongTensor([0, 2, 4, 5])
        >>> indices = torch.LongTensor([1, 2, 0, 2, 0])
        >>> graph = gb.from_csc(indptr, indices)
        >>> output_format = gb.LinkPredictionEdgeFormat.INDEPENDENT
        >>> node_pairs = (torch.tensor([0, 1]), torch.tensor([1, 2]))
        >>> item_set = gb.ItemSet(node_pairs)
        >>> minibatch_sampler = gb.MinibatchSampler(
            ...item_set, batch_size=1,
            ...)
        >>> neg_sampler = gb.UniformNegativeSampler(
            ...minibatch_sampler, 2, output_format, graph)
        >>> for data in neg_sampler:
            ...  print(data)
            ...
        (tensor([0, 0, 0]), tensor([1, 1, 2]), tensor([1, 0, 0]))
        (tensor([1, 1, 1]), tensor([2, 1, 2]), tensor([1, 0, 0]))

        >>> from dgl import graphbolt as gb
        >>> indptr = torch.LongTensor([0, 2, 4, 5])
        >>> indices = torch.LongTensor([1, 2, 0, 2, 0])
        >>> graph = gb.from_csc(indptr, indices)
        >>> output_format = gb.LinkPredictionEdgeFormat.CONDITIONED
        >>> node_pairs = (torch.tensor([0, 1]), torch.tensor([1, 2]))
        >>> item_set = gb.ItemSet(node_pairs)
        >>> minibatch_sampler = gb.MinibatchSampler(
            ...item_set, batch_size=1,
            ...)
        >>> neg_sampler = gb.UniformNegativeSampler(
            ...minibatch_sampler, 2, output_format, graph)
        >>> for data in neg_sampler:
            ...  print(data)
            ...
        (tensor([0]), tensor([1]), tensor([[0, 0]]), tensor([[2, 1]]))
        (tensor([1]), tensor([2]), tensor([[1, 1]]), tensor([[1, 2]]))
        """
        super().__init__(datapipe, negative_ratio, output_format)
        self.graph = graph

    def _sample_with_etype(self, node_pairs, etype=None):
        return self.graph.sample_negative_edges_uniform(
            etype,
            node_pairs,
            self.negative_ratio,
        )
