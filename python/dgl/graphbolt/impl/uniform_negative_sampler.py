"""Uniform negative sampler for GraphBolt."""

from torch.utils.data import functional_datapipe

from ..negative_sampler import NegativeSampler


@functional_datapipe("sample_uniform_negative")
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
        graph,
        negative_ratio,
    ):
        """
        Initlization for a uniform negative sampler.

        Parameters
        ----------
        datapipe : DataPipe
            The datapipe.
        graph : CSCSamplingGraph
            The graph on which to perform negative sampling.
        negative_ratio : int
            The proportion of negative samples to positive samples.

        Examples
        --------
        >>> from dgl import graphbolt as gb
        >>> indptr = torch.LongTensor([0, 2, 4, 5])
        >>> indices = torch.LongTensor([1, 2, 0, 2, 0])
        >>> graph = gb.from_csc(indptr, indices)
        >>> node_pairs = (torch.tensor([0, 1]), torch.tensor([1, 2]))
        >>> item_set = gb.ItemSet(node_pairs, names="node_pairs")
        >>> item_sampler = gb.ItemSampler(
            ...item_set, batch_size=1,
            ...)
        >>> neg_sampler = gb.UniformNegativeSampler(
            ...item_sampler, graph, 2)
        >>> for minibatch in neg_sampler:
            ...  print(minibatch.negative_srcs)
            ...  print(minibatch.negative_dsts)
            ...
        (tensor([0, 0, 0]), tensor([1, 1, 2]), tensor([1, 0, 0]))
        (tensor([1, 1, 1]), tensor([2, 1, 2]), tensor([1, 0, 0]))
        """
        super().__init__(datapipe, negative_ratio)
        self.graph = graph

    def _sample_with_etype(self, node_pairs, etype=None):
        return self.graph.sample_negative_edges_uniform(
            etype,
            node_pairs,
            self.negative_ratio,
        )
