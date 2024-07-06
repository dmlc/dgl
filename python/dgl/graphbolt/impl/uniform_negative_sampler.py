"""Uniform negative sampler for GraphBolt."""

import torch
from torch.utils.data import functional_datapipe

from ..negative_sampler import NegativeSampler

__all__ = ["UniformNegativeSampler"]


@functional_datapipe("sample_uniform_negative")
class UniformNegativeSampler(NegativeSampler):
    """Sample negative destination nodes for each source node based on a uniform
    distribution.

    Functional name: :obj:`sample_uniform_negative`.

    It's important to note that the term 'negative' refers to false negatives,
    indicating that the sampled pairs are not ensured to be absent in the graph.
    For each edge ``(u, v)``, it is supposed to generate `negative_ratio` pairs
    of negative edges ``(u, v')``, where ``v'`` is chosen uniformly from all
    the nodes in the graph.

    Parameters
    ----------
    datapipe : DataPipe
        The datapipe.
    graph : FusedCSCSamplingGraph
        The graph on which to perform negative sampling.
    negative_ratio : int
        The proportion of negative samples to positive samples.

    Examples
    --------
    >>> from dgl import graphbolt as gb
    >>> indptr = torch.LongTensor([0, 1, 2, 3, 4])
    >>> indices = torch.LongTensor([1, 2, 3, 0])
    >>> graph = gb.fused_csc_sampling_graph(indptr, indices)
    >>> seeds = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
    >>> item_set = gb.ItemSet(seeds, names="seeds")
    >>> item_sampler = gb.ItemSampler(
    ...     item_set, batch_size=4,)
    >>> neg_sampler = gb.UniformNegativeSampler(
    ...     item_sampler, graph, 2)
    >>> for minibatch in neg_sampler:
    ...       print(minibatch.seeds)
    ...       print(minibatch.labels)
    ...       print(minibatch.indexes)
    tensor([[0, 1], [1, 2], [2, 3], [3, 0], [0, 1], [0, 3], [1, 1], [1, 2],
        [2, 1], [2, 0], [3, 0], [3, 2]])
    tensor([1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
    tensor([0, 1, 2, 3, 0, 0, 1, 1, 2, 2, 3, 3])
    """

    def __init__(
        self,
        datapipe,
        graph,
        negative_ratio,
    ):
        super().__init__(datapipe, negative_ratio)
        self.graph = graph

    def _sample_with_etype(self, seeds, etype=None):
        assert seeds.ndim == 2 and seeds.shape[1] == 2, (
            "Only tensor with shape N*2 is supported for negative"
            + f" sampling, but got {seeds.shape}."
        )
        # Sample negative edges, and concatenate positive edges with them.
        all_seeds = self.graph.sample_negative_edges_uniform(
            etype,
            seeds,
            self.negative_ratio,
        )
        # Construct indexes for all node pairs.
        pos_num = seeds.shape[0]
        negative_ratio = self.negative_ratio
        pos_indexes = torch.arange(0, pos_num, device=all_seeds.device)
        neg_indexes = pos_indexes.repeat_interleave(negative_ratio)
        indexes = torch.cat((pos_indexes, neg_indexes))
        # Construct labels for all node pairs.
        neg_num = all_seeds.shape[0] - pos_num
        labels = torch.empty(pos_num + neg_num, device=all_seeds.device)
        labels[:pos_num] = 1
        labels[pos_num:] = 0
        return all_seeds, labels, indexes
