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
    >>> node_pairs = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
    >>> item_set = gb.ItemSet(node_pairs, names="node_pairs")
    >>> item_sampler = gb.ItemSampler(
    ...     item_set, batch_size=4,)
    >>> neg_sampler = gb.UniformNegativeSampler(
    ...     item_sampler, graph, 2)
    >>> for minibatch in neg_sampler:
    ...       print(minibatch.negative_srcs)
    ...       print(minibatch.negative_dsts)
    None
    tensor([[2, 1],
        [2, 1],
        [3, 2],
        [1, 3]])
    """

    def __init__(
        self,
        datapipe,
        graph,
        negative_ratio,
    ):
        super().__init__(datapipe, negative_ratio)
        self.graph = graph

    def _sample_with_etype(self, node_pairs, etype=None, use_seeds=False):
        if use_seeds:
            assert node_pairs.ndim == 2 and node_pairs.shape[1] == 2, (
                "Only tensor with shape N*2 is supported for negative"
                + f" sampling, but got {node_pairs.shape}."
            )
            # Sample negative edges, and concatenate positive edges with them.
            seeds = self.graph.sample_negative_edges_uniform_2(
                etype,
                node_pairs,
                self.negative_ratio,
            )
            # Construct indexes for all node pairs.
            num_pos_node_pairs = node_pairs.shape[0]
            negative_ratio = self.negative_ratio
            pos_indexes = torch.arange(
                0,
                num_pos_node_pairs,
                device=seeds.device,
            )
            neg_indexes = pos_indexes.repeat_interleave(negative_ratio)
            indexes = torch.cat((pos_indexes, neg_indexes))
            # Construct labels for all node pairs.
            pos_num = node_pairs.shape[0]
            neg_num = seeds.shape[0] - pos_num
            labels = torch.empty(pos_num + neg_num, device=seeds.device)
            labels[:pos_num] = 1
            labels[pos_num:] = 0
            return seeds, labels, indexes
        else:
            return self.graph.sample_negative_edges_uniform(
                etype,
                node_pairs,
                self.negative_ratio,
            )
