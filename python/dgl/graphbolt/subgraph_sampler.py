"""Subgraph samplers"""

import torch
from torchdata.datapipes.iter import Mapper

from .data_format import LinkPredictionEdgeFormat
from .utils import unique_and_compact_node_pairs


class SubgraphSampler(Mapper):
    """A subgraph sampler.

    It is an iterator equivalent to the following:

    .. code:: python

       for data in datapipe:
           yield _sample(data)

    Parameters
    ----------
    datapipe : DataPipe
        The datapipe.
    fn : callable
        The subgraph sampling function.
    """

    def __init__(
        self,
        datapipe,
        fanouts,
        replace=False,
        prob_name=None,
    ):
        """
        Initlization for a subgraph sampler.

        Parameters
        ----------
        datapipe : DataPipe
            The datapipe.
        fanouts: list[list[int]]
            The number of edges to be sampled for each node with or without
            considering edge types.
        replace: bool
            Boolean indicating whether the sample is preformed with or
            without replacement. If True, a value can be selected multiple
            times. Otherwise, each value can be selected only once.
        prob_name: str, optional
            The name of an edge attribute used a. This
            attribute tensor should contain (unnormalized) probabilities
            corresponding to each neighboring edge of a node. It must be a 1D
            floating-point or boolean tensor, with the number of elements
            equalling the total number of edges.
        """
        super().__init__(datapipe, self._sample)
        self.fanouts = fanouts
        self.replace = replace
        self.prob_name = prob_name

    def _sample(self, data):
        adjs = []
        num_layers = len(self.fanouts)
        seeds = data
        for hop in range(num_layers):
            subgraph = self._sample_sub_graph(
                seeds,
                hop,
            )
            seeds, compacted_subgraph = unique_and_compact_node_pairs(
                subgraph.node_pairs, seeds
            )
            adjs.insert(0, compacted_subgraph)
        return seeds, adjs

    def _sample_sub_graph(self, seeds, hop):
        raise NotImplemented


class LinkSampler(SubgraphSampler):
    def __init__(
        self,
        datapipe,
        input_format,
        fanouts,
        replace,
        prob_name,
    ):
        """
        Initlization for a link prediction oriented subgraph sampler.

        Parameters
        ----------
        datapipe : DataPipe
            The datapipe.
        input_format:  LinkPredictionEdgeFormat
            Determines the edge format of the input data.
        fanouts: list[list[int]]
            The number of edges to be sampled for each node with or without
            considering edge types.
        replace: bool
            Boolean indicating whether the sample is preformed with or
            without replacement. If True, a value can be selected multiple
            times. Otherwise, each value can be selected only once.
        prob_name: str, optional
            The name of an edge attribute used a. This
            attribute tensor should contain (unnormalized) probabilities
            corresponding to each neighboring edge of a node. It must be a 1D
            floating-point or boolean tensor, with the number of elements
            equalling the total number of edges.
        """
        super().__init__(datapipe, fanouts, replace, prob_name)
        self.input_format = input_format

        def _sample(self, data):
            seeds, compacted_pairs = self._preprocess(data)
            seeds, adjs = super._sample(seeds)
            return seeds, compacted_pairs, adjs

        def _preprocess(self, data):
            u, v = data[:2]
            if self.input_format == LinkPredictionEdgeFormat.CONDITIONED:
                neg_u, neg_v = data[2:4]
                u = torch.cat((u, neg_u.view(-1)))
                v = torch.cat((v, neg_v.view(-1)))
            elif self.input_format == LinkPredictionEdgeFormat.HEAD_CONDITIONED:
                neg_u = data[2]
                u = torch.cat((u, neg_u.view(-1)))
            elif self.input_format == LinkPredictionEdgeFormat.TAIL_CONDITIONED:
                neg_v = data[2]
                v = torch.cat((v, neg_v.view(-1)))
            seeds, compacted_pairs = unique_and_compact_node_pairs((u, v))
            return seeds, compacted_pairs
