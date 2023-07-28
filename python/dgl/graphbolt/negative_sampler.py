"""Negative samplers."""

from _collections_abc import Mapping

import torch
from torchdata.datapipes.iter import Mapper


class NegativeSampler(Mapper):
    """
    A negative sampler used to generate negative samples and return
    a mix of positive and negative samples.
    """

    def __init__(
        self,
        datapipe,
        negative_ratio,
    ):
        """
        Initlization for a negative sampler.

        Parameters
        ----------
        datapipe : DataPipe
            The datapipe.
        negative_ratio : int
            The proportion of negative samples to positive samples.
        """
        super().__init__(datapipe, self._sample)
        assert negative_ratio > 0, "Negative_ratio should be positive Integer."
        self.negative_ratio = negative_ratio

    def _sample(self, node_pairs):
        """
        Generate a mix of positive and negative samples.

        Parameters
        ----------
        node_pairs : Tuple[Tensor] or Dict[etype, Tuple[Tensor]]
            A tuple of tensors or a dictionary represents source-destination
            node pairs of positive edges, where positive means the edge must
            exist in the graph.

        Returns
        -------
        Tuple[Tensor] or Dict[etype, Tuple[Tensor]]
            A collection of edges or a dictionary that maps etypes to edges,
            which includes both positive and negative samples.
        """
        if isinstance(node_pairs, Mapping):
            return {
                etype: self._generate(pos_pairs, etype)
                for etype, pos_pairs in node_pairs.items()
            }
        else:
            return self._generate(node_pairs, None)

    def _generate(self, node_pairs, etype=None):
        """Generate a mix of positive and negative samples for a given etype.

        Parameters
        ----------
        node_pairs : Tuple[Tensor]
            A tuple of tensors or a dictionary represents source-destination
            node pairs of positive edges, where positive means the edge must
            exist in the graph.
        etype : (str, str, str)
            Canonical edge type.
        Returns
        -------
        Tuple[Tensor]
            A mixed collection of positive and negative node pairs.
        """
        raise NotImplementedError

    def _generate_negative_pairs(self, node_pairs, etype=None):
        """Generate negative pairs for a given etype form positive pairs.
        Parameters
        ----------
        node_pairs : Tuple[Tensor]
            A tuple of tensors or a dictionary represents source-destination
            node pairs of positive edges, where positive means the edge must
            exist in the graph.
        etype : (str, str, str)
            Canonical edge type.
        Returns
        -------
        Tuple[Tensor]
            A collection of negative node pairs.
        """
        raise NotImplementedError


class IndependentNegativeSampler(NegativeSampler):
    """
    A kind of negative sampler. `Independent` denotes the data format, where
    data is structured as triples `[u, v, label]`. Each triple represents an
    edge between nodes 'u' and 'v', and the 'label' indicates whether the edge
    is negative (0) or positive (1).
    """

    def _generate(self, node_pairs, etype=None):
        neg_src, neg_dst = self._generate_negative_pairs(node_pairs, etype)
        pos_src, pos_dst = node_pairs
        pos_label = torch.ones_like(pos_src)
        neg_label = torch.zeros_like(neg_src)
        src = torch.cat([pos_src, neg_src])
        dst = torch.cat([pos_dst, neg_dst])
        label = torch.cat([pos_label, neg_label])
        return (src, dst, label)


class ConditionedNegativeSampler(NegativeSampler):
    """
    A kind of negative sampler. `Conditioned` denotes the data format, where
    data is structured as `[u, v, [neg_u], [neg_v]]`. Here, 'u' and 'v' denote
    the source-destination positive node pairs, while 'neg_u' combined with
    'neg_v' creates corresponding negative node pairs. The length of 'neg_u'
    and 'neg_v' is same as 'negative_ratio'.
    """

    def _generate(self, node_pairs, etype=None):
        neg_src, neg_dst = self._generate_negative_pairs(node_pairs, etype)
        pos_src, pos_dst = node_pairs
        neg_src = neg_src.view(-1, self.negative_ratio)
        neg_dst = neg_dst.view(-1, self.negative_ratio)
        return (pos_src, pos_dst, neg_src, neg_dst)
