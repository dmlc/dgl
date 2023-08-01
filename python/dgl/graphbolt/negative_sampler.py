"""Negative samplers."""

from _collections_abc import Mapping

import torch
from torchdata.datapipes.iter import Mapper

from .data_format import LinkPredictionEdgeFormat


class NegativeSampler(Mapper):
    """
    A negative sampler used to generate negative samples and return
    a mix of positive and negative samples.
    """

    def __init__(
        self,
        datapipe,
        negative_ratio,
        output_format,
    ):
        """
        Initlization for a negative sampler.

        Parameters
        ----------
        datapipe : DataPipe
            The datapipe.
        negative_ratio : int
            The proportion of negative samples to positive samples.
        output_format : LinkPredictionEdgeFormat
            Determines the edge format of the output data:
                - Conditioned format: Outputs data as quadruples
                `[u, v, [negative heads], [negative tails]]`. Here, 'u' and 'v'
                are the source and destination nodes of positive edges,  while
                'negative heads' and 'negative tails' refer to the source and
                destination nodes of negative edges.
                - Independent format: Outputs data as triples `[u, v, label]`.
                In this case, 'u' and 'v' are the source and destination nodes
                of an edge, and 'label' indicates whether the edge is negative
                (0) or positive (1).
        """
        super().__init__(datapipe, self._sample)
        assert negative_ratio > 0, "Negative_ratio should be positive Integer."
        self.negative_ratio = negative_ratio
        self.output_format = output_format

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
                etype: self._collate(
                    pos_pairs, self._sample_with_etype(pos_pairs, etype)
                )
                for etype, pos_pairs in node_pairs.items()
            }
        else:
            return self._collate(
                node_pairs, self._sample_with_etype(node_pairs, None)
            )

    def _sample_with_etype(self, node_pairs, etype=None):
        """Generate negative pairs for a given etype form positive pairs
        for a given etype.

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

    def _collate(self, pos_pairs, neg_pairs):
        """Collates positive and negative samples.

        Parameters
        ----------
        pos_pairs : Tuple[Tensor]
            A tuple of tensors represents source-destination node pairs of
            positive edges, where positive means the edge must exist in
            the graph.
        neg_pairs : Tuple[Tensor]
            A tuple of tensors represents source-destination node pairs of
            negative edges, where negative means the edge may not exist in
            the graph.

        Returns
        -------
        Tuple[Tensor]
            A mixed collection of positive and negative node pairs.
        """
        if self.output_format == LinkPredictionEdgeFormat.INDEPENDENT:
            pos_src, pos_dst = pos_pairs
            neg_src, neg_dst = neg_pairs
            pos_label = torch.ones_like(pos_src)
            neg_label = torch.zeros_like(neg_src)
            src = torch.cat([pos_src, neg_src])
            dst = torch.cat([pos_dst, neg_dst])
            label = torch.cat([pos_label, neg_label])
            return (src, dst, label)
        elif self.output_format == LinkPredictionEdgeFormat.CONDITIONED:
            pos_src, pos_dst = pos_pairs
            neg_src, neg_dst = neg_pairs
            neg_src = neg_src.view(-1, self.negative_ratio)
            neg_dst = neg_dst.view(-1, self.negative_ratio)
            return (pos_src, pos_dst, neg_src, neg_dst)
        else:
            raise ValueError("Unsupported output format.")
