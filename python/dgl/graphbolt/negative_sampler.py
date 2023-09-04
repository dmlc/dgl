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
            Determines the edge format of the output minibatch.
        """
        super().__init__(datapipe, self._sample)
        assert negative_ratio > 0, "Negative_ratio should be positive Integer."
        self.negative_ratio = negative_ratio
        self.output_format = output_format

    def _sample(self, minibatch):
        """
        Generate a mix of positive and negative samples.

        Parameters
        ----------
        minibatch : MiniBatch
            An instance of 'MiniBatch' class requires the 'node_pairs' field.
            This function is responsible for generating negative edges
            corresponding to the positive edges defined by the 'node_pairs'. In
            cases where negative edges already exist, this function will
            overwrite them.

        Returns
        -------
        MiniBatch
            An instance of 'MiniBatch' encompasses both positive and negative
            samples.
        """
        node_pairs = minibatch.node_pairs
        assert node_pairs is not None
        if isinstance(node_pairs, Mapping):
            if self.output_format == LinkPredictionEdgeFormat.INDEPENDENT:
                minibatch.labels = {}
            else:
                minibatch.negative_head, minibatch.negative_tail = {}, {}
            for etype, pos_pairs in node_pairs.items():
                self._collate(
                    minibatch, self._sample_with_etype(pos_pairs, etype), etype
                )
            if self.output_format == LinkPredictionEdgeFormat.HEAD_CONDITIONED:
                minibatch.negative_tail = None
            if self.output_format == LinkPredictionEdgeFormat.TAIL_CONDITIONED:
                minibatch.negative_head = None
        else:
            self._collate(minibatch, self._sample_with_etype(node_pairs))
        return minibatch

    def _sample_with_etype(self, node_pairs, etype=None):
        """Generate negative pairs for a given etype form positive pairs
        for a given etype.

        Parameters
        ----------
        node_pairs : Tuple[Tensor, Tensor]
            A tuple of tensors that represent source-destination node pairs of
            positive edges, where positive means the edge must exist in the
            graph.
        etype : str
            Canonical edge type.

        Returns
        -------
        Tuple[Tensor, Tensor]
            A collection of negative node pairs.
        """
        raise NotImplementedError

    def _collate(self, minibatch, neg_pairs, etype=None):
        """Collates positive and negative samples into minibatch.

        Parameters
        ----------
        minibatch : MiniBatch
            The input minibatch, which contains positive node pairs, will be filled
            with negative information in this function.
        neg_pairs : Tuple[Tensor, Tensor]
            A tuple of tensors represents source-destination node pairs of
            negative edges, where negative means the edge may not exist in
            the graph.
        etype : str
            Canonical edge type.
        """
        pos_src, pos_dst = (
            minibatch.node_pairs[etype]
            if etype is not None
            else minibatch.node_pairs
        )
        neg_src, neg_dst = neg_pairs
        if self.output_format == LinkPredictionEdgeFormat.INDEPENDENT:
            pos_labels = torch.ones_like(pos_src)
            neg_labels = torch.zeros_like(neg_src)
            src = torch.cat([pos_src, neg_src])
            dst = torch.cat([pos_dst, neg_dst])
            labels = torch.cat([pos_labels, neg_labels])
            if etype is not None:
                minibatch.node_pairs[etype] = (src, dst)
                minibatch.labels[etype] = labels
            else:
                minibatch.node_pairs = (src, dst)
                minibatch.labels = labels
        else:
            if self.output_format == LinkPredictionEdgeFormat.CONDITIONED:
                neg_src = neg_src.view(-1, self.negative_ratio)
                neg_dst = neg_dst.view(-1, self.negative_ratio)
            elif (
                self.output_format == LinkPredictionEdgeFormat.HEAD_CONDITIONED
            ):
                neg_src = neg_src.view(-1, self.negative_ratio)
                neg_dst = None
            elif (
                self.output_format == LinkPredictionEdgeFormat.TAIL_CONDITIONED
            ):
                neg_dst = neg_dst.view(-1, self.negative_ratio)
                neg_src = None
            else:
                raise TypeError(
                    f"Unsupported output format {self.output_format}."
                )
            if etype is not None:
                minibatch.negative_head[etype] = neg_src
                minibatch.negative_tail[etype] = neg_dst
            else:
                minibatch.negative_head = neg_src
                minibatch.negative_tail = neg_dst
