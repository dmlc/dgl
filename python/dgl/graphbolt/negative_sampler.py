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
            Determines the edge format of the output data.
        """
        super().__init__(datapipe, self._sample)
        assert negative_ratio > 0, "Negative_ratio should be positive Integer."
        self.negative_ratio = negative_ratio
        self.output_format = output_format

    def _sample(self, data):
        """
        Generate a mix of positive and negative samples.

        Parameters
        ----------
        LinkPredictionBlock : Tuple[Tensor] or Dict[etype, Tuple[Tensor]]
            An instance of 'LinkPredictionBlock', where 'node_pair' is required
            while 'negative_head' and 'negative_tail' should be None. If either
            of them has value, it inidcates there have already negative samples
            , thus the function will do nothing.

        Returns
        -------
        LinkPredictionBlock
            An instance of 'LinkPredictionBlock' encompasses both positive and
            negative samples.
        """
        if data.negative_head is None and data.negative_tail is None:
            node_pairs = data.node_pair
            if isinstance(node_pairs, Mapping):
                for etype, pos_pairs in node_pairs.items():
                    self._collate(
                        data, self._sample_with_etype(pos_pairs, etype), etype
                    )
            else:
                self._collate(data, self._sample_with_etype(node_pairs))
        return data

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

    def _collate(self, data, neg_pairs, etype=None):
        """Collates positive and negative samples into data.

        Parameters
        ----------
        data : LinkPredictionBlock
            The input data, which contains positive node pairs, will be filled
            with negative information in this function.
        neg_pairs : Tuple[Tensor]
            A tuple of tensors represents source-destination node pairs of
            negative edges, where negative means the edge may not exist in
            the graph.
        etype : (str, str, str)
            Canonical edge type.
        """
        pos_src, pos_dst = data.node_pair
        neg_src, neg_dst = neg_pairs
        if self.output_format == LinkPredictionEdgeFormat.INDEPENDENT:
            pos_label = torch.ones_like(pos_src)
            neg_label = torch.zeros_like(neg_src)
            src = torch.cat([pos_src, neg_src])
            dst = torch.cat([pos_dst, neg_dst])
            label = torch.cat([pos_label, neg_label])
            if etype:
                data.node_pair[etype] = (src, dst)
                data.label[etype] = label
            else:
                data.node_pair = (src, dst)
                data.label = label
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
            if etype:
                data.negative_head[etype] = neg_src
                data.negative_tail[etype] = neg_dst
            else:
                data.negative_head = neg_src
                data.negative_tail = neg_dst
