"""Negative samplers."""

from _collections_abc import Mapping

import torch

from torch.utils.data import functional_datapipe

from .minibatch_transformer import MiniBatchTransformer

__all__ = [
    "NegativeSampler",
]


@functional_datapipe("sample_negative")
class NegativeSampler(MiniBatchTransformer):
    """
    A negative sampler used to generate negative samples and return
    a mix of positive and negative samples.

    Functional name: :obj:`sample_negative`.

    Parameters
    ----------
    datapipe : DataPipe
        The datapipe.
    negative_ratio : int
        The proportion of negative samples to positive samples.
    """

    def __init__(
        self,
        datapipe,
        negative_ratio,
    ):
        super().__init__(datapipe, self._sample)
        assert negative_ratio > 0, "Negative_ratio should be positive Integer."
        self.negative_ratio = negative_ratio

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
        if minibatch.seeds is None:
            node_pairs = minibatch.node_pairs
            assert node_pairs is not None
            if isinstance(node_pairs, Mapping):
                minibatch.negative_srcs, minibatch.negative_dsts = {}, {}
                for etype, pos_pairs in node_pairs.items():
                    self._collate(
                        minibatch,
                        self._sample_with_etype(pos_pairs, etype),
                        etype,
                    )
            else:
                self._collate(minibatch, self._sample_with_etype(node_pairs))
        else:
            seeds = minibatch.seeds
            assert (
                len(seeds.shape) == 2 and seeds.shape[1] == 2
            ), "Only 2-D tensor representing node paris is supported for negative sampling."
            if isinstance(seeds, Mapping):
                for etype, pos_pairs in seeds.items():
                    self._collate(
                        minibatch,
                        self._sample_with_etype(
                            pos_pairs, etype, use_seeds=True
                        ),
                        etype,
                    )
                    minibatch.indexes[etype] = self._construct_indexes(
                        pos_pairs
                    )
            else:
                self._collate(
                    minibatch, self._sample_with_etype(seeds, use_seeds=True)
                )
                minibatch.indexes = self._construct_indexes(seeds)
        return minibatch

    def _sample_with_etype(self, node_pairs, etype=None, use_seeds=False):
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

    def _construct_indexes(self, node_pairs):
        """Generate indexes for postive and negative edges. Positve edge and
        the negative edges sampled from it will have same query.

        Parameters
        ----------
        node_pairs: torch.Tensor
            A N*2 tensor representing N positive edges.

        Returns
        -------
        torch.Tensor
            The indexes indicate to which query the edge belongs.
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
        if minibatch.seeds is None:
            neg_src, neg_dst = neg_pairs
            if neg_src is not None:
                neg_src = neg_src.view(-1, self.negative_ratio)
            if neg_dst is not None:
                neg_dst = neg_dst.view(-1, self.negative_ratio)
            if etype is not None:
                minibatch.negative_srcs[etype] = neg_src
                minibatch.negative_dsts[etype] = neg_dst
            else:
                minibatch.negative_srcs = neg_src
                minibatch.negative_dsts = neg_dst
        else:
            neg_labels = torch.zeros(neg_pairs.shape[0])
            if etype is None:
                if minibatch.labels is None:
                    minibatch.labels = torch.ones(minibatch.seeds.shape[0])
                minibatch.labels = torch.cat((minibatch.labels, neg_labels))
                minibatch.seeds = torch.cat((minibatch.seeds, neg_pairs))
            else:
                if minibatch.labels[etype] is None:
                    minibatch.labels[etype] = torch.ones(
                        minibatch.seeds[etype].shape[0]
                    )
                minibatch.labels[etype] = torch.cat(
                    (minibatch.labels[etype], neg_labels)
                )
                minibatch.seeds[etype] = torch.cat(
                    (minibatch.seeds[etype], neg_pairs)
                )
