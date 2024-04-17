"""Negative samplers."""

from _collections_abc import Mapping

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
        Generate a mix of positive and negative samples. If `seeds` in
        minibatch is not None, `labels` and `indexes` will be constructed
        after negative sampling, based on corresponding seeds.

        Parameters
        ----------
        minibatch : MiniBatch
            An instance of 'MiniBatch' class requires the 'seeds' field. This
            function is responsible for generating negative edges corresponding
            to the positive edges defined by the 'seeds'.

        Returns
        -------
        MiniBatch
            An instance of 'MiniBatch' encompasses both positive and negative
            samples.
        """
        seeds = minibatch.seeds
        if isinstance(seeds, Mapping):
            if minibatch.indexes is None:
                minibatch.indexes = {}
            if minibatch.labels is None:
                minibatch.labels = {}
            for etype, pos_pairs in seeds.items():
                (
                    minibatch.seeds[etype],
                    minibatch.labels[etype],
                    minibatch.indexes[etype],
                ) = self._sample_with_etype(pos_pairs, etype)
        else:
            (
                minibatch.seeds,
                minibatch.labels,
                minibatch.indexes,
            ) = self._sample_with_etype(seeds)
        return minibatch

    def _sample_with_etype(self, seeds, etype=None):
        """Generate negative pairs for a given etype form positive pairs
        for a given etype. If `seeds` is a 2D tensor, which represents
        `seeds` is used in minibatch, corresponding labels and indexes will be
        constructed.

        Parameters
        ----------
        seeds : Tensor, Tensor
            A N*2 tensors that represent source-destination node pairs of
            positive edges, where positive means the edge must exist in the
            graph.
        etype : str
            Canonical edge type.

        Returns
        -------
        Tensor
            A collection of postive and negative node pairs.
        Tensor
            Corresponding labels. If label is True, corresponding edge is
            positive. If label is False, corresponding edge is negative.
        Tensor
            Corresponding indexes, indicates to which query an edge belongs.

        """
        raise NotImplementedError
