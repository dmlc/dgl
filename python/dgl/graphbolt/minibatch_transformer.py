"""Mini-batch transformer"""

from torch.utils.data import functional_datapipe

from torch.utils.data.datapipes.iter import Mapper

from .minibatch import MiniBatch

__all__ = [
    "MiniBatchTransformer",
]


@functional_datapipe("transform")
class MiniBatchTransformer(Mapper):
    """A mini-batch transformer used to manipulate mini-batch.

    Functional name: :obj:`transform`.

    Parameters
    ----------
    datapipe : DataPipe
        The datapipe.
    transformer:
        The function applied to each minibatch which is responsible for
        transforming the minibatch.
    """

    def __init__(
        self,
        datapipe,
        transformer=None,
    ):
        super().__init__(datapipe, self._transformer)
        self.transformer = transformer or self._identity

    def _transformer(self, minibatch):
        minibatch = self.transformer(minibatch)
        assert isinstance(
            minibatch, (MiniBatch,)
        ), "The transformer output should be an instance of MiniBatch"
        return minibatch

    @staticmethod
    def _identity(minibatch):
        return minibatch
