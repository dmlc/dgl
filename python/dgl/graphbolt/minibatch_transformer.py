"""Mini-batch transformer"""

from torch.utils.data import functional_datapipe

from torchdata.datapipes.iter import Mapper

from .minibatch import MiniBatch


@functional_datapipe("transform")
class MiniBatchTransformer(Mapper):
    """A mini-batch transformer used to manipulate mini-batch"""

    def __init__(
        self,
        datapipe,
        transformer,
    ):
        """
        Initlization for a subgraph transformer.
        Parameters
        ----------
        datapipe : DataPipe
            The datapipe.
        transformer:
            The function applied to each minibatch which is responsible for
            transforming the minibatch.
        """
        super().__init__(datapipe, self._transformer)
        self.transformer = transformer

    def _transformer(self, minibatch):
        minibatch = self.transformer(minibatch)
        assert isinstance(
            minibatch, MiniBatch
        ), "The transformer output should be an instance of MiniBatch"
        return minibatch
