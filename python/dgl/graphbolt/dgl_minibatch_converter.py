"""Computing pack converter"""

from torch.utils.data import functional_datapipe

from torchdata.datapipes.iter import Mapper

from .minibatch import MiniBatch


@functional_datapipe("to_dgl_minibatch")
class DGLMiniBatchConverter(Mapper):
    """Convert a graphbolt mini-batch to a dgl mini-batch."""

    def __init__(
        self,
        datapipe,
    ):
        """
        Initlization for a subgraph transformer.
        Parameters
        ----------
        datapipe : DataPipe
            The datapipe.
        """
        super().__init__(datapipe, MiniBatch.to_dgl)
