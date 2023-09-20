"""Computing pack converter"""

from enum import Enum

from torch.utils.data import functional_datapipe

from torchdata.datapipes.iter import Mapper

from .minibatch import MiniBatch

# Define an enumeration for days of the week
class ComputingLib(Enum):
    """Enum denotes the supported computing libs."""

    DGL = "dgl"
    # TODO: Support below libs.
    # DGL_SPARSE = "dgl_sparse"
    # PYG = "pyg"


@functional_datapipe("to_computing_pack")
class ComputingPackConverter(Mapper):
    """A converter used to convert a mini-batch to a computing pack."""

    def __init__(
        self,
        datapipe,
        converter,
    ):
        """
        Initlization for a subgraph transformer.
        Parameters
        ----------
        datapipe : DataPipe
            The datapipe.
        converter : str or UDF
            - If a string is provided, utilize a built-in conversion function.
            - If a function is provided, it should accept a MiniBatch as its
            input and be responsible for the conversion process.
        """
        super().__init__(datapipe, self._transformer)
        if isinstance(converter, str):
            if converter == ComputingLib.DGL:
                self.converter = MiniBatch.to_dgl_blocks
            else:
                raise TypeError(f"Unsupported computing lib: {converter}")
        else:
            self.converter = converter
        super().__init__(datapipe, self.converter)
