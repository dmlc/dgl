"""Graph Bolt CUDA-related Data Pipelines"""

from torchdata.datapipes.iter import IterDataPipe

from ..utils import recursive_apply


def _to(x, device):
    return x.to(device) if hasattr(x, "to") else x


class CopyTo(IterDataPipe):
    """DataPipe that transfers each element yielded from the previous DataPipe
    to the given device.

    This is equivalent to

    .. code:: python

       for data in datapipe:
           yield data.to(device)

    Parameters
    ----------
    datapipe : DataPipe
        The DataPipe.
    device : torch.device
        The PyTorch CUDA device.
    """

    def __init__(self, datapipe, device):
        super().__init__()
        self.datapipe = datapipe
        self.device = device

    def __iter__(self):
        for data in self.datapipe:
            data = recursive_apply(data, _to, self.device)
            yield data
