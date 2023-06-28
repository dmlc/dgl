"""Graph Bolt CUDA-related Data Pipelines"""

import torch
from torchdata.datapipes.iter import IterDataPipe

from ..utils import recursive_apply


def _to(x, device):
    return x.to(device) if hasattr(x, "to") else x


class CopyTo(IterDataPipe):
    """DataPipe that transfers each element yielded from the previous DataPipe
    to the given device.

    This is equivalent to

    .. code:: python

       for data in dp:
           yield data.to(device)

    Parameters
    ----------
    dp : DataPipe
        The DataPipe.
    device : torch.device
        The PyTorch CUDA device.
    stream : torch.cuda.Stream
        The CUDA stream to perform transfer on.
    """

    def __init__(self, dp, device, stream=None):
        super().__init__()
        self.dp = dp
        self.device = device

    def __iter__(self):
        for data in self.dp:
            data = recursive_apply(data, _to, self.device)
            yield data
