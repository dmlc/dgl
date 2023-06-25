"""Graph Bolt CUDA-related Data Pipelines"""

import torch
from torchdata.datapipes.iter import IterDataPipe

from ..utils import recursive_apply


def _to(x, device):
    return x.to(device) if hasattr(x, "to") else x


def _record_stream(x, stream):
    if stream is None:
        return x
    if hasattr(x, "record_stream"):
        x.record_stream(stream)
    return x


class CopyToDevice(IterDataPipe):
    """DataPipe that transfers each element yielded from the previous DataPipe
    to the given CUDA device.

    Yields the same element from the previous DataPipe, with tensors and graphs
    transferred to the given CUDA device.

    If :attr:`stream` argument is given, yields a pair with the first element
    the same element but with data transferred, and the second element the
    recorded stream event.  In this case, normally :class:`WaitStreamEvent`
    should follow, which waits for the recorded stream event and yields the data
    afterwards.

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
        self.stream = stream

    def __iter__(self):
        if self.stream is not None:
            current_stream = torch.cuda.current_stream()
            current_stream.wait_stream(self.stream)
        else:
            current_stream = None

        for data in self.dp:
            with torch.cuda.stream(self.stream):
                data = recursive_apply(data, _to, self.device)
                data = recursive_apply(data, _record_stream, current_stream)

            if self.stream is not None:
                yield data, self.stream.record_event()
            else:
                yield data


class WaitStreamEvent(IterDataPipe):
    """Waits for the stream event to finish and yields the transferred item
    from the :class:`CopyToDevice` DataPipe.

    See also
    --------
    CopyToDevice
    """

    def __init__(self, dp):
        super().__init__()
        self.dp = dp

    def __iter__(self):
        for data, stream_event in self.dp:
            stream_event.wait()
            yield data
