"""Base types and utilities for Graph Bolt."""

from torch.utils.data import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe, Mapper

from ..utils import recursive_apply

from .minibatch import MiniBatch

__all__ = [
    "CANONICAL_ETYPE_DELIMITER",
    "etype_str_to_tuple",
    "etype_tuple_to_str",
    "CopyTo",
]

CANONICAL_ETYPE_DELIMITER = ":"


def etype_tuple_to_str(c_etype):
    """Convert canonical etype from tuple to string.

    Examples
    --------
    >>> c_etype = ("user", "like", "item")
    >>> c_etype_str = _etype_tuple_to_str(c_etype)
    >>> print(c_etype_str)
    "user:like:item"
    """
    assert isinstance(c_etype, tuple) and len(c_etype) == 3, (
        "Passed-in canonical etype should be in format of (str, str, str). "
        f"But got {c_etype}."
    )
    return CANONICAL_ETYPE_DELIMITER.join(c_etype)


def etype_str_to_tuple(c_etype):
    """Convert canonical etype from tuple to string.

    Examples
    --------
    >>> c_etype_str = "user:like:item"
    >>> c_etype = _etype_str_to_tuple(c_etype_str)
    >>> print(c_etype)
    ("user", "like", "item")
    """
    ret = tuple(c_etype.split(CANONICAL_ETYPE_DELIMITER))
    assert len(ret) == 3, (
        "Passed-in canonical etype should be in format of 'str:str:str'. "
        f"But got {c_etype}."
    )
    return ret


def _to(x, device):
    return x.to(device) if hasattr(x, "to") else x


@functional_datapipe("copy_to")
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
        fn:
            The function applied to each minibatch which is responsible for
            converting subgraph structures, potentially utilizing other fields
            within the minibatch as arguments.
        """
        super().__init__(datapipe, self._transformer)
        self.transformer = transformer

    def _transformer(self, minibatch):
        minibatch = transformer(minibatch)
        assert isinstance(
            minibatch, MiniBatch
        ), "The transformer output should be a instance of MiniBatch"
        return minibatch
