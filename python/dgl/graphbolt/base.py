"""Base types and utilities for Graph Bolt."""

from dataclasses import dataclass

import torch
from torch.utils.data import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from ..utils import recursive_apply

__all__ = [
    "CANONICAL_ETYPE_DELIMITER",
    "ORIGINAL_EDGE_ID",
    "etype_str_to_tuple",
    "etype_tuple_to_str",
    "CopyTo",
    "isin",
    "CSCFormatBase",
]

CANONICAL_ETYPE_DELIMITER = ":"
ORIGINAL_EDGE_ID = "_ORIGINAL_EDGE_ID"


def isin(elements, test_elements):
    """Tests if each element of elements is in test_elements. Returns a boolean
    tensor of the same shape as elements that is True for elements in
    test_elements and False otherwise.

    Parameters
    ----------
    elements : torch.Tensor
        A 1D tensor represents the input elements.
    test_elements : torch.Tensor
        A 1D tensor represents the values to test against for each input.

    Examples
    --------
    >>> isin(torch.tensor([1, 2, 3, 4]), torch.tensor([2, 3]))
    tensor([[False,  True,  True,  False]])
    """
    assert elements.dim() == 1, "Elements should be 1D tensor."
    assert test_elements.dim() == 1, "Test_elements should be 1D tensor."
    return torch.ops.graphbolt.isin(elements, test_elements)


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


def apply_to(x, device):
    """Apply `to` function to object x only if it has `to`."""

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
            data = recursive_apply(data, apply_to, self.device)
            yield data


@dataclass
class CSCFormatBase:
    r"""Basic class representing data in Compressed Sparse Column (CSC) format.

    Examples
    --------
    >>> indptr = torch.tensor([0, 1, 3])
    >>> indices = torch.tensor([1, 4, 2])
    >>> csc_foramt_base = CSCFormatBase(indptr=indptr, indices=indices)
    >>> print(csc_format_base.indptr)
    ... torch.tensor([0, 1, 3])
    >>> print(csc_foramt_base)
    ... torch.tensor([1, 4, 2])
    """
    indptr: torch.Tensor = None
    indices: torch.Tensor = None
