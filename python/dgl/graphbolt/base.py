"""Base types and utilities for Graph Bolt."""

from collections import deque
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
    "FutureWaiter",
    "Waiter",
    "Bufferer",
    "EndMarker",
    "isin",
    "index_select",
    "expand_indptr",
    "CSCFormatBase",
    "seed",
]

CANONICAL_ETYPE_DELIMITER = ":"
ORIGINAL_EDGE_ID = "_ORIGINAL_EDGE_ID"


def seed(val):
    """Set the random seed of Graphbolt.

    Parameters
    ----------
    val : int
        The seed.
    """
    torch.ops.graphbolt.set_seed(val)


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


@torch.library.impl_abstract("graphbolt::expand_indptr")
def expand_indptr_abstract(indptr, dtype, node_ids, output_size):
    if output_size is None:
        output_size = torch.library.get_ctx().new_dynamic_size()
    if dtype is None:
        dtype = node_ids.dtype
    return indptr.new_empty(output_size, dtype=dtype)


def expand_indptr(indptr, dtype=None, node_ids=None, output_size=None):
    """Converts a given indptr offset tensor to a COO format tensor. If
    node_ids is not given, it is assumed to be equal to
    torch.arange(indptr.size(0) - 1, dtype=dtype, device=indptr.device).

    This is equivalent to

    .. code:: python

       if node_ids is None:
           node_ids = torch.arange(len(indptr) - 1, dtype=dtype, device=indptr.device)
       return node_ids.to(dtype).repeat_interleave(indptr.diff())

    Parameters
    ----------
    indptr : torch.Tensor
        A 1D tensor represents the csc_indptr tensor.
    dtype : Optional[torch.dtype]
        The dtype of the returned output tensor.
    node_ids : Optional[torch.Tensor]
        A 1D tensor represents the column node ids that the returned tensor will
        be populated with.
    output_size : Optional[int]
        The size of the output tensor. Should be equal to indptr[-1]. Using this
        argument avoids a stream synchronization to calculate the output shape.

    Returns
    -------
    torch.Tensor
        The converted COO tensor with values from node_ids.
    """
    assert indptr.dim() == 1, "Indptr should be 1D tensor."
    assert not (
        node_ids is None and dtype is None
    ), "One of node_ids or dtype must be given."
    assert (
        node_ids is None or node_ids.dim() == 1
    ), "Node_ids should be 1D tensor."
    if dtype is None:
        dtype = node_ids.dtype
    return torch.ops.graphbolt.expand_indptr(
        indptr, dtype, node_ids, output_size
    )


def index_select(tensor, index):
    """Returns a new tensor which indexes the input tensor along dimension dim
    using the entries in index.

    The returned tensor has the same number of dimensions as the original tensor
    (tensor). The first dimension has the same size as the length of index;
    other dimensions have the same size as in the original tensor.

    When tensor is a pinned tensor and index.is_cuda is True, the operation runs
    on the CUDA device and the returned tensor will also be on CUDA.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor.
    index : torch.Tensor
        The 1-D tensor containing the indices to index.

    Returns
    -------
    torch.Tensor
        The indexed input tensor, equivalent to tensor[index].
    """
    assert index.dim() == 1, "Index should be 1D tensor."
    return torch.ops.graphbolt.index_select(tensor, index)


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
    """Convert canonical etype from string to tuple.

    Examples
    --------
    >>> c_etype_str = "user:like:item"
    >>> c_etype = _etype_str_to_tuple(c_etype_str)
    >>> print(c_etype)
    ("user", "like", "item")
    """
    if isinstance(c_etype, tuple):
        return c_etype
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
    to the given device. For MiniBatch, only the related attributes
    (automatically inferred) will be transferred by default. If you want to
    transfer any other attributes, indicate them in the ``extra_attrs``.

    Functional name: :obj:`copy_to`.

    When ``data`` has ``to`` method implemented, ``CopyTo`` will be equivalent
    to

    .. code:: python

       for data in datapipe:
           yield data.to(device)

    For :class:`~dgl.graphbolt.MiniBatch`, only a part of attributes will be
    transferred to accelerate the process by default:

    - When ``seed_nodes`` is not None and ``node_pairs`` is None, node related
    task is inferred. Only ``labels``, ``sampled_subgraphs``, ``node_features``
    and ``edge_features`` will be transferred.

    - When ``node_pairs`` is not None and ``seed_nodes`` is None, edge/link
    related task is inferred. Only ``labels``, ``compacted_node_pairs``,
    ``compacted_negative_srcs``, ``compacted_negative_dsts``,
    ``sampled_subgraphs``, ``node_features`` and ``edge_features`` will be
    transferred.

    - Otherwise, all attributes will be transferred.

    - If you want some other attributes to be transferred as well, please
    specify the name in the ``extra_attrs``. For instance, the following code
    will copy ``seed_nodes`` to the GPU as well:

    .. code:: python

       datapipe = datapipe.copy_to(device="cuda", extra_attrs=["seed_nodes"])

    Parameters
    ----------
    datapipe : DataPipe
        The DataPipe.
    device : torch.device
        The PyTorch CUDA device.
    extra_attrs: List[string]
        The extra attributes of the data in the DataPipe you want to be carried
        to the specific device. The attributes specified in the ``extra_attrs``
        will be transferred regardless of the task inferred. It could also be
        applied to classes other than :class:`~dgl.graphbolt.MiniBatch`.
    """

    def __init__(self, datapipe, device, extra_attrs=None):
        super().__init__()
        self.datapipe = datapipe
        self.device = device
        self.extra_attrs = extra_attrs

    def __iter__(self):
        for data in self.datapipe:
            data = recursive_apply(data, apply_to, self.device)
            if self.extra_attrs is not None:
                for attr in self.extra_attrs:
                    setattr(
                        data,
                        attr,
                        recursive_apply(
                            getattr(data, attr), apply_to, self.device
                        ),
                    )
            yield data


@functional_datapipe("mark_end")
class EndMarker(IterDataPipe):
    """Used to mark the end of a datapipe and is a no-op."""

    def __init__(self, datapipe):
        self.datapipe = datapipe

    def __iter__(self):
        yield from self.datapipe


@functional_datapipe("buffer")
class Bufferer(IterDataPipe):
    """Buffers items before yielding them.

    Parameters
    ----------
    datapipe : DataPipe
        The data pipeline.
    buffer_size : int, optional
        The size of the buffer which stores the fetched samples. If data coming
        from datapipe has latency spikes, consider setting to a higher value.
        Default is 1.
    """

    def __init__(self, datapipe, buffer_size=1):
        self.datapipe = datapipe
        if buffer_size <= 0:
            raise ValueError(
                "'buffer_size' is required to be a positive integer."
            )
        self.buffer = deque(maxlen=buffer_size)

    def __iter__(self):
        for data in self.datapipe:
            if len(self.buffer) < self.buffer.maxlen:
                self.buffer.append(data)
            else:
                return_data = self.buffer.popleft()
                self.buffer.append(data)
                yield return_data
        while len(self.buffer) > 0:
            yield self.buffer.popleft()


@functional_datapipe("wait")
class Waiter(IterDataPipe):
    """Calls the wait function of all items."""

    def __init__(self, datapipe):
        self.datapipe = datapipe

    def __iter__(self):
        for data in self.datapipe:
            data.wait()
            yield data


@functional_datapipe("wait_future")
class FutureWaiter(IterDataPipe):
    """Calls the result function of all items and returns their results."""

    def __init__(self, datapipe):
        self.datapipe = datapipe

    def __iter__(self):
        for data in self.datapipe:
            yield data.result()


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

    def __init__(self, indptr: torch.Tensor, indices: torch.Tensor):
        self.indptr = indptr
        self.indices = indices
        if not indptr.is_cuda:
            assert self.indptr[-1] == len(
                self.indices
            ), "The last element of indptr should be the same as the length of indices."

    def __repr__(self) -> str:
        return _csc_format_base_str(self)

    def to(self, device: torch.device) -> None:  # pylint: disable=invalid-name
        """Copy `CSCFormatBase` to the specified device using reflection."""

        for attr in dir(self):
            # Only copy member variables.
            if not callable(getattr(self, attr)) and not attr.startswith("__"):
                setattr(
                    self,
                    attr,
                    recursive_apply(
                        getattr(self, attr), lambda x: apply_to(x, device)
                    ),
                )

        return self


def _csc_format_base_str(csc_format_base: CSCFormatBase) -> str:
    final_str = "CSCFormatBase("

    def _add_indent(_str, indent):
        lines = _str.split("\n")
        lines = [lines[0]] + [" " * indent + line for line in lines[1:]]
        return "\n".join(lines)

    final_str += (
        f"indptr={_add_indent(str(csc_format_base.indptr), 21)},\n" + " " * 14
    )
    final_str += (
        f"indices={_add_indent(str(csc_format_base.indices), 22)},\n" + ")"
    )
    return final_str
