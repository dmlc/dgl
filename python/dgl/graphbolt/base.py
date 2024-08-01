"""Base types and utilities for Graph Bolt."""

from collections import deque
from dataclasses import dataclass

import torch
from torch.torch_version import TorchVersion

if (
    TorchVersion(torch.__version__) >= "2.3.0"
    and TorchVersion(torch.__version__) < "2.3.1"
):
    # Due to https://github.com/dmlc/dgl/issues/7380, for torch 2.3.0, we need
    # to check if dill is available before using it.
    torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = (
        torch.utils._import_utils.dill_available()
    )

# pylint: disable=wrong-import-position
from torch.utils.data import functional_datapipe, IterDataPipe

from .internal_utils import (
    get_nonproperty_attributes,
    recursive_apply,
    recursive_apply_reduce_all,
)

__all__ = [
    "CANONICAL_ETYPE_DELIMITER",
    "ORIGINAL_EDGE_ID",
    "etype_str_to_tuple",
    "etype_tuple_to_str",
    "CopyTo",
    "Waiter",
    "Bufferer",
    "EndMarker",
    "isin",
    "index_select",
    "expand_indptr",
    "indptr_edge_ids",
    "CSCFormatBase",
    "seed",
    "seed_type_str_to_ntypes",
    "get_host_to_device_uva_stream",
    "get_device_to_host_uva_stream",
]

CANONICAL_ETYPE_DELIMITER = ":"
ORIGINAL_EDGE_ID = "_ORIGINAL_EDGE_ID"


# There needs to be a single instance of the uva_stream, if it is created
# multiple times, it leads to multiple CUDA memory pools and memory leaks.
def get_host_to_device_uva_stream():
    """The host to device copy stream to be used for pipeline parallelism."""
    if not hasattr(get_host_to_device_uva_stream, "stream"):
        get_host_to_device_uva_stream.stream = torch.cuda.Stream(priority=-1)
    return get_host_to_device_uva_stream.stream


def get_device_to_host_uva_stream():
    """The device to host copy stream to be used for pipeline parallelism."""
    if not hasattr(get_device_to_host_uva_stream, "stream"):
        get_device_to_host_uva_stream.stream = torch.cuda.Stream(priority=-1)
    return get_device_to_host_uva_stream.stream


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


if TorchVersion(torch.__version__) >= TorchVersion("2.2.0a0"):

    torch_fake_decorator = (
        torch.library.impl_abstract
        if TorchVersion(torch.__version__) < TorchVersion("2.4.0a0")
        else torch.library.register_fake
    )

    @torch_fake_decorator("graphbolt::expand_indptr")
    def expand_indptr_fake(indptr, dtype, node_ids, output_size):
        """Fake implementation of expand_indptr for torch.compile() support."""
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


if TorchVersion(torch.__version__) >= TorchVersion("2.2.0a0"):

    torch_fake_decorator = (
        torch.library.impl_abstract
        if TorchVersion(torch.__version__) < TorchVersion("2.4.0a0")
        else torch.library.register_fake
    )

    @torch_fake_decorator("graphbolt::indptr_edge_ids")
    def indptr_edge_ids_fake(indptr, dtype, offset, output_size):
        """Fake implementation of indptr_edge_ids for torch.compile() support."""
        if output_size is None:
            output_size = torch.library.get_ctx().new_dynamic_size()
        if dtype is None:
            dtype = offset.dtype
        return indptr.new_empty(output_size, dtype=dtype)


def indptr_edge_ids(indptr, dtype=None, offset=None, output_size=None):
    """Converts a given indptr offset tensor to a COO format tensor for the edge
    ids. For a given indptr [0, 2, 5, 7] and offset tensor [0, 100, 200], the
    output will be [0, 1, 100, 101, 102, 201, 202]. If offset was not provided,
    the output would be [0, 1, 0, 1, 2, 0, 1].

    Parameters
    ----------
    indptr : torch.Tensor
        A 1D tensor represents the csc_indptr tensor.
    dtype : Optional[torch.dtype]
        The dtype of the returned output tensor.
    offset : Optional[torch.Tensor]
        A 1D tensor represents the offsets that the returned tensor will be
        populated with.
    output_size : Optional[int]
        The size of the output tensor. Should be equal to indptr[-1]. Using this
        argument avoids a stream synchronization to calculate the output shape.

    Returns
    -------
    torch.Tensor
        The converted COO edge ids tensor.
    """
    assert indptr.dim() == 1, "Indptr should be 1D tensor."
    assert offset is None or offset.dim() == 1, "Offset should be 1D tensor."
    if dtype is None:
        dtype = offset.dtype
    return torch.ops.graphbolt.indptr_edge_ids(
        indptr, dtype, offset, output_size
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
        The indexed input tensor, equivalent to tensor[index]. If index is in
        pinned memory, then the result is placed into pinned memory as well.
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


def seed_type_str_to_ntypes(seed_type, seed_size):
    """Convert seeds type to node types from string to list.

    Examples
    --------
    1. node pairs

    >>> seed_type = "user:like:item"
    >>> seed_size = 2
    >>> node_type = seed_type_str_to_ntypes(seed_type, seed_size)
    >>> print(node_type)
    ["user", "item"]

    2. hyperlink

    >>> seed_type = "query:user:item"
    >>> seed_size = 3
    >>> node_type = seed_type_str_to_ntypes(seed_type, seed_size)
    >>> print(node_type)
    ["query", "user", "item"]
    """
    assert isinstance(
        seed_type, str
    ), f"Passed-in seed type should be string, but got {type(seed_type)}"
    ntypes = seed_type.split(CANONICAL_ETYPE_DELIMITER)
    is_hyperlink = len(ntypes) == seed_size
    if not is_hyperlink:
        ntypes = ntypes[::2]
    return ntypes


def apply_to(x, device, non_blocking=False):
    """Apply `to` function to object x only if it has `to`."""

    if device == "pinned" and hasattr(x, "pin_memory"):
        return x.pin_memory()
    if not hasattr(x, "to"):
        return x
    if not non_blocking:
        return x.to(device)
    return x.to(device, non_blocking=True)


def is_object_pinned(obj):
    """Recursively check all members of the object and return True if only if
    all are pinned."""

    for attr in get_nonproperty_attributes(obj):
        member_result = recursive_apply_reduce_all(
            getattr(obj, attr),
            lambda x: x is None or x.is_pinned(),
        )
        if not member_result:
            return False
    return True


@functional_datapipe("copy_to")
class CopyTo(IterDataPipe):
    """DataPipe that transfers each element yielded from the previous DataPipe
    to the given device. For MiniBatch, only the related attributes
    (automatically inferred) will be transferred by default.

    Functional name: :obj:`copy_to`.

    When ``data`` has ``to`` method implemented, ``CopyTo`` will be equivalent
    to

    .. code:: python

       for data in datapipe:
           yield data.to(device)

    Parameters
    ----------
    datapipe : DataPipe
        The DataPipe.
    device : torch.device
        The PyTorch CUDA device.
    non_blocking : bool
        Whether the copy should be performed without blocking. All elements have
        to be already in pinned system memory if enabled. Default is False.
    """

    def __init__(self, datapipe, device, non_blocking=False):
        super().__init__()
        self.datapipe = datapipe
        self.device = torch.device(device)
        self.non_blocking = non_blocking

    def __iter__(self):
        for data in self.datapipe:
            yield recursive_apply(
                data, apply_to, self.device, self.non_blocking
            )


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

    def __getstate__(self):
        state = (self.datapipe, self.buffer.maxlen)
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        self.datapipe, buffer_size = state
        self.buffer = deque(maxlen=buffer_size)

    def reset(self):
        """Resets the state of the datapipe."""
        self.buffer.clear()


@functional_datapipe("wait")
class Waiter(IterDataPipe):
    """Calls the wait function of all items."""

    def __init__(self, datapipe):
        self.datapipe = datapipe

    def __iter__(self):
        for data in self.datapipe:
            data.wait()
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

    def __init__(self, indptr: torch.Tensor, indices: torch.Tensor):
        self.indptr = indptr
        self.indices = indices
        if not indptr.is_cuda:
            assert self.indptr[-1] == len(
                self.indices
            ), "The last element of indptr should be the same as the length of indices."

    def __repr__(self) -> str:
        return _csc_format_base_str(self)

    def to(  # pylint: disable=invalid-name
        self, device: torch.device, non_blocking=False
    ) -> None:
        """Copy `CSCFormatBase` to the specified device using reflection."""

        for attr in dir(self):
            # Only copy member variables.
            if not callable(getattr(self, attr)) and not attr.startswith("__"):
                setattr(
                    self,
                    attr,
                    recursive_apply(
                        getattr(self, attr),
                        apply_to,
                        device,
                        non_blocking=non_blocking,
                    ),
                )

        return self

    def pin_memory(self):
        """Copy `SampledSubgraph` to the pinned memory using reflection."""

        return self.to("pinned")

    def is_pinned(self) -> bool:
        """Check whether `SampledSubgraph` is pinned using reflection."""

        return is_object_pinned(self)


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
