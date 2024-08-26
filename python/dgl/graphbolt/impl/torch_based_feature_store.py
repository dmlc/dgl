"""Torch-based feature store for GraphBolt."""

import copy
import textwrap
from typing import Dict, List

import numpy as np
import torch

from ..base import (
    get_device_to_host_uva_stream,
    get_host_to_device_uva_stream,
    index_select,
)
from ..feature_store import Feature
from ..internal_utils import gb_warning, is_wsl
from .basic_feature_store import BasicFeatureStore
from .ondisk_metadata import OnDiskFeatureData

__all__ = ["TorchBasedFeature", "DiskBasedFeature", "TorchBasedFeatureStore"]


class _Waiter:
    def __init__(self, event, values):
        self.event = event
        self.values = values

    def wait(self):
        """Returns the stored value when invoked."""
        self.event.wait()
        values = self.values
        # Ensure there is no memory leak.
        self.event = self.values = None
        return values


class TorchBasedFeature(Feature):
    r"""A wrapper of pytorch based feature.

    Initialize a torch based feature store by a torch feature.
    Note that the feature can be either in memory or on disk.

    Parameters
    ----------
    torch_feature : torch.Tensor
        The torch feature.
        Note that the dimension of the tensor should be greater than 1.

    Examples
    --------
    >>> import torch
    >>> from dgl import graphbolt as gb

    1. The feature is in memory.

    >>> torch_feat = torch.arange(10).reshape(2, -1)
    >>> feature = gb.TorchBasedFeature(torch_feat)
    >>> feature.read()
    tensor([[0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9]])
    >>> feature.read(torch.tensor([0]))
    tensor([[0, 1, 2, 3, 4]])
    >>> feature.update(torch.tensor([[1 for _ in range(5)]]),
    ...                      torch.tensor([1]))
    >>> feature.read(torch.tensor([0, 1]))
    tensor([[0, 1, 2, 3, 4],
            [1, 1, 1, 1, 1]])
    >>> feature.size()
    torch.Size([5])

    2. The feature is on disk. Note that you can use gb.numpy_save_aligned as a
    replacement for np.save to potentially get increased performance.

    >>> import numpy as np
    >>> arr = np.array([[1, 2], [3, 4]])
    >>> np.save("/tmp/arr.npy", arr)
    >>> torch_feat = torch.from_numpy(np.load("/tmp/arr.npy", mmap_mode="r+"))
    >>> feature = gb.TorchBasedFeature(torch_feat)
    >>> feature.read()
    tensor([[1, 2],
            [3, 4]])
    >>> feature.read(torch.tensor([0]))
    tensor([[1, 2]])

    3. Pinned CPU feature.

    >>> torch_feat = torch.arange(10).reshape(2, -1).pin_memory()
    >>> feature = gb.TorchBasedFeature(torch_feat)
    >>> feature.read().device
    device(type='cuda', index=0)
    >>> feature.read(torch.tensor([0]).cuda()).device
    device(type='cuda', index=0)
    """

    def __init__(self, torch_feature: torch.Tensor, metadata: Dict = None):
        super().__init__()
        self._is_inplace_pinned = set()
        assert isinstance(torch_feature, torch.Tensor), (
            f"torch_feature in TorchBasedFeature must be torch.Tensor, "
            f"but got {type(torch_feature)}."
        )
        assert torch_feature.dim() > 1, (
            f"dimension of torch_feature in TorchBasedFeature must be greater "
            f"than 1, but got {torch_feature.dim()} dimension."
        )
        # Make sure the tensor is contiguous.
        self._tensor = torch_feature.contiguous()
        self._metadata = metadata

    def __del__(self):
        # torch.Tensor.pin_memory() is not an inplace operation. To make it
        # truly in-place, we need to use cudaHostRegister. Then, we need to use
        # cudaHostUnregister to unpin the tensor in the destructor.
        # https://github.com/pytorch/pytorch/issues/32167#issuecomment-753551842
        for tensor in self._is_inplace_pinned:
            assert self._inplace_unpinner(tensor.data_ptr()) == 0

    def read(self, ids: torch.Tensor = None):
        """Read the feature by index.

        If the feature is on pinned CPU memory and `ids` is on GPU or pinned CPU
        memory, it will be read by GPU and the returned tensor will be on GPU.
        Otherwise, the returned tensor will be on CPU.

        Parameters
        ----------
        ids : torch.Tensor, optional
            The index of the feature. If specified, only the specified indices
            of the feature are read. If None, the entire feature is returned.

        Returns
        -------
        torch.Tensor
            The read feature.
        """
        if ids is None:
            if self._tensor.is_pinned():
                return self._tensor.cuda()
            return self._tensor
        return index_select(self._tensor, ids)

    def read_async(self, ids: torch.Tensor):
        r"""Read the feature by index asynchronously.

        Parameters
        ----------
        ids : torch.Tensor
            The index of the feature. Only the specified indices of the
            feature are read.
        Returns
        -------
        A generator object.
            The returned generator object returns a future on
            ``read_async_num_stages(ids.device)``\ th invocation. The return result
            can be accessed by calling ``.wait()``. on the returned future object.
            It is undefined behavior to call ``.wait()`` more than once.

        Examples
        --------
        >>> import dgl.graphbolt as gb
        >>> feature = gb.Feature(...)
        >>> ids = torch.tensor([0, 2])
        >>> for stage, future in enumerate(feature.read_async(ids)):
        ...     pass
        >>> assert stage + 1 == feature.read_async_num_stages(ids.device)
        >>> result = future.wait()  # result contains the read values.
        """
        assert self._tensor.device.type == "cpu"
        if ids.is_cuda and self.is_pinned():
            current_stream = torch.cuda.current_stream()
            host_to_device_stream = get_host_to_device_uva_stream()
            host_to_device_stream.wait_stream(current_stream)
            with torch.cuda.stream(host_to_device_stream):
                ids.record_stream(torch.cuda.current_stream())
                values = index_select(self._tensor, ids)
                values.record_stream(current_stream)
                values_copy_event = torch.cuda.Event()
                values_copy_event.record()

            yield _Waiter(values_copy_event, values)
        elif ids.is_cuda:
            ids_device = ids.device
            current_stream = torch.cuda.current_stream()
            device_to_host_stream = get_device_to_host_uva_stream()
            device_to_host_stream.wait_stream(current_stream)
            with torch.cuda.stream(device_to_host_stream):
                ids.record_stream(torch.cuda.current_stream())
                ids = ids.to(self._tensor.device, non_blocking=True)
                ids_copy_event = torch.cuda.Event()
                ids_copy_event.record()

            yield  # first stage is done.

            ids_copy_event.synchronize()
            values = torch.ops.graphbolt.index_select_async(self._tensor, ids)
            yield

            host_to_device_stream = get_host_to_device_uva_stream()
            with torch.cuda.stream(host_to_device_stream):
                values_cuda = values.wait().to(ids_device, non_blocking=True)
                values_cuda.record_stream(current_stream)
                values_copy_event = torch.cuda.Event()
                values_copy_event.record()

            yield _Waiter(values_copy_event, values_cuda)
        else:
            yield torch.ops.graphbolt.index_select_async(self._tensor, ids)

    def read_async_num_stages(self, ids_device: torch.device):
        """The number of stages of the read_async operation. See read_async
        function for directions on its use. This function is required to return
        the number of yield operations when read_async is used with a tensor
        residing on ids_device.

        Parameters
        ----------
        ids_device : torch.device
            The device of the ids parameter passed into read_async.
        Returns
        -------
        int
            The number of stages of the read_async operation.
        """
        if ids_device.type == "cuda":
            if self._tensor.is_cuda:
                # If the ids and the tensor are on cuda, no need for async.
                return 0
            return 1 if self.is_pinned() else 3
        else:
            return 1

    def size(self):
        """Get the size of the feature.

        Returns
        -------
        torch.Size
            The size of the feature.
        """
        return self._tensor.size()[1:]

    def count(self):
        """Get the count of the feature.

        Returns
        -------
        int
            The count of the feature.
        """
        return self._tensor.size()[0]

    def update(self, value: torch.Tensor, ids: torch.Tensor = None):
        """Update the feature store.

        Parameters
        ----------
        value : torch.Tensor
            The updated value of the feature.
        ids : torch.Tensor, optional
            The indices of the feature to update. If specified, only the
            specified indices of the feature will be updated. For the feature,
            the `ids[i]` row is updated to `value[i]`. So the indices and value
            must have the same length. If None, the entire feature will be
            updated.
        """
        if ids is None:
            self._tensor = value
        else:
            assert ids.shape[0] == value.shape[0], (
                f"ids and value must have the same length, "
                f"but got {ids.shape[0]} and {value.shape[0]}."
            )
            assert self.size() == value.size()[1:], (
                f"The size of the feature is {self.size()}, "
                f"while the size of the value is {value.size()[1:]}."
            )
            if self._tensor.is_pinned() and value.is_cuda and ids.is_cuda:
                raise NotImplementedError(
                    "Update the feature on pinned CPU memory by GPU is not "
                    "supported yet."
                )
            self._tensor[ids] = value

    def metadata(self):
        """Get the metadata of the feature.

        Returns
        -------
        Dict
            The metadata of the feature.
        """
        return (
            self._metadata if self._metadata is not None else super().metadata()
        )

    def pin_memory_(self):
        """In-place operation to copy the feature to pinned memory. Returns the
        same object modified in-place."""
        # torch.Tensor.pin_memory() is not an inplace operation. To make it
        # truly in-place, we need to use cudaHostRegister. Then, we need to use
        # cudaHostUnregister to unpin the tensor in the destructor.
        # https://github.com/pytorch/pytorch/issues/32167#issuecomment-753551842
        x = self._tensor
        if not x.is_pinned() and x.device.type == "cpu":
            assert (
                x.is_contiguous()
            ), "Tensor pinning is only supported for contiguous tensors."
            cudart = torch.cuda.cudart()
            assert (
                cudart.cudaHostRegister(
                    x.data_ptr(), x.numel() * x.element_size(), 0
                )
                == 0
            )

            self._is_inplace_pinned.add(x)
            self._inplace_unpinner = cudart.cudaHostUnregister

        return self

    def is_pinned(self):
        """Returns True if the stored feature is pinned."""
        return self._tensor.is_pinned()

    def to(self, device):  # pylint: disable=invalid-name
        """Copy `TorchBasedFeature` to the specified device."""
        # copy.copy is a shallow copy so it does not copy tensor memory.
        self2 = copy.copy(self)
        if device == "pinned":
            self2._tensor = self2._tensor.pin_memory()
        else:
            self2._tensor = self2._tensor.to(device)
        return self2

    def __repr__(self) -> str:
        ret = (
            "{Classname}(\n"
            "    feature={feature},\n"
            "    metadata={metadata},\n"
            ")"
        )

        feature_str = textwrap.indent(
            str(self._tensor), " " * len("    feature=")
        ).strip()
        metadata_str = textwrap.indent(
            str(self.metadata()), " " * len("    metadata=")
        ).strip()

        return ret.format(
            Classname=self.__class__.__name__,
            feature=feature_str,
            metadata=metadata_str,
        )


class DiskBasedFeature(Feature):
    r"""A wrapper of disk based feature.

    Initialize a disk based feature fetcher by a numpy file. Note that you can
    use gb.numpy_save_aligned as a replacement for np.save to potentially get
    increased performance.

    Parameters
    ----------
    path : string
        The path to the numpy feature file.
        Note that the dimension of the numpy should be greater than 1.
    metadata : Dict
        The metadata of the feature.
    num_threads : int
        The number of threads driving io_uring queues.
    Examples
    --------
    >>> import torch
    >>> from dgl import graphbolt as gb
    >>> torch_feat = torch.arange(10).reshape(2, -1)
    >>> pth = "path/to/feat.npy"
    >>> np.save(pth, torch_feat)
    >>> feature = gb.DiskBasedFeature(pth)
    >>> feature.read(torch.tensor([0]))
    tensor([[0, 1, 2, 3, 4]])
    >>> feature.size()
    torch.Size([5])
    """

    def __init__(self, path: str, metadata: Dict = None, num_threads=None):
        super().__init__()
        mmap_mode = "r+"
        ondisk_data = np.load(path, mmap_mode=mmap_mode)
        assert ondisk_data.flags[
            "C_CONTIGUOUS"
        ], "DiskBasedFeature only supports C_CONTIGUOUS array."
        self._tensor = torch.from_numpy(ondisk_data)

        self._metadata = metadata
        if torch.ops.graphbolt.detect_io_uring():
            self._ondisk_npy_array = torch.ops.graphbolt.ondisk_npy_array(
                path, self._tensor.dtype, self._tensor.shape, num_threads
            )

    def read(self, ids: torch.Tensor = None):
        """Read the feature by index.
        The returned tensor will be on CPU.
        Parameters
        ----------
        ids : torch.Tensor
            The index of the feature. Only the specified indices of the
            feature are read.
        Returns
        -------
        torch.Tensor
            The read feature.
        """
        if ids is None:
            return self._tensor
        elif torch.ops.graphbolt.detect_io_uring():
            try:
                return self._ondisk_npy_array.index_select(ids).wait()
            except RuntimeError:
                raise IndexError
        else:
            return index_select(self._tensor, ids)

    def read_async(self, ids: torch.Tensor):
        r"""Read the feature by index asynchronously.

        Parameters
        ----------
        ids : torch.Tensor
            The index of the feature. Only the specified indices of the
            feature are read.
        Returns
        -------
        A generator object.
            The returned generator object returns a future on
            ``read_async_num_stages(ids.device)``\ th invocation. The return result
            can be accessed by calling ``.wait()``. on the returned future object.
            It is undefined behavior to call ``.wait()`` more than once.

        Examples
        --------
        >>> import dgl.graphbolt as gb
        >>> feature = gb.Feature(...)
        >>> ids = torch.tensor([0, 2])
        >>> for stage, future in enumerate(feature.read_async(ids)):
        ...     pass
        >>> assert stage + 1 == feature.read_async_num_stages(ids.device)
        >>> result = future.wait()  # result contains the read values.
        """
        assert torch.ops.graphbolt.detect_io_uring()
        if ids.is_cuda:
            ids_device = ids.device
            current_stream = torch.cuda.current_stream()
            device_to_host_stream = get_device_to_host_uva_stream()
            device_to_host_stream.wait_stream(current_stream)
            with torch.cuda.stream(device_to_host_stream):
                ids.record_stream(torch.cuda.current_stream())
                ids = ids.to(self._tensor.device, non_blocking=True)
                ids_copy_event = torch.cuda.Event()
                ids_copy_event.record()

            yield  # first stage is done.

            ids_copy_event.synchronize()
            values = self._ondisk_npy_array.index_select(ids)
            yield

            host_to_device_stream = get_host_to_device_uva_stream()
            with torch.cuda.stream(host_to_device_stream):
                values_cuda = values.wait().to(ids_device, non_blocking=True)
                values_cuda.record_stream(current_stream)
                values_copy_event = torch.cuda.Event()
                values_copy_event.record()

            yield _Waiter(values_copy_event, values_cuda)
        else:
            yield self._ondisk_npy_array.index_select(ids)

    def read_async_num_stages(self, ids_device: torch.device):
        """The number of stages of the read_async operation. See read_async
        function for directions on its use. This function is required to return
        the number of yield operations when read_async is used with a tensor
        residing on ids_device.

        Parameters
        ----------
        ids_device : torch.device
            The device of the ids parameter passed into read_async.
        Returns
        -------
        int
            The number of stages of the read_async operation.
        """
        return 3 if ids_device.type == "cuda" else 1

    def size(self):
        """Get the size of the feature.
        Returns
        -------
        torch.Size
            The size of the feature.
        """
        return self._tensor.size()[1:]

    def count(self):
        """Get the count of the feature.

        Returns
        -------
        int
            The count of the feature.
        """
        return self._tensor.size()[0]

    def update(self, value: torch.Tensor, ids: torch.Tensor = None):
        """Disk based feature does not support update for now."""
        raise NotImplementedError

    def metadata(self):
        """Get the metadata of the feature.
        Returns
        -------
        Dict
            The metadata of the feature.
        """
        return (
            self._metadata if self._metadata is not None else super().metadata()
        )

    def read_into_memory(self) -> TorchBasedFeature:
        """Change disk-based feature to torch-based feature."""
        return TorchBasedFeature(self._tensor, self._metadata)

    def to(self, _):  # pylint: disable=invalid-name
        """Placeholder `DiskBasedFeature` to implementation. It is a no-op."""
        gb_warning(
            "`DiskBasedFeature.to(device)` is not supported. Leaving unmodified."
        )
        return self

    def pin_memory_(self):  # pylint: disable=invalid-name
        r"""Placeholder `DiskBasedFeature` pin_memory_ implementation. It is a no-op."""
        gb_warning(
            "`DiskBasedFeature.pin_memory_()` is not supported. Leaving unmodified."
        )
        return self

    def __repr__(self) -> str:
        ret = (
            "{Classname}(\n"
            "    feature={feature},\n"
            "    metadata={metadata},\n"
            ")"
        )

        feature_str = textwrap.indent(
            str(self._tensor), " " * len("    feature=")
        ).strip()
        metadata_str = textwrap.indent(
            str(self.metadata()), " " * len("    metadata=")
        ).strip()

        return ret.format(
            Classname=self.__class__.__name__,
            feature=feature_str,
            metadata=metadata_str,
        )


class TorchBasedFeatureStore(BasicFeatureStore):
    r"""A store to manage multiple pytorch based feature for access.

    The feature stores are described by the `feat_data`. The `feat_data` is a
    list of `OnDiskFeatureData`.

    For a feature store, its format must be either "pt" or "npy" for Pytorch or
    Numpy formats. If the format is "pt", the feature store must be loaded in
    memory. If the format is "npy", the feature store can be loaded in memory or
    on disk. Note that you can use gb.numpy_save_aligned as a replacement for
    np.save to potentially get increased performance.

    Parameters
    ----------
    feat_data : List[OnDiskFeatureData]
        The description of the feature stores.

    Examples
    --------
    >>> import torch
    >>> import numpy as np
    >>> from dgl import graphbolt as gb
    >>> edge_label = torch.tensor([[1], [2], [3]])
    >>> node_feat = torch.tensor([[1, 2, 3], [4, 5, 6]])
    >>> torch.save(edge_label, "/tmp/edge_label.pt")
    >>> gb.numpy_save_aligned("/tmp/node_feat.npy", node_feat.numpy())
    >>> feat_data = [
    ...     gb.OnDiskFeatureData(domain="edge", type="author:writes:paper",
    ...         name="label", format="torch", path="/tmp/edge_label.pt",
    ...         in_memory=True),
    ...     gb.OnDiskFeatureData(domain="node", type="paper", name="feat",
    ...         format="numpy", path="/tmp/node_feat.npy", in_memory=False),
    ... ]
    >>> feature_store = gb.TorchBasedFeatureStore(feat_data)
    """

    def __init__(self, feat_data: List[OnDiskFeatureData]):
        features = {}
        for spec in feat_data:
            key = (spec.domain, spec.type, spec.name)
            metadata = spec.extra_fields
            if spec.format == "torch":
                assert spec.in_memory, (
                    f"Pytorch tensor can only be loaded in memory, "
                    f"but the feature {key} is loaded on disk."
                )
                features[key] = TorchBasedFeature(
                    torch.load(spec.path), metadata=metadata
                )
            elif spec.format == "numpy":
                if spec.in_memory:
                    # TorchBasedFeature is always in memory by default.
                    features[key] = TorchBasedFeature(
                        torch.as_tensor(np.load(spec.path)), metadata=metadata
                    )
                else:
                    # DiskBasedFeature is always out of memory by default.
                    features[key] = DiskBasedFeature(
                        spec.path, metadata=metadata
                    )
            else:
                raise ValueError(f"Unknown feature format {spec.format}")
        super().__init__(features)

    def pin_memory_(self):
        """In-place operation to copy the feature store to pinned memory.
        Returns the same object modified in-place."""
        if is_wsl():
            gb_warning(
                "In place pinning is not supported on WSL. "
                "Returning the out of place pinned `TorchBasedFeatureStore`."
            )
            return self.to("pinned")
        for feature in self._features.values():
            feature.pin_memory_()

        return self

    def is_pinned(self):
        """Returns True if all the stored features are pinned."""
        return all(feature.is_pinned() for feature in self._features.values())

    def to(self, device):  # pylint: disable=invalid-name
        """Copy `TorchBasedFeatureStore` to the specified device."""
        # copy.copy is a shallow copy so it does not copy tensor memory.
        self2 = copy.copy(self)
        self2._features = {k: v.to(device) for k, v in self2._features.items()}
        return self2

    def __repr__(self) -> str:
        ret = "{Classname}(\n" + "    {features}\n" + ")"
        features_str = textwrap.indent(str(self._features), "    ").strip()
        return ret.format(
            Classname=self.__class__.__name__, features=features_str
        )
