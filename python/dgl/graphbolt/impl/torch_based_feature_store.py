"""Torch-based feature store for GraphBolt."""
from typing import Dict, List

import numpy as np
import torch

from ..feature_store import Feature
from .basic_feature_store import BasicFeatureStore
from .ondisk_metadata import OnDiskFeatureData

__all__ = ["TorchBasedFeature", "TorchBasedFeatureStore"]


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

    2. The feature is on disk.

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
        return torch.ops.graphbolt.index_select(self._tensor, ids)

    def size(self):
        """Get the size of the feature.

        Returns
        -------
        torch.Size
            The size of the feature.
        """
        return self._tensor.size()[1:]

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
            assert self.size() == value.size()[1:], (
                f"ids is None, so the entire feature will be updated. "
                f"But the size of the feature is {self.size()}, "
                f"while the size of the value is {value.size()[1:]}."
            )
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
        """In-place operation to copy the feature to pinned memory."""
        self._tensor = self._tensor.pin_memory()

    def __repr__(self) -> str:
        return _torch_based_feature_str(self)


class TorchBasedFeatureStore(BasicFeatureStore):
    r"""A store to manage multiple pytorch based feature for access.

    The feature stores are described by the `feat_data`. The `feat_data` is a
    list of `OnDiskFeatureData`.

    For a feature store, its format must be either "pt" or "npy" for Pytorch or
    Numpy formats. If the format is "pt", the feature store must be loaded in
    memory. If the format is "npy", the feature store can be loaded in memory or
    on disk.

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
    >>> np.save("/tmp/node_feat.npy", node_feat.numpy())
    >>> feat_data = [
    ...     gb.OnDiskFeatureData(domain="edge", type="author:writes:paper",
    ...         name="label", format="torch", path="/tmp/edge_label.pt",
    ...         in_memory=True),
    ...     gb.OnDiskFeatureData(domain="node", type="paper", name="feat",
    ...         format="numpy", path="/tmp/node_feat.npy", in_memory=False),
    ... ]
    >>> feature_sotre = gb.TorchBasedFeatureStore(feat_data)
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
                mmap_mode = "r+" if not spec.in_memory else None
                features[key] = TorchBasedFeature(
                    torch.as_tensor(np.load(spec.path, mmap_mode=mmap_mode)),
                    metadata=metadata,
                )
            else:
                raise ValueError(f"Unknown feature format {spec.format}")
        super().__init__(features)

    def pin_memory_(self):
        """In-place operation to copy the feature store to pinned memory."""
        for feature in self._features.values():
            feature.pin_memory_()

    def __repr__(self) -> str:
        return _torch_based_feature_store_str(self._features)


def _torch_based_feature_str(feature: TorchBasedFeature) -> str:
    final_str = "TorchBasedFeature("
    indent_len = len(final_str)

    def _add_indent(_str, indent):
        lines = _str.split("\n")
        lines = [lines[0]] + [" " * indent + line for line in lines[1:]]
        return "\n".join(lines)

    feature_str = "feature=" + _add_indent(
        str(feature._tensor), indent_len + len("feature=")
    )
    final_str += feature_str + ",\n" + " " * indent_len
    metadata_str = "metadata=" + _add_indent(
        str(feature.metadata()), indent_len + len("metadata=")
    )
    final_str += metadata_str + ",\n)"
    return final_str


def _torch_based_feature_store_str(
    features: Dict[str, TorchBasedFeature]
) -> str:
    final_str = "TorchBasedFeatureStore"
    indent_len = len(final_str)

    def _add_indent(_str, indent):
        lines = _str.split("\n")
        lines = [lines[0]] + [" " * indent + line for line in lines[1:]]
        return "\n".join(lines)

    features_str = _add_indent(str(features), indent_len)
    final_str += features_str
    return final_str
