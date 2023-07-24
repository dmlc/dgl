"""Torch-based feature store for GraphBolt."""
from typing import List

import numpy as np

import torch

from ..feature_store import FeatureStore
from .ondisk_metadata import OnDiskFeatureData

__all__ = ["TorchBasedFeatureStore", "load_feature_stores"]


class TorchBasedFeatureStore(FeatureStore):
    r"""Torch based feature store."""

    def __init__(self, torch_feature: torch.Tensor):
        """Initialize a torch based feature store by a torch feature.

        Note that the feature can be either in memory or on disk.

        Parameters
        ----------
        torch_feature : torch.Tensor
            The torch feature.

        Examples
        --------
        >>> import torch
        >>> torch_feat = torch.arange(0, 5)
        >>> feature_store = TorchBasedFeatureStore(torch_feat)
        >>> feature_store.read()
        tensor([0, 1, 2, 3, 4])
        >>> feature_store.read(torch.tensor([0, 1, 2]))
        tensor([0, 1, 2])
        >>> feature_store.update(torch.ones(3, dtype=torch.long),
        ... torch.tensor([0, 1, 2]))
        >>> feature_store.read(torch.tensor([0, 1, 2, 3]))
        tensor([1, 1, 1, 3])

        >>> import numpy as np
        >>> arr = np.arange(0, 5)
        >>> np.save("/tmp/arr.npy", arr)
        >>> torch_feat = torch.as_tensor(np.load("/tmp/arr.npy",
        ...         mmap_mode="r+"))
        >>> feature_store = TorchBasedFeatureStore(torch_feat)
        >>> feature_store.read()
        tensor([0, 1, 2, 3, 4])
        >>> feature_store.read(torch.tensor([0, 1, 2]))
        tensor([0, 1, 2])
        """
        super(TorchBasedFeatureStore, self).__init__()
        assert isinstance(torch_feature, torch.Tensor), (
            f"torch_feature in TorchBasedFeatureStore must be torch.Tensor, "
            f"but got {type(torch_feature)}."
        )
        self._tensor = torch_feature

    def read(self, ids: torch.Tensor = None):
        """Read the feature by index.

        The returned tensor is always in memory, no matter whether the feature
        store is in memory or on disk.

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
            return self._tensor
        return self._tensor[ids]

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
            assert self._tensor.shape == value.shape, (
                f"ids is None, so the entire feature will be updated. "
                f"But the shape of the feature is {self._tensor.shape}, "
                f"while the shape of the value is {value.shape}."
            )
            self._tensor[:] = value
        else:
            assert ids.shape[0] == value.shape[0], (
                f"ids and value must have the same length, "
                f"but got {ids.shape[0]} and {value.shape[0]}."
            )
            self._tensor[ids] = value


def load_feature_stores(feat_data: List[OnDiskFeatureData]):
    r"""Load feature stores from disk.

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

    Returns
    -------
    dict
        The loaded feature stores. The keys are the names of the feature stores,
        and the values are the feature stores.

    Examples
    --------
    >>> import torch
    >>> import numpy as np
    >>> from dgl import graphbolt as gb
    >>> edge_label = torch.tensor([1, 2, 3])
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
    >>> gb.load_feature_stores(feat_data)
    ... {("edge", "author:writes:paper", "label"):
    ... <dgl.graphbolt.feature_store.TorchBasedFeatureStore object at
    ... 0x7ff093cb4df0>, ("node", "paper", "feat"):
    ... <dgl.graphbolt.feature_store.TorchBasedFeatureStore object at
    ... 0x7ff093cb4dc0>}
    """
    feat_stores = {}
    for spec in feat_data:
        key = (spec.domain, spec.type, spec.name)
        if spec.format == "torch":
            assert spec.in_memory, (
                f"Pytorch tensor can only be loaded in memory, "
                f"but the feature {key} is loaded on disk."
            )
            feat_stores[key] = TorchBasedFeatureStore(torch.load(spec.path))
        elif spec.format == "numpy":
            mmap_mode = "r+" if not spec.in_memory else None
            feat_stores[key] = TorchBasedFeatureStore(
                torch.as_tensor(np.load(spec.path, mmap_mode=mmap_mode))
            )
        else:
            raise ValueError(f"Unknown feature format {spec.format}")
    return feat_stores
