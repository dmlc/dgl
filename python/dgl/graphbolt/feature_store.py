"""Feature store for GraphBolt."""
import torch


class FeatureStore:
    r"""Base class for feature store."""

    def __init__(self):
        pass

    def read(self, ids: torch.Tensor = None):
        """Read from the feature store.

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
        raise NotImplementedError

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
        raise NotImplementedError


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
