"""Torch-based feature store for GraphBolt."""
import torch

from dgl.cuda import GPUCache

from ..feature_store import FeatureStore

__all__ = ["GPUCachedFeatureStore"]


class GPUCachedFeatureStore(FeatureStore):
    r"""GPU cached feature store wrapping a fallback feature store."""

    def __init__(self, fallback_store: FeatureStore, cache_size: int):
        """Initialize GPU cached feature store with a given fallback.
        Places the GPU cache to torch.cuda.current_device().

        Parameters
        ----------
        fallback_store : FeatureStore
            The fallback feature.
        cache_size : int
            The capacity of the GPU cache, the number of features to store.

        Examples
        --------
        >>> import torch
        >>> torch_feat = torch.arange(0, 5)
        >>> fallback_store = TorchBasedFeatureStore(torch_feat)
        >>> feature_store = GPUCachedFeatureStore(fallback_store, device)
        >>> feature_store.read()
        tensor([0, 1, 2, 3, 4])
        >>> feature_store.read(torch.tensor([0, 1, 2]))
        tensor([0, 1, 2])
        >>> feature_store.update(torch.ones(3, dtype=torch.long),
        ... torch.tensor([0, 1, 2]))
        >>> feature_store.read(torch.tensor([0, 1, 2, 3]))
        tensor([1, 1, 1, 3])
        """
        super(GPUCachedFeatureStore, self).__init__()
        assert isinstance(fallback_store, FeatureStore), (
            f"fallback_store must be FeatureStore, "
            f"but got {type(fallback_store)}."
        )
        self._fallback_store = fallback_store
        self.cache_size = cache_size
        feat0 = fallback_store.read(torch.tensor([0]))
        self.item_shape = feat0.shape[1:]
        feat0 = torch.reshape(feat0, (1, -1))
        self._store = GPUCache(cache_size, feat0.shape[1])

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
            return self._fallback_store.read()
        keys = ids.to("cuda")
        values, missing_index, missing_keys = self._store.query(keys)
        missing_values = self._fallback_store.read(missing_keys).to("cuda")
        values[missing_index] = missing_values
        self._store.replace(missing_keys, missing_values)
        return torch.reshape(values, (values.shape[0],) + self.item_shape)

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
            self._fallback_store.update(value)
            size = min(self.cache_size, value.shape[0])
            self._store.replace(
                torch.arange(0, size, device="cuda"),
                value[:size].to("cuda"),
            )
        else:
            assert ids.shape[0] == value.shape[0], (
                f"ids and value must have the same length, "
                f"but got {ids.shape[0]} and {value.shape[0]}."
            )
            self._fallback_store.update(value, ids)
            self._store.replace(ids.to("cuda"), value.to("cuda"))
