"""CPU cached feature for GraphBolt."""

import torch

from ..feature_store import Feature

from .feature_cache import CPUFeatureCache

__all__ = ["CPUCachedFeature"]


def bytes_to_number_of_items(cache_capacity_in_bytes, single_item):
    """Returns the number of rows to be cached."""
    item_bytes = single_item.nbytes
    # Round up so that we never get a size of 0, unless bytes is 0.
    return (cache_capacity_in_bytes + item_bytes - 1) // item_bytes


class CPUCachedFeature(Feature):
    r"""CPU cached feature wrapping a fallback feature.

    Parameters
    ----------
    fallback_feature : Feature
        The fallback feature.
    max_cache_size_in_bytes : int
        The capacity of the cache in bytes.
    policy : str
        The cache eviction policy algorithm name. See gb.impl.CPUFeatureCache
        for the list of available policies.
    pin_memory : bool
        Whether the cache storage should be allocated on system pinned memory.
        Default is False.
    """

    def __init__(
        self,
        fallback_feature: Feature,
        max_cache_size_in_bytes: int,
        policy: str = None,
        pin_memory: bool = False,
    ):
        super(CPUCachedFeature, self).__init__()
        assert isinstance(fallback_feature, Feature), (
            f"The fallback_feature must be an instance of Feature, but got "
            f"{type(fallback_feature)}."
        )
        self._fallback_feature = fallback_feature
        self.max_cache_size_in_bytes = max_cache_size_in_bytes
        # Fetching the feature dimension from the underlying feature.
        feat0 = fallback_feature.read(torch.tensor([0]))
        cache_size = bytes_to_number_of_items(max_cache_size_in_bytes, feat0)
        self._feature = CPUFeatureCache(
            (cache_size,) + feat0.shape[1:],
            feat0.dtype,
            policy=policy,
            pin_memory=pin_memory,
        )

    def read(self, ids: torch.Tensor = None):
        """Read the feature by index.

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
            return self._fallback_feature.read()
        values, missing_index, missing_keys = self._feature.query(ids)
        missing_values = self._fallback_feature.read(missing_keys)
        values[missing_index] = missing_values
        self._feature.replace(missing_keys, missing_values)
        return values

    def read_async(self, ids: torch.Tensor):
        """Read the feature by index asynchronously.
        Parameters
        ----------
        ids : torch.Tensor
            The index of the feature. Only the specified indices of the
            feature are read.
        Returns
        -------
        A generator object.
            The returned generator object returns a future on
            `read_async_num_stages(ids.device)`th invocation. The return result
            can be accessed by calling `.wait()`. on the returned future object.
            It is undefined behavior to call `.wait()` more than once.
        Example Usage
        --------
        >>> import dgl.graphbolt as gb
        >>> feature = gb.Feature(...)
        >>> ids = torch.tensor([0, 2])
        >>> async_handle = feature.read_async(ids)
        >>> for _ in range(feature.read_async_num_stages(ids.device)):
        ...     future = next(async_handle)
        >>> result = future.wait()  # result contains the read values.
        """
        raise NotImplementedError

    def read_async_num_stages(self, ids_device: torch.device):
        """The number of stages of the read_async operation. See read_async
        function for directions on its use.
        Parameters
        ----------
        ids_device : torch.device
            The device of the ids parameter passed into read_async.
        Returns
        -------
        int
            The number of stages of the read_async operation.
        """
        raise NotImplementedError

    def size(self):
        """Get the size of the feature.

        Returns
        -------
        torch.Size
            The size of the feature.
        """
        return self._fallback_feature.size()

    def update(self, value: torch.Tensor, ids: torch.Tensor = None):
        """Update the feature.

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
            feat0 = value[:1]
            self._fallback_feature.update(value)
            cache_size = min(
                bytes_to_number_of_items(self.max_cache_size_in_bytes, feat0),
                value.shape[0],
            )
            self._feature = None  # Destroy the existing cache first.
            self._feature = CPUFeatureCache(
                (cache_size,) + feat0.shape[1:], feat0.dtype
            )
        else:
            self._fallback_feature.update(value, ids)
            self._feature.replace(ids, value)
