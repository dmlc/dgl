"""GPU cached feature for GraphBolt."""

import torch

from ..feature_store import Feature

from .gpu_cache import GPUCache

__all__ = ["GPUCachedFeature"]


def num_cache_items(cache_capacity_in_bytes, single_item):
    """Returns the number of rows to be cached."""
    item_bytes = single_item.nbytes
    # Round up so that we never get a size of 0, unless bytes is 0.
    return (cache_capacity_in_bytes + item_bytes - 1) // item_bytes


class GPUCachedFeature(Feature):
    r"""GPU cached feature wrapping a fallback feature.

    Places the GPU cache to torch.cuda.current_device().

    Parameters
    ----------
    fallback_feature : Feature
        The fallback feature.
    max_cache_size_in_bytes : int
        The capacity of the GPU cache in bytes.

    Examples
    --------
    >>> import torch
    >>> from dgl import graphbolt as gb
    >>> torch_feat = torch.arange(10).reshape(2, -1).to("cuda")
    >>> cache_size = 5
    >>> fallback_feature = gb.TorchBasedFeature(torch_feat)
    >>> feature = gb.GPUCachedFeature(fallback_feature, cache_size)
    >>> feature.read()
    tensor([[0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9]], device='cuda:0')
    >>> feature.read(torch.tensor([0]).to("cuda"))
    tensor([[0, 1, 2, 3, 4]], device='cuda:0')
    >>> feature.update(torch.tensor([[1 for _ in range(5)]]).to("cuda"),
    ...                torch.tensor([1]).to("cuda"))
    >>> feature.read(torch.tensor([0, 1]).to("cuda"))
    tensor([[0, 1, 2, 3, 4],
            [1, 1, 1, 1, 1]], device='cuda:0')
    >>> feature.size()
    torch.Size([5])
    """

    def __init__(self, fallback_feature: Feature, max_cache_size_in_bytes: int):
        super(GPUCachedFeature, self).__init__()
        assert isinstance(fallback_feature, Feature), (
            f"The fallback_feature must be an instance of Feature, but got "
            f"{type(fallback_feature)}."
        )
        self._fallback_feature = fallback_feature
        self.max_cache_size_in_bytes = max_cache_size_in_bytes
        # Fetching the feature dimension from the underlying feature.
        feat0 = fallback_feature.read(torch.tensor([0]))
        cache_size = num_cache_items(max_cache_size_in_bytes, feat0)
        self._feature = GPUCache((cache_size,) + feat0.shape[1:], feat0.dtype)

    def read(self, ids: torch.Tensor = None):
        """Read the feature by index.

        The returned tensor is always in GPU memory, no matter whether the
        fallback feature is in memory or on disk.

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
            `read_async_num_stages(ids.device)`th invocation. The return result
            can be accessed by calling `.wait()`. on the returned future object.
            It is undefined behavior to call `.wait()` more than once.

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
        values, missing_index, missing_keys = self._feature.query(ids)

        fallback_reader = self._fallback_feature.read_async(missing_keys)
        fallback_num_stages = self._fallback_feature.read_async_num_stages(
            missing_keys.device
        )
        for i in range(fallback_num_stages):
            missing_values_future = next(fallback_reader, None)
            if i < fallback_num_stages - 1:
                yield  # fallback feature stages.

        class _Waiter:
            def __init__(
                self,
                feature,
                values,
                missing_index,
                missing_keys,
                missing_values_future,
            ):
                self.feature = feature
                self.values = values
                self.missing_index = missing_index
                self.missing_keys = missing_keys
                self.missing_values_future = missing_values_future

            def wait(self):
                """Returns the stored value when invoked."""
                missing_values = self.missing_values_future.wait()
                self.feature.replace(self.missing_keys, missing_values)
                self.values[self.missing_index] = missing_values
                values = self.values
                # Ensure there is no memory leak.
                self.feature = self.values = self.missing_index = None
                self.missing_keys = self.missing_values_future = None
                return values

        yield _Waiter(
            self._feature,
            values,
            missing_index,
            missing_keys,
            missing_values_future,
        )

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
        assert ids_device.type == "cuda"
        return self._fallback_feature.read_async_num_stages(ids_device)

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
                num_cache_items(self.max_cache_size_in_bytes, feat0),
                value.shape[0],
            )
            self._feature = None  # Destroy the existing cache first.
            self._feature = GPUCache(
                (cache_size,) + feat0.shape[1:], feat0.dtype
            )
        else:
            self._fallback_feature.update(value, ids)
            self._feature.replace(ids, value)
