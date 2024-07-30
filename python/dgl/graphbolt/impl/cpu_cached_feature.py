"""CPU cached feature for GraphBolt."""

import torch

from ..base import get_device_to_host_uva_stream, get_host_to_device_uva_stream
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
        self._is_pinned = pin_memory

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
        return self._feature.query_and_then_replace(
            ids, self._fallback_feature.read
        )

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
        >>> for stage, future in enumerate(feature.read_async(ids)):
        ...     pass
        >>> assert stage + 1 == feature.read_async_num_stages(ids.device)
        >>> result = future.wait()  # result contains the read values.
        """
        policy = self._feature._policy
        cache = self._feature._cache
        if ids.is_cuda and self._is_pinned:
            ids_device = ids.device
            current_stream = torch.cuda.current_stream()
            device_to_host_stream = get_device_to_host_uva_stream()
            device_to_host_stream.wait_stream(current_stream)
            with torch.cuda.stream(device_to_host_stream):
                ids.record_stream(torch.cuda.current_stream())
                ids = ids.to("cpu", non_blocking=True)
                ids_copy_event = torch.cuda.Event()
                ids_copy_event.record()

            yield  # first stage is done.

            ids_copy_event.synchronize()
            policy_future = policy.query_async(ids)

            yield

            (
                positions,
                index,
                missing_keys,
                found_pointers,
                found_offsets,
                missing_offsets,
            ) = policy_future.wait()
            self._feature.total_queries += ids.shape[0]
            self._feature.total_miss += missing_keys.shape[0]
            host_to_device_stream = get_host_to_device_uva_stream()
            with torch.cuda.stream(host_to_device_stream):
                positions_cuda = positions.to(ids_device, non_blocking=True)
                values_from_cpu = cache.index_select(positions_cuda)
                values_from_cpu.record_stream(current_stream)
                values_from_cpu_copy_event = torch.cuda.Event()
                values_from_cpu_copy_event.record()

            positions_future = policy.replace_async(
                missing_keys, missing_offsets
            )

            fallback_reader = self._fallback_feature.read_async(missing_keys)
            for _ in range(
                self._fallback_feature.read_async_num_stages(
                    missing_keys.device
                )
            ):
                missing_values_future = next(fallback_reader, None)
                yield  # fallback feature stages.

            values_from_cpu_copy_event.wait()
            reading_completed = policy.reading_completed_async(
                found_pointers, found_offsets
            )

            missing_values = missing_values_future.wait()
            positions, pointers, offsets = positions_future.wait()
            replace_future = cache.replace_async(positions, missing_values)

            host_to_device_stream = get_host_to_device_uva_stream()
            with torch.cuda.stream(host_to_device_stream):
                index = index.to(ids_device, non_blocking=True)
                missing_values_cuda = missing_values.to(
                    ids_device, non_blocking=True
                )
                index.record_stream(current_stream)
                missing_values_cuda.record_stream(current_stream)
                missing_values_copy_event = torch.cuda.Event()
                missing_values_copy_event.record()

            yield

            reading_completed.wait()
            replace_future.wait()
            writing_completed = policy.writing_completed_async(
                pointers, offsets
            )

            class _Waiter:
                def __init__(self, events, existing, missing, index):
                    self.events = events
                    self.existing = existing
                    self.missing = missing
                    self.index = index

                def wait(self):
                    """Returns the stored value when invoked."""
                    for event in self.events:
                        event.wait()
                    values = torch.empty(
                        (self.index.shape[0],) + self.missing.shape[1:],
                        dtype=self.missing.dtype,
                        device=ids_device,
                    )
                    num_found = self.existing.size(0)
                    found_index = self.index[:num_found]
                    missing_index = self.index[num_found:]
                    values[found_index] = self.existing
                    values[missing_index] = self.missing
                    # Ensure there is no memory leak.
                    self.events = self.existing = None
                    self.missing = self.index = None
                    return values

            yield _Waiter(
                [missing_values_copy_event, writing_completed],
                values_from_cpu,
                missing_values_cuda,
                index,
            )
        elif ids.is_cuda:
            ids_device = ids.device
            current_stream = torch.cuda.current_stream()
            device_to_host_stream = get_device_to_host_uva_stream()
            device_to_host_stream.wait_stream(current_stream)
            with torch.cuda.stream(device_to_host_stream):
                ids.record_stream(torch.cuda.current_stream())
                ids = ids.to("cpu", non_blocking=True)
                ids_copy_event = torch.cuda.Event()
                ids_copy_event.record()

            yield  # first stage is done.

            ids_copy_event.synchronize()
            policy_future = policy.query_async(ids)

            yield

            (
                positions,
                index,
                missing_keys,
                found_pointers,
                found_offsets,
                missing_offsets,
            ) = policy_future.wait()
            self._feature.total_queries += ids.shape[0]
            self._feature.total_miss += missing_keys.shape[0]
            values_future = cache.query_async(positions, index, ids.shape[0])

            positions_future = policy.replace_async(
                missing_keys, missing_offsets
            )

            fallback_reader = self._fallback_feature.read_async(missing_keys)
            for _ in range(
                self._fallback_feature.read_async_num_stages(
                    missing_keys.device
                )
            ):
                missing_values_future = next(fallback_reader, None)
                yield  # fallback feature stages.

            values = values_future.wait()
            reading_completed = policy.reading_completed_async(
                found_pointers, found_offsets
            )

            missing_index = index[positions.size(0) :]

            missing_values = missing_values_future.wait()
            positions, pointers, offsets = positions_future.wait()
            replace_future = cache.replace_async(positions, missing_values)
            values = torch.ops.graphbolt.scatter_async(
                values, missing_index, missing_values
            )

            yield

            host_to_device_stream = get_host_to_device_uva_stream()
            with torch.cuda.stream(host_to_device_stream):
                values_cuda = values.wait().to(ids_device, non_blocking=True)
                values_cuda.record_stream(current_stream)
                values_copy_event = torch.cuda.Event()
                values_copy_event.record()

            reading_completed.wait()
            replace_future.wait()
            writing_completed = policy.writing_completed_async(
                pointers, offsets
            )

            class _Waiter:
                def __init__(self, events, values):
                    self.events = events
                    self.values = values

                def wait(self):
                    """Returns the stored value when invoked."""
                    for event in self.events:
                        event.wait()
                    values = self.values
                    # Ensure there is no memory leak.
                    self.events = self.values = None
                    return values

            yield _Waiter([values_copy_event, writing_completed], values_cuda)
        else:
            policy_future = policy.query_async(ids)

            yield

            (
                positions,
                index,
                missing_keys,
                found_pointers,
                found_offsets,
                missing_offsets,
            ) = policy_future.wait()
            self._feature.total_queries += ids.shape[0]
            self._feature.total_miss += missing_keys.shape[0]
            values_future = cache.query_async(positions, index, ids.shape[0])

            positions_future = policy.replace_async(
                missing_keys, missing_offsets
            )

            fallback_reader = self._fallback_feature.read_async(missing_keys)
            for _ in range(
                self._fallback_feature.read_async_num_stages(
                    missing_keys.device
                )
            ):
                missing_values_future = next(fallback_reader, None)
                yield  # fallback feature stages.

            values = values_future.wait()
            reading_completed = policy.reading_completed_async(
                found_pointers, found_offsets
            )

            missing_index = index[positions.size(0) :]

            missing_values = missing_values_future.wait()
            positions, pointers, offsets = positions_future.wait()
            replace_future = cache.replace_async(positions, missing_values)
            values = torch.ops.graphbolt.scatter_async(
                values, missing_index, missing_values
            )

            yield

            reading_completed.wait()
            replace_future.wait()
            writing_completed = policy.writing_completed_async(
                pointers, offsets
            )

            class _Waiter:
                def __init__(self, event, values):
                    self.event = event
                    self.values = values

                def wait(self):
                    """Returns the stored value when invoked."""
                    self.event.wait()
                    values = self.values.wait()
                    # Ensure there is no memory leak.
                    self.event = self.values = None
                    return values

            yield _Waiter(writing_completed, values)

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
            return 4 + self._fallback_feature.read_async_num_stages(
                torch.device("cpu")
            )
        else:
            return 3 + self._fallback_feature.read_async_num_stages(ids_device)

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
