"""CPU Feature Cache implementation wrapper for graphbolt."""
import torch

__all__ = ["CPUFeatureCache"]

caching_policies = {
    "s3-fifo": torch.ops.graphbolt.s3_fifo_cache_policy,
    "sieve": torch.ops.graphbolt.sieve_cache_policy,
    "lru": torch.ops.graphbolt.lru_cache_policy,
    "clock": torch.ops.graphbolt.clock_cache_policy,
}


class CPUFeatureCache(object):
    r"""High level wrapper for the CPU feature cache.

    Parameters
    ----------
    cache_shape : List[int]
        The shape of the cache. cache_shape[0] gives us the capacity.
    dtype : torch.dtype
        The data type of the elements stored in the cache.
    policy: str, optional
        The cache policy. Default is "sieve". "s3-fifo", "lru" and "clock" are
        also available.
    num_parts: int, optional
        The number of cache partitions for parallelism. Default is
        `torch.get_num_threads()`.
    pin_memory: bool, optional
        Whether the cache storage should be pinned.
    """

    def __init__(
        self,
        cache_shape,
        dtype,
        policy=None,
        num_parts=None,
        pin_memory=False,
    ):
        if policy is None:
            policy = "sieve"
        assert (
            policy in caching_policies
        ), f"{list(caching_policies.keys())} are the available caching policies."
        if num_parts is None:
            num_parts = torch.get_num_threads()
        min_num_cache_items = num_parts * (10 if policy == "s3-fifo" else 1)
        # Since we partition the cache, each partition needs to have a positive
        # number of slots. In addition, each "s3-fifo" partition needs at least
        # 10 slots since the small queue is 10% and the small queue needs a
        # positive size.
        if cache_shape[0] < min_num_cache_items:
            cache_shape = (min_num_cache_items,) + cache_shape[1:]
        self._policy = caching_policies[policy](cache_shape[0], num_parts)
        self._cache = torch.ops.graphbolt.feature_cache(
            cache_shape, dtype, pin_memory
        )
        self.total_miss = 0
        self.total_queries = 0

    def is_pinned(self):
        """Returns True if the cache storage is pinned."""
        return self._cache.is_pinned()

    @property
    def max_size_in_bytes(self):
        """Return the size taken by the cache in bytes."""
        return self._cache.nbytes

    def query(self, keys, offset=0):
        """Queries the cache.

        Parameters
        ----------
        keys : Tensor
            The keys to query the cache with.
        offset : int
            The offset to be added to the keys. Default is 0.

        Returns
        -------
        tuple(Tensor, Tensor, Tensor, Tensor)
            A tuple containing
            (values, missing_indices, missing_keys, missing_offsets) where
            values[missing_indices] corresponds to cache misses that should be
            filled by quering another source with missing_keys. If keys is
            pinned, then the returned values tensor is pinned as well. The
            missing_offsets tensor has the partition offsets of missing_keys.
        """
        self.total_queries += keys.shape[0]
        (
            positions,
            index,
            missing_keys,
            found_pointers,
            found_offsets,
            missing_offsets,
        ) = self._policy.query(keys, offset)
        values = self._cache.query(positions, index, keys.shape[0])
        self._policy.reading_completed(found_pointers, found_offsets)
        self.total_miss += missing_keys.shape[0]
        missing_index = index[positions.size(0) :]
        return values, missing_index, missing_keys, missing_offsets

    def query_and_replace(self, keys, reader_fn, offset=0):
        """Queries the cache. Then inserts the keys that are not found by
        reading them by calling `reader_fn(missing_keys)`, which are then
        inserted into the cache using the selected caching policy algorithm
        to remove the old entries if it is full.

        Parameters
        ----------
        keys : Tensor
            The keys to query the cache with.
        reader_fn : reader_fn(keys: torch.Tensor) -> torch.Tensor
            A function that will take a missing keys tensor and will return
            their values.
        offset : int
            The offset to be added to the keys. Default is 0.

        Returns
        -------
        Tensor
            A tensor containing values corresponding to the keys. Should equal
            `reader_fn(keys)`, computed in a faster way.
        """
        self.total_queries += keys.shape[0]
        (
            positions,
            index,
            pointers,
            missing_keys,
            found_offsets,
            missing_offsets,
        ) = self._policy.query_and_replace(keys, offset)
        found_cnt = keys.size(0) - missing_keys.size(0)
        found_positions = positions[:found_cnt]
        values = self._cache.query(found_positions, index, keys.shape[0])
        found_pointers = pointers[:found_cnt]
        self._policy.reading_completed(found_pointers, found_offsets)
        self.total_miss += missing_keys.shape[0]
        missing_index = index[found_cnt:]
        missing_values = reader_fn(missing_keys)
        values[missing_index] = missing_values
        missing_positions = positions[found_cnt:]
        self._cache.replace(missing_positions, missing_values)
        missing_pointers = pointers[found_cnt:]
        self._policy.writing_completed(missing_pointers, missing_offsets)
        return values

    def replace(self, keys, values, offsets=None, offset=0):
        """Inserts key-value pairs into the cache using the selected caching
        policy algorithm to remove old key-value pairs if it is full.

        Parameters
        ----------
        keys : Tensor
            The keys to insert to the cache.
        values : Tensor
            The values to insert to the cache.
        offsets : Tensor, optional
            The partition offsets of the keys.
        offset : int
            The offset to be added to the keys. Default is 0.
        """
        positions, pointers, offsets = self._policy.replace(
            keys, offsets, offset
        )
        self._cache.replace(positions, values)
        self._policy.writing_completed(pointers, offsets)

    @property
    def miss_rate(self):
        """Returns the cache miss rate since creation."""
        return self.total_miss / self.total_queries
