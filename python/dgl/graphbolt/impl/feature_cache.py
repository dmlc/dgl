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

    def query(self, keys):
        """Queries the cache.

        Parameters
        ----------
        keys : Tensor
            The keys to query the cache with.

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
        ) = self._policy.query(keys)
        values = self._cache.query(positions, index, keys.shape[0])
        self._policy.reading_completed(found_pointers, found_offsets)
        self.total_miss += missing_keys.shape[0]
        missing_index = index[positions.size(0) :]
        return values, missing_index, missing_keys, missing_offsets

    def replace(self, keys, values, offsets=None):
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
        """
        positions, pointers, offsets = self._policy.replace(keys, offsets)
        self._cache.replace(positions, values)
        self._policy.writing_completed(pointers, offsets)

    @property
    def miss_rate(self):
        """Returns the cache miss rate since creation."""
        return self.total_miss / self.total_queries
