"""HugeCTR gpu_cache wrapper for graphbolt."""
import torch

__all__ = ["FeatureCache"]


class FeatureCache(object):
    r"""High level wrapper for the CPU feature cache.

    Parameters
    ----------
    cache_shape : List[int]
        The shape of the cache. cache_shape[0] gives us the capacity.
    dtype : torch.dtype
        The data type of the elements stored in the cache.
    num_parts: int, optional
        The number of cache partitions for parallelism. Default is 1.
    policy: str, optional
        The cache policy to be used. Default is "s3-fifo".
    """

    def __init__(self, cache_shape, dtype, num_parts=1, policy="s3-fifo"):
        policies = ["s3-fifo"]
        assert (
            policy in policies
        ), f"{policies} are the available caching policies."
        assert num_parts >= 1
        self._policy = (
            torch.ops.graphbolt.s3_fifo_cache_policy(cache_shape[0])
            if num_parts == 1
            else torch.ops.graphbolt.partitioned_s3_fifo_cache_policy(
                cache_shape[0], num_parts
            )
        )
        self._cache = torch.ops.graphbolt.feature_cache(cache_shape, dtype)
        self.total_miss = 0
        self.total_queries = 0

    def query(self, keys, pin_memory=False):
        """Queries the cache.

        Parameters
        ----------
        keys : Tensor
            The keys to query the cache with.
        pin_memory : bool, optional
            Whether the output values tensor should be pinned. Default is False.

        Returns
        -------
        tuple(Tensor, Tensor, Tensor)
            A tuple containing (values, missing_indices, missing_keys) where
            values[missing_indices] corresponds to cache misses that should be
            filled by quering another source with missing_keys.
        """
        self.total_queries += keys.shape[0]
        positions, index, missing_keys = self._policy.query(keys)
        values = self._cache.query(positions, index, keys.shape[0], pin_memory)
        self.total_miss += missing_keys.shape[0]
        missing_index = index[positions.size(0) :]
        return values, missing_index, missing_keys

    def replace(self, keys, values):
        """Inserts key-value pairs into the cache using the S3-FIFO
        algorithm to remove old key-value pairs if it is full.

        Parameters
        ----------
        keys: Tensor
            The keys to insert to the cache.
        values: Tensor
            The values to insert to the cache.
        """
        positions = self._policy.replace(keys)
        self._cache.replace(keys, values, positions)

    @property
    def miss_rate(self):
        """Returns the cache miss rate since creation."""
        return self.total_miss / self.total_queries
