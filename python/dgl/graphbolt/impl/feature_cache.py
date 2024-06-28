"""HugeCTR gpu_cache wrapper for graphbolt."""
import torch

__all__ = ["FeatureCache"]


class FeatureCache(object):
    """High-level wrapper for CPU embedding cache"""

    def __init__(self, cache_shape, dtype):
        self._cache = torch.ops.graphbolt.feature_cache(cache_shape, dtype)
        self.total_miss = 0
        self.total_queries = 0

    def query(self, keys, pin_memory=False):
        """Queries the GPU cache.

        Parameters
        ----------
        keys : Tensor
            The keys to query the GPU cache with.

        Returns
        -------
        tuple(Tensor, Tensor, Tensor)
            A tuple containing (values, missing_indices, missing_keys) where
            values[missing_indices] corresponds to cache misses that should be
            filled by quering another source with missing_keys.
        """
        self.total_queries += keys.shape[0]
        values, missing_index, missing_keys = self._cache.query(
            keys, pin_memory
        )
        self.total_miss += missing_keys.shape[0]
        return values, missing_index, missing_keys

    def replace(self, keys, values):
        """Inserts key-value pairs into the GPU cache using the S3-FIFO
        algorithm to remove old key-value pairs if it is full.

        Parameters
        ----------
        keys: Tensor
            The keys to insert to the GPU cache.
        values: Tensor
            The values to insert to the GPU cache.
        """
        self._cache.replace(keys, values)

    @property
    def miss_rate(self):
        """Returns the cache miss rate since creation."""
        return self.total_miss / self.total_queries
