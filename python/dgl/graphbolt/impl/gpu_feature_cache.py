"""HugeCTR gpu_cache wrapper for graphbolt."""
from functools import reduce
from operator import mul

import torch


class GPUFeatureCache(object):
    """High-level wrapper for GPU embedding cache"""

    def __init__(self, cache_shape, dtype):
        major, _ = torch.cuda.get_device_capability()
        assert (
            major >= 7
        ), "GPUFeatureCache is supported only on CUDA compute capability >= 70 (Volta)."
        self._cache = torch.ops.graphbolt.gpu_cache(cache_shape, dtype)
        element_size = torch.tensor([], dtype=dtype).element_size()
        self.max_size_in_bytes = reduce(mul, cache_shape) * element_size
        self.total_miss = 0
        self.total_queries = 0

    def query(self, keys, async_op=False):
        """Queries the GPU cache.

        Parameters
        ----------
        keys : Tensor
            The keys to query the GPU cache with.
        async_op: bool
            Boolean indicating whether the call is asynchronous. If so, the
            result can be obtained by calling wait on the returned future.

        Returns
        -------
        tuple(Tensor, Tensor, Tensor)
            A tuple containing (values, missing_indices, missing_keys) where
            values[missing_indices] corresponds to cache misses that should be
            filled by quering another source with missing_keys.
        """

        class _Waiter:
            def __init__(self, gpu_cache, future):
                self.gpu_cache = gpu_cache
                self.future = future

            def wait(self):
                """Returns the stored value when invoked."""
                gpu_cache = self.gpu_cache
                values, missing_index, missing_keys = (
                    self.future.wait() if async_op else self.future
                )
                # Ensure there is no leak.
                self.gpu_cache = self.future = None

                gpu_cache.total_queries += values.shape[0]
                gpu_cache.total_miss += missing_keys.shape[0]
                return values, missing_index, missing_keys

        if async_op:
            return _Waiter(self, self._cache.query_async(keys))
        else:
            return _Waiter(self, self._cache.query(keys)).wait()

    def replace(self, keys, values):
        """Inserts key-value pairs into the GPU cache using the Least-Recently
        Used (LRU) algorithm to remove old key-value pairs if it is full.

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
