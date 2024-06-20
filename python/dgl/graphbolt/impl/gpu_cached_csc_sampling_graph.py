"""HugeCTR gpu_cache wrapper for graphbolt."""
import torch


class GPUGraphCache(object):
    """High-level wrapper for GPU graph cache"""

    def __init__(self, num_edges, dtypes):
        major, _ = torch.cuda.get_device_capability()
        assert (
            major >= 7
        ), "GPUCache is supported only on CUDA compute capability >= 70 (Volta)."
        self._cache = torch.ops.graphbolt.gpu_graph_cache(num_edges, dtypes)
        self.total_miss = 0
        self.total_queries = 0

    def query(self, keys):
        """Queries the GPU cache.

        Parameters
        ----------
        keys : Tensor
            The keys to query the GPU cache with.

        Returns
        -------
        tuple(Tensor, Tensor, Tensor)
            A tuple containing (indptr, missing_indices, missing_keys) where
            values[missing_indices] corresponds to cache misses that should be
            filled by quering another source with missing_keys.
        """
        self.total_queries += keys.shape[0]
        (
            indptr,
            edge_tensors,
            selected_index,
            missing_index,
            missing_position,
            num_cache_enter,
        ) = self._cache.query(keys)
        self.total_miss += missing_index.shape[0]

        def replace_functional(missing_indptr, missing_edge_tensors):
            if num_cache_enter > 0:
                self._cache.replace(
                    keys,
                    missing_index,
                    missing_position,
                    num_cache_enter,
                    missing_indptr,
                    missing_edge_tensors,
                )

        return indptr, edge_tensors, selected_index, replace_functional

    @staticmethod
    def combine_fetched_graphs(
        indptr1, edge_tensors1, index1, indptr2, edge_tensors2, index2
    ):
        permutation = torch.cat([index1, index2]).sort()[1]
        assert len(edge_tensors1) == len(edge_tensors2)
        indptr = torch.cat([indptr1[:-1], indptr2 + indptr1[-1]])
        edge_tensors = []
        for e1, e2 in zip(edge_tensors1, edge_tensors2):
            output_indptr, e = torch.ops.graphbolt.index_select_csc(
                indptr,
                torch.cat([e1, e2]),
                permutation,
                e1.size(0) + e2.size(0),
            )
            edge_tensors.append(e)
        return output_indptr, edge_tensors

    @property
    def miss_rate(self):
        """Returns the cache miss rate since creation."""
        return self.total_miss / self.total_queries
