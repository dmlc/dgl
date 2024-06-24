"""HugeCTR gpu_cache wrapper for graphbolt."""
import torch


class GPUGraphCache(object):
    r"""High-level wrapper for GPU graph cache.

    Places the GPU graph cache to torch.cuda.current_device().

    Parameters
    ----------
    num_edges : int
        Upperbound on number of edges to cache.
    threshold : int
        The number of accesses before the neighborhood of a vertex is cached.
    indptr_dtype : torch.dtype
        The dtype of the indptr tensor of the graph.
    dtypes : list[torch.dtype]
        The dtypes of the edge tensors that are going to be cached.
    """

    def __init__(self, num_edges, threshold, indptr_dtype, dtypes):
        major, _ = torch.cuda.get_device_capability()
        assert (
            major >= 7
        ), "GPUGraphCache is supported only on CUDA compute capability >= 70 (Volta)."
        self._cache = torch.ops.graphbolt.gpu_graph_cache(
            num_edges, threshold, indptr_dtype, dtypes
        )
        self.total_miss = 0
        self.total_queries = 0

    def query(self, keys):
        """Queries the GPU cache.

        Parameters
        ----------
        keys : Tensor
            The keys to query the GPU graph cache with.

        Returns
        -------
        tuple(Tensor, func)
            A tuple containing (missing_keys, replace_fn) where replace_fn is a
            function that should be called with the graph structure
            corresponding to the missing keys. Its arguments are
            (Tensor, list(Tensor)).
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
            return self.combine_fetched_graphs(
                indptr,
                edge_tensors,
                selected_index,
                missing_indptr,
                missing_edge_tensors,
                missing_index,
            )

        return keys[missing_index], replace_functional

    @staticmethod
    def combine_fetched_graphs(
        indptr1, edge_tensors1, index1, indptr2, edge_tensors2, index2
    ):
        """Combines the graph structure found in the cache and the fetched graph
        structure from an outside source into a single graph structure.
        """
        if index2.size(0) == 0:
            return indptr1, edge_tensors1
        permutation = torch.cat([index1, index2]).sort()[1]
        assert len(edge_tensors1) == len(edge_tensors2)
        indptr = torch.cat([indptr1[:-1], indptr2 + indptr1[-1]])
        edge_tensors = []
        for a, b in zip(edge_tensors1, edge_tensors2):
            output_indptr, e = torch.ops.graphbolt.index_select_csc(
                indptr,
                torch.cat([a, b]),
                permutation,
                a.size(0) + b.size(0),
            )
            edge_tensors.append(e)
        return output_indptr, edge_tensors

    @property
    def miss_rate(self):
        """Returns the cache miss rate since creation."""
        return self.total_miss / self.total_queries
