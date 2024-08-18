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
    has_original_edge_ids : bool
        Whether the graph to be cached has original edge ids.
    """

    def __init__(
        self, num_edges, threshold, indptr_dtype, dtypes, has_original_edge_ids
    ):
        major, _ = torch.cuda.get_device_capability()
        assert (
            major >= 7
        ), "GPUGraphCache is supported only on CUDA compute capability >= 70 (Volta)."
        self._cache = torch.ops.graphbolt.gpu_graph_cache(
            num_edges, threshold, indptr_dtype, dtypes, has_original_edge_ids
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
            (Tensor, list(Tensor)), where the first tensor is the missing indptr
            and the second list is the missing edge tensors.
        """
        self.total_queries += keys.shape[0]
        (
            index,
            position,
            num_hit,
            num_threshold,
        ) = self._cache.query(keys)
        self.total_miss += keys.shape[0] - num_hit

        def replace_functional(missing_indptr, missing_edge_tensors):
            return self._cache.replace(
                keys,
                index,
                position,
                num_hit,
                num_threshold,
                missing_indptr,
                missing_edge_tensors,
            )

        return keys[index[num_hit:]], replace_functional

    def query_async(self, keys):
        """Queries the GPU cache asynchronously.

        Parameters
        ----------
        keys : Tensor
            The keys to query the GPU graph cache with.

        Returns
        -------
        A generator object.
            The returned generator object returns the missing keys on the second
            invocation and expects the fetched indptr and edge tensors on the
            next invocation. The third and last invocation returns a future
            object and the return result can be accessed by calling `.wait()`
            on the returned future object. It is undefined behavior to call
            `.wait()` more than once.
        """
        future = self._cache.query_async(keys)

        yield

        index, position, num_hit, num_threshold = future.wait()

        self.total_queries += keys.shape[0]
        self.total_miss += keys.shape[0] - num_hit

        missing_indptr, missing_edge_tensors = yield keys[index[num_hit:]]

        yield self._cache.replace_async(
            keys,
            index,
            position,
            num_hit,
            num_threshold,
            missing_indptr,
            missing_edge_tensors,
        )

    @property
    def miss_rate(self):
        """Returns the cache miss rate since creation."""
        return self.total_miss / self.total_queries
