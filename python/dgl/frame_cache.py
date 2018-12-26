"""Cache for frames in DGLGraph."""
import math
import numpy as np

from . import backend as F
from . import utils
from .frame import Frame, FrameRef
from ._ffi.function import _init_api

class CachedFrame:
    """The cached frame.

    It contains both the frame and cached part of the frame,
    so we can fetch any data from it.

    Parameters
    ----------
    frame : FrameRef
        The frame that we want to create cache on.
    ids : utils.Index
        The row Ids whose are stored in the cache.
    ctx : context
        The context where the cache is stored.
    """
    def __init__(self, frame, ids, ctx):
        ids = F.sort_1d(ids.tousertensor(), return_type="data")
        self._cached_ids = utils.toindex(ids)
        self._frame = frame
        cols = {}
        for key in frame:
            col = frame[key]
            cols.update({key: F.copy_to(col[ids], ctx)})
        self._cache = cols
        self._ctx = ctx
        self._opt_lookup = {
            2: _CAPI_DGLCacheLookup2,
            4: _CAPI_DGLCacheLookup4,
            8: _CAPI_DGLCacheLookup8,
            16: _CAPI_DGLCacheLookup16,
            32: _CAPI_DGLCacheLookup32,
        }
        self._num_access = 0
        self._num_hits = 0

    @property
    def keys(self):
        """The names of the cached columns.
        """
        return self._cache.keys()

    @property
    def context(self):
        """The context where the cache is stored.
        """
        return self._ctx

    def refresh(self, keys):
        """Refresh cached data of the specified columns.

        Parameters
        ----------
        keys : list of string
            The names of the columns to be freshed.
        """
        ids = self._cached_ids.tousertensor()
        for key in keys:
            col = self._frame[key]
            self._cache.update({key: F.copy_to(col[ids], self._ctx)})

    def cache_lookup(self, ids):
        """Lookup in the cached frame.

        For performance, it is a batched version.

        Parameters
        ----------
        ids : list of utils.Index
            The row Ids in the frame.

        Returns
        -------
        List of SubgraphCachedFrame
            The cached frames for the subsets.
        """
        # TODO(zhengda) if the cache isn't ready, we shouldn't read the cache.
        ret = []
        dgl_ids = [i.todgltensor() for i in ids]
        empty_ids = []
        if len(dgl_ids) > 1 and len(dgl_ids) not in self._opt_lookup.keys():
            remain = 2**int(math.ceil(math.log2(len(dgl_ids)))) - len(dgl_ids)
            for _ in range(remain):
                empty = utils.toindex(F.empty(shape=(0), dtype=F.dtype(dgl_ids[0]), ctx=F.cpu()))
                empty_ids.append(empty)
                dgl_ids.append(empty.todgltensor())

        if len(dgl_ids) > 1:
            res = self._opt_lookup[len(dgl_ids)](self._cached_ids.todgltensor(), len(ids), *dgl_ids)
            for i, id in enumerate(ids):
                cached_out_idx = utils.toindex(res(i * 4))
                uncached_out_idx = utils.toindex(res(i * 4 + 1))
                cache_idx = utils.toindex(res(i * 4 + 2))
                global_uncached_ids = utils.toindex(res(i * 4 + 3))
                ret.append(SubgraphCachedFrame(self._frame, self._cache, self._ctx,
                                               cached_out_idx, cache_idx,
                                               uncached_out_idx, global_uncached_ids))
                self._num_access += len(id)
                self._num_hits += len(cached_out_idx)
        else:
            res = _CAPI_DGLCacheLookup(self._cached_ids.todgltensor(), dgl_ids[0])
            cached_out_idx = utils.toindex(res(0))
            uncached_out_idx = utils.toindex(res(1))
            cache_idx = utils.toindex(res(2))
            global_uncached_ids = utils.toindex(res(3))
            ret.append(SubgraphCachedFrame(self._frame, self._cache, self._ctx,
                                           cached_out_idx, cache_idx,
                                           uncached_out_idx, global_uncached_ids))
            self._num_access += len(ids[0])
            self._num_hits += len(cached_out_idx)
        return ret

class SubgraphCachedFrame:
    """The cached frame for the subgraph.

    It contains all the parts to reconstruct the frame for the subgraph.


    Parameters
    ----------
    frame : FrameRef
        The original frame.
    cache : dict of Tensor.
        The cached data
    cached_out_idx : utils.Index
        The index of the cached data in the reconstructed frame.
    cache_idx : utils.Index
        The index of the cached data in the cache.
    uncached_out_idx : utils.Index
        The index of the uncached data in the reconstructed frame.
    global_uncached_ids : utils.Index
        The index of the uncached data in the original frame.
    """
    def __init__(self, frame, cache, ctx, cached_out_idx, cache_idx,
                 uncached_out_idx, global_uncached_ids):
        self._frame = frame
        self._cache = cache
        self._ctx = ctx
        # The index where cached data should be written to.
        self._cached_out_idx = cached_out_idx.tousertensor()
        # The index where cached data should be read from the cache.
        self._cache_idx = cache_idx.tousertensor()
        # The index of uncached data. It'll be read from the global frame.
        self._global_uncached_ids = global_uncached_ids.tousertensor()
        # The index of uncached data should be written to.
        self._uncached_out_idx = uncached_out_idx.tousertensor()

    @property
    def context(self):
        """The context where the cache is stored.
        """
        return self._ctx

    def merge(self):
        """Merge all parts of data.

        Returns
        -------
        Dict of Tensors
            The reconstructed data.
        """
        ret = {}
        cache_idx = F.copy_to(self._cache_idx, ctx=self._ctx)
        cached_out_idx = F.copy_to(self._cached_out_idx, self._ctx)
        uncached_out_idx = F.copy_to(self._uncached_out_idx, self._ctx)
        for key in self._cache:
            col = self._cache[key]
            shape = (len(self._cache_idx) + len(self._global_uncached_ids),) + col.shape[1:]
            data = F.empty(shape=shape, dtype=col.dtype, ctx=self._ctx)
            # fill cached data.
            data[cached_out_idx] = F.gather_row(col, cache_idx)
            # fill uncached data
            col = self._frame[key]
            data[uncached_out_idx] = F.copy_to(F.gather_row(col, self._global_uncached_ids), self._ctx)
            ret.update({key: data})
        return ret

_init_api("dgl.frame_cache")
