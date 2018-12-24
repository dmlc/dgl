"""Cache for frames in DGLGraph."""
import math
import numpy as np

from . import backend as F
from . import utils
from .frame import Frame, FrameRef
from .graph_index import search_nids
from ._ffi.function import _init_api

class FrameRowCache:
    def __init__(self, frame, ids, ctx):
        self._cached_ids = ids
        self._frame = frame
        ids = ids.tousertensor()
        cols = {}
        for key in frame:
            col = frame[key]
            cols.update({key: F.copy_to(col[ids], ctx)})
        self._cache = FrameRef(Frame(cols))
        self._ctx = ctx
        self._opt_lookup = {
            2: _CAPI_DGLCacheLookup2,
            4: _CAPI_DGLCacheLookup4,
            8: _CAPI_DGLCacheLookup8,
            16: _CAPI_DGLCacheLookup16,
            32: _CAPI_DGLCacheLookup32,
        }

    @property
    def context(self):
        return self._ctx

    def cache_lookup(self, ids):
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
                ret.append(SubgraphFrameCache(self._frame, self._cache, self._ctx,
                                              cached_out_idx, cache_idx,
                                              uncached_out_idx, global_uncached_ids))
        else:
            res = _CAPI_DGLCacheLookup(self._cached_ids.todgltensor(), dgl_ids[0])
            cached_out_idx = utils.toindex(res(0))
            uncached_out_idx = utils.toindex(res(1))
            cache_idx = utils.toindex(res(2))
            global_uncached_ids = utils.toindex(res(3))
            ret.append(SubgraphFrameCache(self._frame, self._cache, self._ctx,
                                          cached_out_idx, cache_idx,
                                          uncached_out_idx, global_uncached_ids))
        return ret

class SubgraphFrameCache:
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
        return self._ctx

    def merge(self):
        ret = {}
        for key in self._cache:
            col = self._cache[key]
            shape = (len(self._cache_idx) + len(self._global_uncached_ids),) + col.shape[1:]
            data = F.empty(shape=shape, dtype=col.dtype, ctx=self._ctx)
            # fill cached data.
            data[self._cached_out_idx] = col[self._cache_idx]
            # fill uncached data
            col = self._frame[key]
            data[self._uncached_out_idx] = F.copy_to(col[self._global_uncached_ids], self._ctx)
            ret.update({key: data})
        return ret

_init_api("dgl.frame_cache")
