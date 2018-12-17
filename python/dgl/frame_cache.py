"""Cache for frames in DGLGraph."""
import numpy as np
import mxnet as mx

from . import backend as F
from . import utils
from .frame import Frame, FrameRef
from .graph_index import search_nids

class FrameRowCache:
    def __init__(self, frame, ids, ctx):
        self._cached_ids = ids
        ids = ids.tousertensor()
        cols = {}
        for key in frame:
            col = frame[key]
            cols.update({key: F.copy_to(col[ids], ctx)})
        self._cache = FrameRef(Frame(cols))
        self._ctx = ctx

    @property
    def context(self):
        return self._ctx

    def cache_lookup(self, ids):
        ids = utils.toindex(ids)
        lids = search_nids(self._cached_ids, ids)
        lids = lids.tonumpy()
        ids = ids.tonumpy()
        cached_out_idx = np.nonzero(lids != -1)[0]
        cache_idx = lids[cached_out_idx]
        uncached_out_idx = np.nonzero(lids == -1)[0]
        global_uncached_ids = ids[uncached_out_idx]

        ret = {}
        for key in self._cache:
            col = self._cache[key]
            shape = (len(ids),) + col.shape[1:]
            data = mx.nd.empty(shape=shape, dtype=col.dtype, ctx=self._ctx)
            data[cached_out_idx] = col[cache_idx]
            ret.update({key: (data, global_uncached_ids, uncached_out_idx)})
        return ret
