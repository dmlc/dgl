"""Columnar storage for graph attributes."""
from __future__ import absolute_import

from collections import MutableMapping
import numpy as np

import dgl.backend as F
from dgl.backend import Tensor
import dgl.utils as utils

class Frame(MutableMapping):
    def __init__(self, data=None):
        if data is None:
            self._columns = dict()
            self._num_rows = 0
        else:
            self._columns = dict(data)
            self._num_rows = F.shape(list(data.values())[0])[0]
            for k, v in data.items():
                assert F.shape(v)[0] == self._num_rows

    @property
    def schemes(self):
        return set(self._columns.keys())

    @property
    def num_columns(self):
        return len(self._columns)

    @property
    def num_rows(self):
        return self._num_rows

    def __contains__(self, key):
        return key in self._columns

    def __getitem__(self, key):
        # get column
        return self._columns[key]

    def __setitem__(self, key, val):
        # set column
        self.add_column(key, val)

    def __delitem__(self, key):
        # delete column
        del self._columns[key]
        if len(self._columns) == 0:
            self._num_rows = 0

    def add_column(self, name, col):
        if self.num_columns == 0:
            self._num_rows = F.shape(col)[0]
        else:
            assert F.shape(col)[0] == self._num_rows
        self._columns[name] = col

    def append(self, other):
        if len(self._columns) == 0:
            for key, col in other.items():
                self._columns[key] = col
        else:
            for key, col in other.items():
                self._columns[key] = F.pack([self[key], col])
        # TODO(minjie): sanity check for num_rows
        if len(self._columns) != 0:
            self._num_rows = F.shape(list(self._columns.values())[0])[0]

    def clear(self):
        self._columns = {}
        self._num_rows = 0

    def __iter__(self):
        return iter(self._columns)

    def __len__(self):
        return self.num_columns

class FrameRef(MutableMapping):
    """Frame reference

    Parameters
    ----------
    frame : dgl.frame.Frame
        The underlying frame.
    index : iterable of int
        The rows that are referenced in the underlying frame.
    """
    def __init__(self, frame=None, index=None):
        self._frame = frame if frame is not None else Frame()
        if index is None:
            self._index_data = slice(0, self._frame.num_rows)
        else:
            # check no duplication
            assert len(index) == len(np.unique(index))
            self._index_data = index
        self._index = None

    @property
    def schemes(self):
        return self._frame.schemes

    @property
    def num_columns(self):
        return self._frame.num_columns

    @property
    def num_rows(self):
        if isinstance(self._index_data, slice):
            return self._index_data.stop
        else:
            return len(self._index_data)

    def __contains__(self, key):
        return key in self._frame

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.get_column(key)
        else:
            return self.select_rows(key)

    def select_rows(self, query):
        rowids = self._getrowid(query)
        def _lazy_select(key):
            idx = rowids.totensor(F.get_context(self._frame[key]))
            return F.gather_row(self._frame[key], idx)
        return utils.LazyDict(_lazy_select, keys=self.schemes)

    def get_column(self, name):
        col = self._frame[name]
        if self.is_span_whole_column():
            return col
        else:
            idx = self.index().totensor(F.get_context(col))
            return F.gather_row(col, idx)

    def __setitem__(self, key, val):
        if isinstance(key, str):
            self.add_column(key, val)
        else:
            self.update_rows(key, val)

    def add_column(self, name, col, inplace=False):
        shp = F.shape(col)
        if self.is_span_whole_column():
            if self.num_columns == 0:
                self._index_data = slice(0, shp[0])
                self._clear_cache()
            assert shp[0] == self.num_rows
            self._frame[name] = col
        else:
            colctx = F.get_context(col)
            if name in self._frame:
                fcol = self._frame[name]
            else:
                fcol = F.zeros((self._frame.num_rows,) + shp[1:])
                fcol = F.to_context(fcol, colctx)
            idx = self.index().totensor(colctx)
            if inplace:
                self._frame[name] = fcol
                self._frame[name][idx] = col
            else:
                newfcol = F.scatter_row(fcol, idx, col)
                self._frame[name] = newfcol

    def update_rows(self, query, other, inplace=False):
        rowids = self._getrowid(query)
        for key, col in other.items():
            if key not in self:
                # add new column
                tmpref = FrameRef(self._frame, rowids)
                tmpref.add_column(key, col, inplace)
            idx = rowids.totensor(F.get_context(self._frame[key]))
            if inplace:
                self._frame[key][idx] = col
            else:
                self._frame[key] = F.scatter_row(self._frame[key], idx, col)

    def __delitem__(self, key):
        if isinstance(key, str):
            del self._frame[key]
            if len(self._frame) == 0:
                self.clear()
        else:
            self.delete_rows(key)

    def delete_rows(self, query):
        query = F.asnumpy(query)
        if isinstance(self._index_data, slice):
            self._index_data = list(range(self._index_data.start, self._index_data.stop))
        arr = np.array(self._index_data, dtype=np.int32)
        self._index_data = list(np.delete(arr, query))
        self._clear_cache()

    def append(self, other):
        span_whole = self.is_span_whole_column()
        contiguous = self.is_contiguous()
        old_nrows = self._frame.num_rows
        self._frame.append(other)
        # update index
        if span_whole:
            self._index_data = slice(0, self._frame.num_rows)
        elif contiguous:
            new_idx = list(range(self._index_data.start, self._index_data.stop))
            new_idx += list(range(old_nrows, self._frame.num_rows))
            self._index_data = new_idx
        self._clear_cache()

    def clear(self):
        self._frame.clear()
        self._index_data = slice(0, 0)
        self._clear_cache()

    def __iter__(self):
        return iter(self._frame)

    def __len__(self):
        return self.num_columns

    def is_contiguous(self):
        # NOTE: this check could have false negative
        return isinstance(self._index_data, slice)

    def is_span_whole_column(self):
        return self.is_contiguous() and self.num_rows == self._frame.num_rows

    def _getrowid(self, query):
        if self.is_contiguous():
            # shortcut for identical mapping
            return query
        else:
            idxtensor = self.index().totensor()
            return utils.toindex(F.gather_row(idxtensor, query.totensor()))

    def index(self):
        if self._index is None:
            if self.is_contiguous():
                self._index = utils.toindex(
                        F.arange(self._index_data.stop, dtype=F.int64))
            else:
                self._index = utils.toindex(self._index_data)
        return self._index

    def _clear_cache(self):
        self._index_tensor = None

def merge_frames(frames, indices, max_index, reduce_func):
    """Merge a list of frames.

    The result frame contains `max_index` number of rows. For each frame in
    the given list, its row is merged as follows:

        merged[indices[i][row]] += frames[i][row]

    Parameters
    ----------
    frames : iterator of dgl.frame.FrameRef
        A list of frames to be merged.
    indices : iterator of dgl.utils.Index
        The indices of the frame rows.
    reduce_func : str
        The reduce function (only 'sum' is supported currently)

    Returns
    -------
    merged : FrameRef
        The merged frame.
    """
    assert reduce_func == 'sum'
    assert len(frames) > 0
    schemes = frames[0].schemes
    # create an adj to merge
    # row index is equal to the concatenation of all the indices.
    row = sum([idx.tolist() for idx in indices], [])
    col = list(range(len(row)))
    n = max_index
    m = len(row)
    row = F.unsqueeze(F.tensor(row, dtype=F.int64), 0)
    col = F.unsqueeze(F.tensor(col, dtype=F.int64), 0)
    idx = F.pack([row, col])
    dat = F.ones((m,))
    adjmat = F.sparse_tensor(idx, dat, [n, m])
    ctx_adjmat = utils.CtxCachedObject(lambda ctx: F.to_context(adjmat, ctx))
    merged = {}
    for key in schemes:
        # the rhs of the spmv is the concatenation of all the frame columns
        feats = F.pack([fr[key] for fr in frames])
        merged_feats = F.spmm(ctx_adjmat.get(F.get_context(feats)), feats)
        merged[key] = merged_feats
    merged = FrameRef(Frame(merged))
    return merged
