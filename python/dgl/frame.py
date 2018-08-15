"""Columnar storage for graph attributes."""
from __future__ import absolute_import

from collections import MutableMapping
import numpy as np

import dgl.backend as F
from dgl.backend import Tensor
from dgl.utils import LazyDict

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
        self._num_rows = F.shape(list(self._columns.values())[0])[0]

    def clear(self):
        self._columns = {}
        self._num_rows = 0

    def __iter__(self):
        return iter(self._columns)

    def __len__(self):
        return self.num_columns

class FrameRef(MutableMapping):
    def __init__(self, frame=None, index=None):
        self._frame = frame if frame is not None else Frame()
        if index is None:
            self._index = slice(0, self._frame.num_rows)
        else:
            # check no duplicate index
            assert len(index) == len(np.unique(index))
            self._index = index
        self._index_tensor = None

    @property
    def schemes(self):
        return self._frame.schemes

    @property
    def num_columns(self):
        return self._frame.num_columns

    @property
    def num_rows(self):
        if isinstance(self._index, slice):
            return self._index.stop
        else:
            return len(self._index)

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
            return F.gather_row(self._frame[key], rowids)
        return LazyDict(_lazy_select, keys=self.schemes)

    def get_column(self, name):
        col = self._frame[name]
        if self.is_span_whole_column():
            return col
        else:
            return F.gather_row(col, self.index_tensor())

    def __setitem__(self, key, val):
        if isinstance(key, str):
            self.add_column(key, val)
        else:
            self.update_rows(key, val)

    def add_column(self, name, col):
        shp = F.shape(col)
        if self.is_span_whole_column():
            if self.num_columns == 0:
                self._index = slice(0, shp[0])
                self._clear_cache()
            assert shp[0] == self.num_rows
            self._frame[name] = col
        else:
            if name in self._frame:
                fcol = self._frame[name]
            else:
                fcol = F.zeros((self._frame.num_rows,) + shp[1:])
            newfcol = F.scatter_row(fcol, self.index_tensor(), col)
            self._frame[name] = newfcol

    def update_rows(self, query, other):
        rowids = self._getrowid(query)
        for key, col in other.items():
            self._frame[key] = F.scatter_row(self._frame[key], rowids, col)

    def __delitem__(self, key):
        if isinstance(key, str):
            del self._frame[key]
            if len(self._frame) == 0:
                self.clear()
        else:
            self.delete_rows(key)

    def delete_rows(self, query):
        query = F.asnumpy(query)
        if isinstance(self._index, slice):
            self._index = list(range(self._index.start, self._index.stop))
        arr = np.array(self._index, dtype=np.int32)
        self._index = list(np.delete(arr, query))
        self._clear_cache()

    def append(self, other):
        span_whole = self.is_span_whole_column()
        contiguous = self.is_contiguous()
        old_nrows = self._frame.num_rows
        self._frame.append(other)
        # update index
        if span_whole:
            self._index = slice(0, self._frame.num_rows)
        else:
            new_idx = list(range(self._index.start, self._index.stop))
            new_idx += list(range(old_nrows, self._frame.num_rows))
            self._index = new_idx
        self._clear_cache()

    def clear(self):
        self._frame.clear()
        self._index = slice(0, 0)
        self._clear_cache()

    def __iter__(self):
        return iter(self._frame)

    def __len__(self):
        return self.num_columns

    def is_contiguous(self):
        # NOTE: this check could have false negative
        return isinstance(self._index, slice)

    def is_span_whole_column(self):
        return self.is_contiguous() and self.num_rows == self._frame.num_rows

    def _getrowid(self, query):
        if isinstance(self._index, slice):
            # shortcut for identical mapping
            return query
        else:
            return F.gather_row(self.index_tensor(), query)

    def index_tensor(self):
        # TODO(minjie): context
        if self._index_tensor is None:
            if self.is_contiguous():
                self._index_tensor = F.arange(self._index.stop, dtype=F.int64)
            else:
                self._index_tensor = F.tensor(self._index, dtype=F.int64)
        return self._index_tensor

    def _clear_cache(self):
        self._index_tensor = None
