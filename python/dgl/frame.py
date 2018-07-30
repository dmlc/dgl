"""Columnar storage for graph attributes."""
from __future__ import absolute_import

import dgl.backend as F
from dgl.backend import Tensor
from dgl.utils import LazyDict

class Frame:
    def __init__(self, data=None):
        if data is None:
            self._columns = dict()
            self._num_rows = 0
        else:
            self._columns = data
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
        if isinstance(key, str):
            return self._columns[key]
        else:
            return self.select_rows(key)

    def __setitem__(self, key, val):
        if isinstance(key, str):
            self._columns[key] = val
        else:
            self.update_rows(key, val)

    def add_column(self, name, col):
        if self.num_columns == 0:
            self._num_rows = F.shape(col)[0]
        else:
            assert F.shape(col)[0] == self._num_rows
        self._columns[name] = col

    def append(self, other):
        if not isinstance(other, Frame):
            other = Frame(data=other)
        if len(self._columns) == 0:
            self._columns = other._columns
            self._num_rows = other._num_rows
        else:
            assert self.schemes == other.schemes
            self._columns = {key : F.pack([self[key], other[key]]) for key in self._columns}
            self._num_rows += other._num_rows

    def clear(self):
        self._columns = {}
        self._num_rows = 0

    def select_rows(self, rowids):
        def _lazy_select(key):
            return F.gather_row(self._columns[key], rowids)
        return LazyDict(_lazy_select, keys=self._columns.keys())

    def update_rows(self, rowids, other):
        if not isinstance(other, Frame):
            other = Frame(data=other)
        for key in other.schemes:
            assert key in self._columns
            self._columns[key] = F.scatter_row(self[key], rowids, other[key])

    def __iter__(self):
        for key, col in self._columns.items():
            yield key, col

    def __len__(self):
        return self.num_columns
