"""Columnar storage for graph attributes."""
from __future__ import absolute_import

import copy
from collections import Mapping
import numpy as np

import dgl.backend as F
from dgl.backend import Tensor
from dgl.utils import ReadOnlyDict

class Frame(Mapping):
    """Columnar storage for node/edge features on the graph.

    Parameters
    ----------
    data : dict
        The frame data.
    index : list
        The index used to query the data.
    """
    def __init__(self, data=None, index=None):
        if data is None:
            self._columns = dict()
            self._num_rows = 0
            self._index = slice(0, 0)
        else:
            self._columns = data
            if index is None:
                self._num_rows = F.shape(list(data.values())[0])[0]
                # sanity check
                for k, v in data.items():
                    assert F.shape(v)[0] == self._num_rows
                self._index = slice(0, self._num_rows)
            else:
                self._num_rows = len(index)
                self._index = index
        # cached index structure for faster access
        self._index_tensor = None

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
            return self.get_column(key)
        else:
            return self.select_rows(key)

    def __setitem__(self, key, val):
        if isinstance(key, str):
            # TODO
            self.add_column(key, val)
        else:
            self.update_rows(key, val)

    def __delitem__(self, key):
        if isinstance(key, str):
            del self._columns[key]
            if len(self._columns) == 0:
                self.clear()
        else:
            self.delete_rows(key)

    def pop(self, key):
        col = self[key]
        del self[key]
        return col

    def get_column(self, name):
        if self._is_span_whole_column():
            return self._columns[name]
        else:
            return F.gather_row(self._columns[name], self.index_tensor)

    def add_column(self, name, col):
        if self.num_columns == 0:
            self._num_rows = F.shape(col)[0]
            self._columns[name] = col
        else:
            assert F.shape(col)[0] == self.num_rows
            if self._is_span_whole_column():
                # simple column replace
                self._columns[name] = col
            elif self._is_slice_index():
                F.
            else:
                storage_num_rows = F.shape(list(self._columns.values())[0])[0]
                feat_shape = F.shape(col)[1:]
                # TODO(minjie): context
                newcol = F.zeros((storage_num_rows,) + feat_shape)

                self._columns[name] = newcol
                # TODO
                self.update_rows(self._index, {name : col})

    def append(self, other):
        # TODO
        if isinstance(other, Frame):
            other = other.clone()
        else:
            other = Frame(data=other)
        if len(self._columns) == 0:
            self._columns = other._columns
            self._num_rows = other._num_rows
            self._index = other._index
        else:
            assert self.schemes == other.schemes
            self._make_contiguous()
            other._make_contiguous()
            self._columns = {key : F.pack([self[key], other[key]]) for key in self._columns}
            self._num_rows += other._num_rows
        self._clear_cache()

    def clear(self):
        self._columns = {}
        self._num_rows = 0
        self._index = slice(0, 0)
        self._clear_cache()

    def select_rows(self, query):
        """Select the rows and returns a sub-frame."""
        rowids = self._getrowid(query)
        return ReadOnlyDict(Frame(self._columns, rowids))

    def update_rows(self, query, other):
        rowids = self._getrowid(query)
        if not isinstance(other, Frame):
            other = Frame(data=other)
        for key in other.schemes:
            assert key in self._columns
            self._columns[key] = F.scatter_row(self[key], rowids, other[key])

    def delete_rows(self, query):
        # TODO: index is None
        rowids = self._getrowid(query)
        arr = np.array(self._index, dtype=np.int32)
        self._index = list(np.delete(arr, rowids))
        self._clear_cache()

    def clone(self):
        """Return a new frame that shares the internal storage."""
        # The columns dictionary is reference copy.
        cols = self._columns
        idx = copy.copy(self._index)
        return Frame(cols, idx)

    def __iter__(self):
        for key in self.schemes:
            yield key

    def __len__(self):
        return self.num_columns

    @property
    def index_tensor(self):
        if self._index_tensor is None:
            if self._index is None:
                self._index_tensor = F.astype(F.arange(self.num_rows), dtype=F.int64)
            else:
                self._index_tensor = F.tensor(self._index, dtype=F.int64)
        return self._index_tensor

    def _is_span_whole_column(self):
        # TODO
        return isinstance(self._index, slice) and self

    def _getrowid(self, query):
        if self._index is None:
            # shortcut for identical mapping
            return query
        else:
            return F.gather_row(self.index_tensor, query)

    def _clear_cache(self):
        self._index_tensor = None

    def _make_contiguous(self):
        """Make the underlying storage contiguous."""
        if self._index is None:
            return
        newcols = {}
        for key, col in self._columns.items():
            newcols[key] = F.gather_row(col, self._index)
        self._columns = newcols
        self._index = None
        self._clear_cache()
