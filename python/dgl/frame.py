"""Columnar storage for DGLGraph."""
from __future__ import absolute_import

from collections import MutableMapping
import numpy as np

from . import backend as F
from .backend import Tensor
from .base import DGLError
from . import utils

class Scheme(object):
    """The column scheme.

    Parameters
    ----------
    shape : tuple of int
        The feature shape.
    dtype : TVMType
        The feature data type.
    """
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def __repr__(self):
        return 'shape=%s dtype=%s' % (repr(self.shape), repr(self.dtype))

    def __eq__(self, other):
        return self.shape == other.shape and self.dtype == other.dtype

    def __ne__(self, other):
        return not self.__eq__(other)

    @staticmethod
    def infer_scheme(tensor):
        """Infer the scheme of the given tensor."""
        return Scheme(F.shape(tensor)[1:], F.get_tvmtype(tensor))

class Column(object):
    """A column is a compact store of features of multiple nodes/edges.

    Currently, we use one dense tensor to batch all the feature tensors
    together.

    Parameters
    ----------
    data : Tensor
        The initial data of the column.
    """
    def __init__(self, data):
        self.data = data
        self.scheme = Scheme.infer_scheme(data)

    def __len__(self):
        """The column length."""
        return F.shape(data)[0]

    def __getitem__(idx):
        """Return the feature data given the index.

        Parameters
        ----------
        idx : utils.Index
            The index.

        Returns
        -------
        Tensor
            The feature data
        """
        user_idx = idx.tousertensor(F.get_context(self.data))
        return F.gather_row(self.data, user_idx)

    def __setitem__(idx, feats):
        """Update the feature data given the index.

        The update is performed out-placely so it can be used in autograd mode.
        For inplace write, please use ``update``.

        Parameters
        ----------
        idx : utils.Index
            The index.
        feats : Tensor
            The new features.
        """
        self.update(idx, feats, inplace=False)

    def update(self, idx, feats, inplace):
        """Update the feature data given the index.

        Parameters
        ----------
        idx : utils.Index
            The index.
        feats : Tensor
            The new features.
        inplace : bool
            If true, use inplace write.
        """
        feat_scheme = Scheme.infer_scheme(feats)
        if feat_scheme != self.scheme:
            raise DGLError("Cannot update column of scheme %s using feature of scheme %s."
                    % (feat_scheme, self.scheme))
        user_idx = idx.tousertensor(F.get_context(self.data))
        if inplace:
            # TODO(minjie): do not use [] operator directly
            self.data[user_idx] = feats
        else:
            self.data = F.scatter_row(self.data, user_idx, feats)

    @staticmethod
    def create(data):
        """Convert the data to column."""
        if isinstance(data, Column):
            return data
        else:
            return Column(data)

class Frame(MutableMapping):
    """The columnar storage for node/edge features.

    The frame is a dictionary from feature fields to feature columns.

    Parameters
    ----------
    data : dict-like, optional
        The frame data in dictionary.
    """
    def __init__(self, data=None):
        if data is None:
            self._columns = dict()
            self._num_rows = 0
        else:
            self._columns = {k : Column.create(v) for k, v in data.items()}
            self._num_rows = F.shape(list(data.values())[0])[0]
            # sanity check
            for name, col in self._columns.items():
                if len(col) != self._num_rows:
                    raise DGLError('All columns in the frame should have the same length.'
                                   ' Got %d and %d.' % (len(col), self._num_rows))

    @property
    def schemes(self):
        """Return a dictionary of column name to column schemes."""
        return {k : col.scheme for k, col in self._columns.items()}

    @property
    def num_columns(self):
        """Return the number of columns in this frame."""
        return len(self._columns)

    @property
    def num_rows(self):
        """Return the number of rows in this frame."""
        return self._num_rows

    def __contains__(self, name):
        """Return true if the given column name exists."""
        return name in self._columns

    def __getitem__(self, name):
        """Return the column of the given name.

        Parameters
        ----------
        name : str
            The column name.

        Returns
        -------
        Column
            The column.
        """
        return self._columns[name]

    def __setitem__(self, name, data):
        """Update the whole column.

        Parameters
        ----------
        name : str
            The column name.
        col : Column or data convertible to Column
            The column data.
        """
        self.update_column(name, data)

    def __delitem__(self, name):
        """Delete the whole column.
        
        Parameters
        ----------
        name : str
            The column name.
        """
        del self._columns[name]
        if len(self._columns) == 0:
            self._num_rows = 0

    def update_column(self, name, data):
        """Add or replace the column with the given name and data.

        Parameters
        ----------
        name : str
            The column name.
        data : Column or data convertible to Column
            The column data.
        """
        col = Column.create(data)
        if self.num_columns == 0:
            self._num_rows = len(col)
        elif len(col) != self._num_rows:
            raise DGLError('All columns in the frame should have the same length.'
                           ' Got %d and %d.' % (len(col), self._num_rows))
        self._columns[name] = col

    def append(self, other):
        """Append another frame's data into this frame.

        If the current frame is empty, it will just use the columns of the
        given frame. Otherwise, the given frame and the current frame should
        have the same column schemes.

        Parameters
        ----------
        other : Frame or dict-like
            The frame data to be appended.
        """
        if len(self._columns) == 0:
            for key, col in other.items():
                self._columns[key] = col
        else:
            for key, col in other.items():
                sch = self._columns[key].scheme
                other_sch = col.scheme
                if sch != other_sch:
                    raise DGLError("Cannot append column of scheme %s to column of scheme %s."
                            % (other_scheme, sch))
                self._columns[key].data = F.pack(
                        [self._columns[key].data, col.data])
                # TODO(minjie): sanity check for num_rows
                self._num_rows = len(self._columns[key])

    def clear(self):
        """Clear this frame. Remove all the columns."""
        self._columns = {}
        self._num_rows = 0

    def __iter__(self):
        """Return an iterator of columns."""
        return iter(self._columns)

    def __len__(self):
        """Return the number of columns."""
        return self.num_columns

    def keys(self):
        """Return the keys."""
        return self._columns.keys()

class FrameRef(MutableMapping):
    """Reference object to a frame.

    Parameters
    ----------
    frame : Frame, optional
        The underlying frame. If not given, the reference will point to a
        new empty frame.
    index : iterable of int, optional
        The rows that are referenced in the underlying frame. If not given,
        the whole frame is referenced. The index should be distinct (no
        duplication is allowed).
    """
    def __init__(self, frame=None, index=None):
        self._frame = frame if frame is not None else Frame()
        if index is None:
            self._index_data = slice(0, self._frame.num_rows)
        else:
            # TODO(minjie): check no duplication
            self._index_data = index
        self._index = None

    @property
    def schemes(self):
        """Return the frame schemes.
        
        Returns
        -------
        dict of str to Scheme
            The frame schemes.
        """
        return self._frame.schemes

    @property
    def num_columns(self):
        """Return the number of columns in the referred frame."""
        return self._frame.num_columns

    @property
    def num_rows(self):
        """Return the number of rows referred."""
        if isinstance(self._index_data, slice):
            return self._index_data.stop
        else:
            return len(self._index_data)

    def index(self):
        """Return the index object.

        Returns
        -------
        utils.Index
            The index.
        """
        if self._index is None:
            if self.is_contiguous():
                self._index = utils.toindex(
                        F.arange(self._index_data.stop, dtype=F.int64))
            else:
                self._index = utils.toindex(self._index_data)
        return self._index

    def __contains__(self, name):
        """Return whether the column name exists."""
        return name in self._frame

    def __iter__(self):
        """Return the iterator of the columns."""
        return iter(self._frame)

    def __len__(self):
        """Return the number of columns."""
        return self.num_columns

    def keys(self):
        """Return the keys."""
        return self._frame.keys()

    def __getitem__(self, key):
        """Get data from the frame.

        If the provided key is string, the corresponding column data will be returned.
        If the provided key is an index, the corresponding rows will be selected. The
        returned rows are saved in a lazy dictionary so only the real selection happens
        when the explicit column name is provided.
        
        Examples (using pytorch)
        ------------------------
        >>> # create a frame of two columns and five rows
        >>> f = Frame({'c1' : torch.zeros([5, 2]), 'c2' : torch.ones([5, 2])})
        >>> fr = FrameRef(f)
        >>> # select the row 1 and 2, the returned `rows` is a lazy dictionary.
        >>> rows = fr[Index([1, 2])]
        >>> rows['c1']  # only select rows for 'c1' column; 'c2' column is not sliced.
        
        Parameters
        ----------
        key : str or utils.Index
            The key.

        Returns
        -------
        Tensor or lazy dict or tensors
            Depends on whether it is a column selection or row selection.
        """
        if isinstance(key, str):
            return self.select_column(key)
        else:
            return self.select_rows(key)

    def select_column(self, name):
        """Return the column of the given name.

        If only part of the rows are referenced, the fetching the whole column will
        also slice out the referenced rows.

        Parameters
        ----------
        name : str
            The column name.

        Returns
        -------
        Tensor
            The column data.
        """
        col = self._frame[name]
        if self.is_span_whole_column():
            return col.data
        else:
            return col[self.index()]

    def select_rows(self, query):
        """Return the rows given the query.

        Parameters
        ----------
        query : utils.Index
            The rows to be selected.

        Returns
        -------
        utils.LazyDict
            The lazy dictionary from str to the selected data.
        """
        rowids = self._getrowid(query)
        return utils.LazyDict(lambda key: self._frame[key][rowids], keys=self.keys())

    def __setitem__(self, key, val):
        """Update the data in the frame.

        If the provided key is string, the corresponding column data will be updated.
        The provided value should be one tensor that have the same scheme and length
        as the column.

        If the provided key is an index, the corresponding rows will be updated. The
        value provided should be a dictionary of string to the data of each column.

        All updates are performed out-placely to be work with autograd. For inplace
        update, use ``update_column`` or ``update_rows``.

        Parameters
        ----------
        key : str or utils.Index
            The key.
        val : Tensor or dict of tensors
            The value.
        """
        if isinstance(key, str):
            self.update_column(key, val, inplace=False)
        else:
            self.update_rows(key, val, inplace=False)

    def update_column(self, name, data, inplace):
        """Update the column.

        If this frameref spans the whole column of the underlying frame, this is
        equivalent to update the column of the frame.

        If this frameref only points to part of the rows, then update the column
        here will correspond to update part of the column in the frame. If the
        given column name does not exist, a new zero column will be created first.

        Parameters
        ----------
        name : str
            The column name.
        data : Tensor
            The update data.
        inplace : bool
            True if the update is performed inplacely.
        """
        if self.is_span_whole_column():
            col = Column.create(data)
            if self.num_columns == 0:
                # the frame is empty
                self._index_data = slice(0, len(col))
                self._clear_cache()
            self._frame[name] = col
        else:
            if name not in self._frame:
                # create a zero column
                shp = F.shape(data)
                ctx = F.get_context(data)
                # TODO(minjie): directly create the zero col on the target device
                zero_col = F.zeros((self._frame.num_rows,) + shp[1:])
                zero_col = F.to_context(zero_col, ctx)
                self._frame[name] = zero_col
            fcol = self._frame[name]
            fcol.update(self.index(), data, inplace)

    def update_rows(self, query, data, inplace):
        """Update the rows.

        If the provided data has new column, it will be added to the frame.

        See Also
        --------
        ``update_column``

        Parameters
        ----------
        query : utils.Index
            The rows to be updated.
        data : dict-like
            The row data.
        inplace : bool
            True if the update is performed inplacely.
        """
        rowids = self._getrowid(query)
        for key, col in data.items():
            if key not in self:
                # add new column
                tmpref = FrameRef(self._frame, rowids)
                tmpref.update_column(key, col, inplace)
            else:
                self._frame[key].update(rowids, col, inplace)

    def __delitem__(self, key):
        """Delete data in the frame.

        If the provided key is a string, the corresponding column will be deleted.
        If the provided key is an index object, the corresponding rows will be deleted.

        Please note that "deleted" rows are not really deleted, but simply removed
        in the reference. As a result, if two FrameRefs point to the same Frame, deleting
        from one ref will not relect on the other. By contrast, deleting columns is real.

        Parameters
        ----------
        key : str or utils.Index
            The key.
        """
        if isinstance(key, str):
            del self._frame[key]
            if len(self._frame) == 0:
                self.clear()
        else:
            self.delete_rows(key)

    def delete_rows(self, query):
        """Delete rows.

        Please note that "deleted" rows are not really deleted, but simply removed
        in the reference. As a result, if two FrameRefs point to the same Frame, deleting
        from one ref will not relect on the other. By contrast, deleting columns is real.

        Parameters
        ----------
        query : utils.Index
            The rows to be deleted.
        """
        query = query.tolist()
        if isinstance(self._index_data, slice):
            self._index_data = list(range(self._index_data.start, self._index_data.stop))
        arr = np.array(self._index_data, dtype=np.int32)
        self._index_data = list(np.delete(arr, query))
        self._clear_cache()

    def append(self, other):
        """Append another frame into this one.

        Parameters
        ----------
        other : dict of str to tensor
            The data to be appended.
        """
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
        """Clear the frame."""
        self._frame.clear()
        self._index_data = slice(0, 0)
        self._clear_cache()

    def is_contiguous(self):
        """Return whether this refers to a contiguous range of rows."""
        # NOTE: this check could have false negative
        return isinstance(self._index_data, slice)

    def is_span_whole_column(self):
        """Return whether this refers to all the rows."""
        return self.is_contiguous() and self.num_rows == self._frame.num_rows

    def _getrowid(self, query):
        """Internal function to convert from the local row ids to the row ids of the frame."""
        if self.is_contiguous():
            # shortcut for identical mapping
            return query
        else:
            idxtensor = self.index().tousertensor()
            return utils.toindex(F.gather_row(idxtensor, query.tousertensor()))

    def _clear_cache(self):
        """Internal function to clear the cached object."""
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
    assert False, 'Buggy code, disabled for now.'
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
