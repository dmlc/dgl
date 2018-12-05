"""Columnar storage for DGLGraph."""
from __future__ import absolute_import

from collections import MutableMapping, namedtuple

import sys
import numpy as np

from . import backend as F
from .base import DGLError, dgl_warning
from .init import zero_initializer
from . import utils

class Scheme(namedtuple('Scheme', ['shape', 'dtype'])):
    """The column scheme.

    Parameters
    ----------
    shape : tuple of int
        The feature shape.
    dtype : backend-specific type object
        The feature data type.
    """
    # FIXME:
    # Python 3.5.2 is unable to pickle torch dtypes; this is a workaround.
    # I also have to create data_type_dict and reverse_data_type_dict
    # attribute just for this bug.
    # I raised an issue in PyTorch bug tracker:
    # https://github.com/pytorch/pytorch/issues/14057
    if sys.version_info.major == 3 and sys.version_info.minor == 5:
        def __reduce__(self):
            state = (self.shape, F.reverse_data_type_dict[self.dtype])
            return self._reconstruct_scheme, state


        @classmethod
        def _reconstruct_scheme(cls, shape, dtype_str):
            dtype = F.data_type_dict[dtype_str]
            return cls(shape, dtype)

def infer_scheme(tensor):
    return Scheme(tuple(F.shape(tensor)[1:]), F.dtype(tensor))

class Column(object):
    """A column is a compact store of features of multiple nodes/edges.

    Currently, we use one dense tensor to batch all the feature tensors
    together (along the first dimension).

    Parameters
    ----------
    data : Tensor
        The initial data of the column.
    scheme : Scheme, optional
        The scheme of the column. Will be inferred if not provided.
    """
    def __init__(self, data, scheme=None):
        self.data = data
        self.scheme = scheme if scheme else infer_scheme(data)

    def __len__(self):
        """The column length."""
        return F.shape(self.data)[0]

    @property
    def shape(self):
        return self.scheme.shape

    def __getitem__(self, idx):
        """Return the feature data given the index.

        Parameters
        ----------
        idx : slice or utils.Index
            The index.

        Returns
        -------
        Tensor
            The feature data
        """
        if isinstance(idx, slice):
            return self.data[idx]
        else:
            user_idx = idx.tousertensor(F.context(self.data))
            return F.gather_row(self.data, user_idx)

    def __setitem__(self, idx, feats):
        """Update the feature data given the index.

        The update is performed out-placely so it can be used in autograd mode.
        For inplace write, please use ``update``.

        Parameters
        ----------
        idx : utils.Index or slice
            The index.
        feats : Tensor
            The new features.
        """
        self.update(idx, feats, inplace=False)

    def update(self, idx, feats, inplace):
        """Update the feature data given the index.

        Parameters
        ----------
        idx : utils.Index or slice
            The index.
        feats : Tensor
            The new features.
        inplace : bool
            If true, use inplace write.
        """
        feat_scheme = infer_scheme(feats)
        if feat_scheme != self.scheme:
            raise DGLError("Cannot update column of scheme %s using feature of scheme %s."
                    % (feat_scheme, self.scheme))

        if isinstance(idx, utils.Index):
            idx = idx.tousertensor(F.context(self.data))

        if inplace:
            F.scatter_row_inplace(self.data, idx, feats)
        else:
            if isinstance(idx, slice):
                # for contiguous indices pack is usually faster than scatter row
                part1 = F.narrow_row(self.data, 0, idx.start)
                part2 = feats
                part3 = F.narrow_row(self.data, idx.stop, len(self))
                self.data = F.cat([part1, part2, part3], dim=0)
            else:
                self.data = F.scatter_row(self.data, idx, feats)

    def extend(self, feats, feat_scheme=None):
        """Extend the feature data.

         Parameters
        ----------
        feats : Tensor
            The new features.
        feat_scheme : Scheme, optional
            The scheme
        """
        if feat_scheme is None:
            feat_scheme = Scheme.infer_scheme(feats)

        if feat_scheme != self.scheme:
            raise DGLError("Cannot update column of scheme %s using feature of scheme %s."
                    % (feat_scheme, self.scheme))

        feats = F.copy_to(feats, F.context(self.data))
        self.data = F.cat([self.data, feats], dim=0)

    @staticmethod
    def create(data):
        """Create a new column using the given data."""
        if isinstance(data, Column):
            return Column(data.data, data.scheme)
        else:
            return Column(data)

class Frame(MutableMapping):
    """The columnar storage for node/edge features.

    The frame is a dictionary from feature fields to feature columns.
    All columns should have the same number of rows (i.e. the same first dimension).

    Parameters
    ----------
    data : dict-like, optional
        The frame data in dictionary. If the provided data is another frame,
        this frame will NOT share columns with the given frame. So any out-place
        update on one will not reflect to the other. The inplace update will
        be seen by both. This follows the semantic of python's container.
    num_rows : int, optional [default=0]
        The number of rows in this frame. If ``data`` is provided, ``num_rows``
        will be ignored and inferred from the given data.
    """
    def __init__(self, data=None, num_rows=0):
        if data is None:
            self._columns = dict()
            self._num_rows = num_rows
        else:
            # Note that we always create a new column for the given data.
            # This avoids two frames accidentally sharing the same column.
            self._columns = {k : Column.create(v) for k, v in data.items()}
            if len(self._columns) != 0:
                self._num_rows = len(next(iter(self._columns.values())))
            else:
                self._num_rows = 0
            # sanity check
            for name, col in self._columns.items():
                if len(col) != self._num_rows:
                    raise DGLError('Expected all columns to have same # rows (%d), '
                                   'got %d on %r.' % (self._num_rows, len(col), name))
        # Initializer for empty values. Initializer is a callable.
        # If is none, then a warning will be raised
        # in the first call and zero initializer will be used later.
        self._initializers = {}  # per-column initializers
        self._default_initializer = None

    def _warn_and_set_initializer(self):
        dgl_warning('Initializer is not set. Use zero initializer instead.'
                    ' To suppress this warning, use `set_initializer` to'
                    ' explicitly specify which initializer to use.')
        self._default_initializer = zero_initializer

    def get_initializer(self, column=None):
        """Get the initializer for empty values for the given column.

        Parameters
        ----------
        column : str
            The column

        Returns
        -------
        callable
            The initializer
        """
        return self._initializers.get(column, self._default_initializer)

    def set_initializer(self, initializer, column=None):
        """Set the initializer for empty values, for a given column or all future
        columns.

        Initializer is a callable that returns a tensor given the shape and data type.

        Parameters
        ----------
        initializer : callable
            The initializer.
        column : str, optional
            The column name
        """
        if column is None:
            self._default_initializer = initializer
        else:
            self._initializers[column] = initializer

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

    def add_column(self, name, scheme, ctx):
        """Add a new column to the frame.

        The frame will be initialized by the initializer.

        Parameters
        ----------
        name : str
            The column name.
        scheme : Scheme
            The column scheme.
        ctx : DGLContext
            The column context.
        """
        if name in self:
            dgl_warning('Column "%s" already exists. Ignore adding this column again.' % name)
            return
        if self.get_initializer(name) is None:
            self._warn_and_set_initializer()
        init_data = self.get_initializer(name)(
                (self.num_rows,) + scheme.shape, scheme.dtype,
                ctx, slice(0, self.num_rows))
        self._columns[name] = Column(init_data, scheme)

    def add_rows(self, num_rows):
        """Add blank rows to this frame.

        For existing fields, the rows will be extended according to their
        initializers.

        Parameters
        ----------
        num_rows : int
            The number of new rows
        """
        feat_placeholders = {}
        for key, col in self._columns.items():
            scheme = col.scheme
            ctx = F.context(col.data)
            if self.get_initializer(key) is None:
                self._warn_and_set_initializer()
            new_data = self.get_initializer(key)(
                    (num_rows,) + scheme.shape, scheme.dtype,
                    ctx, slice(self._num_rows, self._num_rows + num_rows))
            feat_placeholders[key] = new_data
        self._append(Frame(feat_placeholders))
        self._num_rows += num_rows

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
        if len(col) != self.num_rows:
            raise DGLError('Expected data to have %d rows, got %d.' %
                           (self.num_rows, len(col)))
        self._columns[name] = col

    def _append(self, other):
        # NOTE: `other` can be empty.
        if self.num_rows == 0:
            # if no rows in current frame; append is equivalent to
            # directly updating columns.
            self._columns = {key: Column.create(data) for key, data in other.items()}
        else:
            for key, col in other.items():
                if key not in self._columns:
                    # the column does not exist; init a new column
                    self.add_column(key, col.scheme, F.context(col.data))
                self._columns[key].extend(col.data, col.scheme)

    def append(self, other):
        """Append another frame's data into this frame.

        If the current frame is empty, it will just use the columns of the
        given frame. Otherwise, the given data should contain all the
        column keys of this frame.

        Parameters
        ----------
        other : Frame or dict-like
            The frame data to be appended.
        """
        if not isinstance(other, Frame):
            other = Frame(other)
        self._append(other)
        self._num_rows += other.num_rows

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
    """Reference object to a frame on a subset of rows.

    Parameters
    ----------
    frame : Frame, optional
        The underlying frame. If not given, the reference will point to a
        new empty frame.
    index : iterable, slice, or int, optional
        The rows that are referenced in the underlying frame. If not given,
        the whole frame is referenced. The index should be distinct (no
        duplication is allowed).

        Note that if a slice is given, the step must be None.
    """
    def __init__(self, frame=None, index=None):
        self._frame = frame if frame is not None else Frame()
        if index is None:
            # _index_data can be either a slice or an iterable
            self._index_data = slice(0, self._frame.num_rows)
        else:
            # TODO(minjie): check no duplication
            self._index_data = index
        self._index = None
        self._index_or_slice = None

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
            # NOTE: we always assume that slice.step is None
            return self._index_data.stop - self._index_data.start
        else:
            return len(self._index_data)

    def set_initializer(self, initializer, column=None):
        """Set the initializer for empty values.

        Initializer is a callable that returns a tensor given the shape and data type.

        Parameters
        ----------
        initializer : callable
            The initializer.
        column : str, optional
            The column name
        """
        self._frame.set_initializer(initializer, column=column)

    def get_initializer(self, column=None):
        """Get the initializer for empty values for the given column.

        Parameters
        ----------
        column : str
            The column

        Returns
        -------
        callable
            The initializer
        """
        return self._frame.get_initializer(column)

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
                        F.arange(self._index_data.start,
                                 self._index_data.stop))
            else:
                self._index = utils.toindex(self._index_data)
        return self._index

    def index_or_slice(self):
        """Returns the index object or the slice

        Returns
        -------
        utils.Index or slice
            The index or slice
        """
        if self._index_or_slice is None:
            if self.is_contiguous():
                self._index_or_slice = self._index_data
            else:
                self._index_or_slice = utils.toindex(self._index_data)
        return self._index_or_slice

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
        If the provided key is an index or a slice, the corresponding rows will be selected.
        The returned rows are saved in a lazy dictionary so only the real selection happens
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
        key : str or utils.Index or slice
            The key.

        Returns
        -------
        Tensor or lazy dict or tensors
            Depends on whether it is a column selection or row selection.
        """
        if isinstance(key, str):
            return self.select_column(key)
        elif isinstance(key, slice) and key == slice(0, self.num_rows):
            # shortcut for selecting all the rows
            return self
        elif isinstance(key, utils.Index) and key.is_slice(0, self.num_rows):
            # shortcut for selecting all the rows
            return self
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
            return col[self.index_or_slice()]

    def select_rows(self, query):
        """Return the rows given the query.

        Parameters
        ----------
        query : utils.Index or slice
            The rows to be selected.

        Returns
        -------
        utils.LazyDict
            The lazy dictionary from str to the selected data.
        """
        rows = self._getrows(query)
        return utils.LazyDict(lambda key: self._frame[key][rows], keys=self.keys())

    def __setitem__(self, key, val):
        self.set_item_inplace(key, val, inplace=False)

    def set_item_inplace(self, key, val, inplace):
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
        inplace: bool
            If True, update will be done in place
        """
        if isinstance(key, str):
            self.update_column(key, val, inplace=inplace)
        elif isinstance(key, slice) and key == slice(0, self.num_rows):
            # shortcut for updating all the rows
            return self.update(val)
        elif isinstance(key, utils.Index) and key.is_slice(0, self.num_rows):
            # shortcut for selecting all the rows
            return self.update(val)
        else:
            self.update_rows(key, val, inplace=inplace)

    def update_column(self, name, data, inplace):
        """Update the column.

        If this frameref spans the whole column of the underlying frame, this is
        equivalent to update the column of the frame.

        If this frameref only points to part of the rows, then update the column
        here will correspond to update part of the column in the frame. Raise error
        if the given column name does not exist.

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
                ctx = F.context(data)
                self._frame.add_column(name, infer_scheme(data), ctx)
            fcol = self._frame[name]
            fcol.update(self.index_or_slice(), data, inplace)

    def add_rows(self, num_rows):
        """Add blank rows to the underlying frame.

        For existing fields, the rows will be extended according to their
        initializers.

        Note: only available for FrameRef that spans the whole column.  The row
        span will extend to new rows.  Other FrameRefs referencing the same
        frame will not be affected.

        Parameters
        ----------
        num_rows : int
            Number of rows to add
        """
        if not self.is_span_whole_column():
            raise RuntimeError('FrameRef not spanning whole column.')
        self._frame.add_rows(num_rows)
        if self.is_contiguous():
            self._index_data = slice(0, self._index_data.stop + num_rows)
        else:
            self._index_data.extend(range(self.num_rows, self.num_rows + num_rows))

    def update_rows(self, query, data, inplace):
        """Update the rows.

        If the provided data has new column, it will be added to the frame.

        See Also
        --------
        ``update_column``

        Parameters
        ----------
        query : utils.Index or slice
            The rows to be updated.
        data : dict-like
            The row data.
        inplace : bool
            True if the update is performed inplacely.
        """
        rows = self._getrows(query)
        for key, col in data.items():
            if key not in self:
                # add new column
                tmpref = FrameRef(self._frame, rows)
                tmpref.update_column(key, col, inplace)
            else:
                self._frame[key].update(rows, col, inplace)

    def __delitem__(self, key):
        """Delete data in the frame.

        If the provided key is a string, the corresponding column will be deleted.
        If the provided key is an index object or a slice, the corresponding rows will
        be deleted.

        Please note that "deleted" rows are not really deleted, but simply removed
        in the reference. As a result, if two FrameRefs point to the same Frame, deleting
        from one ref will not relect on the other. However, deleting columns is real.

        Parameters
        ----------
        key : str or utils.Index
            The key.
        """
        if isinstance(key, str):
            del self._frame[key]
        else:
            self.delete_rows(key)

    def delete_rows(self, query):
        """Delete rows.

        Please note that "deleted" rows are not really deleted, but simply removed
        in the reference. As a result, if two FrameRefs point to the same Frame, deleting
        from one ref will not relect on the other. By contrast, deleting columns is real.

        Parameters
        ----------
        query : utils.Index or slice
            The rows to be deleted.
        """
        if isinstance(query, slice):
            query = range(query.start, query.stop)
        else:
            query = query.tonumpy()

        if isinstance(self._index_data, slice):
            self._index_data = range(self._index_data.start, self._index_data.stop)
        self._index_data = list(np.delete(self._index_data, query))
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
            if self._index_data.stop == old_nrows:
                new_idx = slice(self._index_data.start, self._frame.num_rows)
            else:
                new_idx = list(range(self._index_data.start, self._index_data.stop))
                new_idx.extend(range(old_nrows, self._frame.num_rows))
            self._index_data = new_idx
        self._clear_cache()

    def clear(self):
        """Clear the frame."""
        self._frame.clear()
        self._index_data = slice(0, 0)
        self._clear_cache()

    def is_contiguous(self):
        """Return whether this refers to a contiguous range of rows."""
        # NOTE: this check could have false negatives
        # NOTE: we always assume that slice.step is None
        return isinstance(self._index_data, slice)

    def is_span_whole_column(self):
        """Return whether this refers to all the rows."""
        return self.is_contiguous() and self.num_rows == self._frame.num_rows

    def _getrows(self, query):
        """Internal function to convert from the local row ids to the row ids of the frame."""
        if self.is_contiguous():
            start = self._index_data.start
            if start == 0:
                # shortcut for identical mapping
                return query
            elif isinstance(query, slice):
                return slice(query.start + start, query.stop + start)
            else:
                query = query.tousertensor()
                return utils.toindex(query + start)
        else:
            idxtensor = self.index().tousertensor()
            query = query.tousertensor()
            return utils.toindex(F.gather_row(idxtensor, query))

    def _clear_cache(self):
        """Internal function to clear the cached object."""
        self._index = None
        self._index_or_slice = None

def frame_like(other, num_rows):
    """Create a new frame that has the same scheme as the given one.

    Parameters
    ----------
    other : Frame
        The given frame.
    num_rows : int
        The number of rows of the new one.

    Returns
    -------
    Frame
        The new frame.
    """
    # TODO(minjie): scheme is not inherited at the moment. Fix this
    #   when moving per-col initializer to column scheme.
    newf = Frame(num_rows=num_rows)
    # set global initializr
    if other.get_initializer() is None:
        other._warn_and_set_initializer()
    newf._default_initializer = other._default_initializer
    # set per-col initializer
    # TODO(minjie): hack; cannot rely on keys as the _initializers
    #   now supports non-exist columns.
    newf._initializers = other._initializers
    return newf
