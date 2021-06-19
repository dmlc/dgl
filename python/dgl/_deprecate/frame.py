"""Columnar storage for DGLGraph."""
from __future__ import absolute_import

from collections import namedtuple
from collections.abc import MutableMapping

import numpy as np

from .. import backend as F
from ..base import DGLError, dgl_warning
from ..init import zero_initializer
from .. import utils

class Scheme(namedtuple('Scheme', ['shape', 'dtype'])):
    """The column scheme.

    Parameters
    ----------
    shape : tuple of int
        The feature shape.
    dtype : backend-specific type object
        The feature data type.
    """
    # Pickling torch dtypes could be problemetic; this is a workaround.
    # I also have to create data_type_dict and reverse_data_type_dict
    # attribute just for this bug.
    # I raised an issue in PyTorch bug tracker:
    # https://github.com/pytorch/pytorch/issues/14057
    def __reduce__(self):
        state = (self.shape, F.reverse_data_type_dict[self.dtype])
        return self._reconstruct_scheme, state

    @classmethod
    def _reconstruct_scheme(cls, shape, dtype_str):
        dtype = F.data_type_dict[dtype_str]
        return cls(shape, dtype)

def infer_scheme(tensor):
    """Infer column scheme from the given tensor data.

    Parameters
    ---------
    tensor : Tensor
        The tensor data.

    Returns
    -------
    Scheme
        The column scheme.
    """
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

    Attributes
    ----------
    data : Tensor
        The data of the column.
    scheme : Scheme
        The scheme of the column.
    """
    def __init__(self, data, scheme=None):
        self.data = data
        self.scheme = scheme if scheme else infer_scheme(data)

    def __len__(self):
        """The column length."""
        return F.shape(self.data)[0]

    @property
    def shape(self):
        """Return the scheme shape (feature shape) of this column."""
        return self.scheme.shape

    def __getitem__(self, idx):
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
        if idx.slice_data() is not None:
            slc = idx.slice_data()
            return F.narrow_row(self.data, slc.start, slc.stop)
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
        idx : utils.Index
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

        if inplace:
            idx = idx.tousertensor(F.context(self.data))
            F.scatter_row_inplace(self.data, idx, feats)
        elif idx.slice_data() is not None:
            # for contiguous indices narrow+concat is usually faster than scatter row
            slc = idx.slice_data()
            parts = [feats]
            if slc.start > 0:
                parts.insert(0, F.narrow_row(self.data, 0, slc.start))
            if slc.stop < len(self):
                parts.append(F.narrow_row(self.data, slc.stop, len(self)))
            self.data = F.cat(parts, dim=0)
        else:
            idx = idx.tousertensor(F.context(self.data))
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
            feat_scheme = infer_scheme(feats)

        if feat_scheme != self.scheme:
            raise DGLError("Cannot update column of scheme %s using feature of scheme %s."
                           % (feat_scheme, self.scheme))

        feats = F.copy_to(feats, F.context(self.data))
        self.data = F.cat([self.data, feats], dim=0)

    def clone(self):
        """Return a deepcopy of this column."""
        return Column(F.clone(self.data), self.scheme)

    @staticmethod
    def create(data):
        """Create a new column using the given data."""
        if isinstance(data, Column):
            return Column(data.data, data.scheme)
        else:
            return Column(data)

    def __repr__(self):
        return repr(self.data)

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
        The number of rows in this frame. If ``data`` is provided and is not empty,
        ``num_rows`` will be ignored and inferred from the given data.
    """
    def __init__(self, data=None, num_rows=0):
        if data is None:
            self._columns = dict()
            self._num_rows = num_rows
        else:
            # Note that we always create a new column for the given data.
            # This avoids two frames accidentally sharing the same column.
            self._columns = {k : Column.create(v) for k, v in data.items()}
            if isinstance(data, (Frame, FrameRef)):
                self._num_rows = data.num_rows
            elif len(self._columns) != 0:
                self._num_rows = len(next(iter(self._columns.values())))
            else:
                self._num_rows = num_rows
            # sanity check
            for name, col in self._columns.items():
                if len(col) != self._num_rows:
                    raise DGLError('Expected all columns to have same # rows (%d), '
                                   'got %d on %r.' % (self._num_rows, len(col), name))
        # Initializer for empty values. Initializer is a callable.
        # If is none, then a warning will be raised
        # in the first call and zero initializer will be used later.
        self._initializers = {}  # per-column initializers
        self._remote_init_builder = None
        self._default_initializer = None

    def _set_zero_default_initializer(self):
        """Set the default initializer to be zero initializer."""
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

    def set_remote_init_builder(self, builder):
        """Set an initializer builder to create a remote initializer for a new column to a frame.

        NOTE(minjie): This is a temporary solution. Will be replaced by KVStore in the future.

        The builder is a callable that returns an initializer. The returned initializer
        is also a callable that returns a tensor given a local tensor and tensor name.

        Parameters
        ----------
        builder : callable
            The builder to construct a remote initializer.
        """
        self._remote_init_builder = builder

    def get_remote_initializer(self, name):
        """Get a remote initializer.

        NOTE(minjie): This is a temporary solution. Will be replaced by KVStore in the future.

        Parameters
        ----------
        name : string
            The column name.
        """
        if self._remote_init_builder is None:
            return None

        if self.get_initializer(name) is None:
            self._set_zero_default_initializer()
        initializer = self.get_initializer(name)
        return self._remote_init_builder(initializer, name)

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

        # If the data is backed by a remote server, we need to move data
        # to the remote server.
        initializer = self.get_remote_initializer(name)
        if initializer is not None:
            init_data = initializer((self.num_rows,) + scheme.shape, scheme.dtype, ctx)
        else:
            if self.get_initializer(name) is None:
                self._set_zero_default_initializer()
            initializer = self.get_initializer(name)
            init_data = initializer((self.num_rows,) + scheme.shape, scheme.dtype,
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
                self._set_zero_default_initializer()
            initializer = self.get_initializer(key)
            new_data = initializer((num_rows,) + scheme.shape, scheme.dtype,
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
        # If the data is backed by a remote server, we need to move data
        # to the remote server.
        initializer = self.get_remote_initializer(name)
        if initializer is not None:
            new_data = initializer(F.shape(data), F.dtype(data), F.context(data))
            new_data[:] = data
            data = new_data
        col = Column.create(data)
        if len(col) != self.num_rows:
            raise DGLError('Expected data to have %d rows, got %d.' %
                           (self.num_rows, len(col)))
        self._columns[name] = col

    def _append(self, other):
        assert self._remote_init_builder is None, \
                "We don't support append if data in the frame is mapped from a remote server."
        # NOTE: `other` can be empty.
        if self.num_rows == 0:
            # if no rows in current frame; append is equivalent to
            # directly updating columns.
            self._columns = {key: Column.create(data) for key, data in other.items()}
        else:
            # pad columns that are not provided in the other frame with initial values
            for key, col in self.items():
                if key in other:
                    continue
                scheme = col.scheme
                ctx = F.context(col.data)
                if self.get_initializer(key) is None:
                    self._set_zero_default_initializer()
                initializer = self.get_initializer(key)
                new_data = initializer((other.num_rows,) + scheme.shape,
                                       scheme.dtype, ctx,
                                       slice(self._num_rows, self._num_rows + other.num_rows))
                other[key] = new_data
            # append other to self
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

    def values(self):
        """Return the values."""
        return self._columns.values()

    def clone(self):
        """Return a clone of this frame.

        The clone frame does not share the underlying storage with this frame,
        i.e., adding or removing columns will not be visible to each other. However,
        they still share the tensor contents so any mutable operation on the column
        tensor are visible to each other. Hence, the function does not allocate extra
        tensor memory. Use :func:`~dgl.Frame.deepclone` for cloning
        a frame that does not share any data.

        Returns
        -------
        Frame
            A cloned frame.
        """
        newframe = Frame(self._columns, self._num_rows)
        newframe._initializers = self._initializers
        newframe._remote_init_builder = self._remote_init_builder
        newframe._default_initializer = self._default_initializer
        return newframe

    def deepclone(self):
        """Return a deep clone of this frame.

        The clone frame has an copy of this frame and any modification to the clone frame
        is not visible to this frame. The function allocate new tensors and copy the contents
        from this frame. Use :func:`~dgl.Frame.clone` for cloning a frame that does not
        allocate extra tensor memory.

        Returns
        -------
        Frame
            A deep-cloned frame.
        """
        newframe = Frame({k : col.clone() for k, col in self._columns.items()}, self._num_rows)
        newframe._initializers = self._initializers
        newframe._remote_init_builder = self._remote_init_builder
        newframe._default_initializer = self._default_initializer
        return newframe

class FrameRef(MutableMapping):
    """Reference object to a frame on a subset of rows.

    Parameters
    ----------
    frame : Frame, optional
        The underlying frame. If not given, the reference will point to a
        new empty frame.
    index : utils.Index, optional
        The rows that are referenced in the underlying frame. If not given,
        the whole frame is referenced. The index should be distinct (no
        duplication is allowed).
    """
    def __init__(self, frame=None, index=None):
        self._frame = frame if frame is not None else Frame()
        # TODO(minjie): check no duplication
        assert index is None or isinstance(index, utils.Index)
        if index is None:
            self._index = utils.toindex(slice(0, self._frame.num_rows))
        else:
            self._index = index

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
        return len(self._index)

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

    def set_remote_init_builder(self, builder):
        """Set an initializer builder to create a remote initializer for a new column to a frame.

        NOTE(minjie): This is a temporary solution. Will be replaced by KVStore in the future.

        The builder is a callable that returns an initializer. The returned initializer
        is also a callable that returns a tensor given a local tensor and tensor name.

        Parameters
        ----------
        builder : callable
            The builder to construct a remote initializer.
        """
        self._frame.set_remote_init_builder(builder)

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

    def values(self):
        """Return the values."""
        return self._frame.values()

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
        key : str or utils.Index
            The key.

        Returns
        -------
        Tensor or lazy dict or tensors
            Depends on whether it is a column selection or row selection.
        """
        if not isinstance(key, (str, utils.Index)):
            raise DGLError('Argument "key" must be either str or utils.Index type.')
        if isinstance(key, str):
            return self.select_column(key)
        elif key.is_slice(0, self.num_rows):
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
            return col[self._index]

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
        """Update the data in the frame. The update is done out-of-place.

        Parameters
        ----------
        key : str or utils.Index
            The key.
        val : Tensor or dict of tensors
            The value.

        See Also
        --------
        update
        """
        self.update_data(key, val, inplace=False)

    def update_data(self, key, val, inplace):
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
        if not isinstance(key, (str, utils.Index)):
            raise DGLError('Argument "key" must be either str or utils.Index type.')
        if isinstance(key, str):
            self.update_column(key, val, inplace=inplace)
        elif key.is_slice(0, self.num_rows):
            # shortcut for updating all the rows
            for colname, col in val.items():
                self.update_column(colname, col, inplace=inplace)
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
            if self.num_columns == 0:
                # the frame is empty
                self._index = utils.toindex(slice(0, len(data)))
            self._frame[name] = data
        else:
            if name not in self._frame:
                ctx = F.context(data)
                self._frame.add_column(name, infer_scheme(data), ctx)
            fcol = self._frame[name]
            fcol.update(self._index, data, inplace)

    def add_rows(self, num_rows):
        """Add blank rows to the underlying frame.

        For existing fields, the rows will be extended according to their
        initializers.

        Note: only available for FrameRef that spans the whole column.  The row
        span will extend to new rows. Other FrameRefs referencing the same
        frame will not be affected.

        Parameters
        ----------
        num_rows : int
            Number of rows to add
        """
        if not self.is_span_whole_column():
            raise RuntimeError('FrameRef not spanning whole column.')
        self._frame.add_rows(num_rows)
        if self._index.slice_data() is not None:
            # the index is a slice
            slc = self._index.slice_data()
            self._index = utils.toindex(slice(slc.start, slc.stop + num_rows))
        else:
            selfidxdata = self._index.tousertensor()
            newdata = F.arange(self.num_rows, self.num_rows + num_rows)
            self._index = utils.toindex(F.cat([selfidxdata, newdata], dim=0))

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
            True if the update is performed inplace.
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
        from one ref will not reflect on the other. However, deleting columns is real.

        Parameters
        ----------
        key : str or utils.Index
            The key.
        """
        if not isinstance(key, (str, utils.Index)):
            raise DGLError('Argument "key" must be either str or utils.Index type.')
        if isinstance(key, str):
            del self._frame[key]
        else:
            self.delete_rows(key)

    def delete_rows(self, query):
        """Delete rows.

        Please note that "deleted" rows are not really deleted, but simply removed
        in the reference. As a result, if two FrameRefs point to the same Frame, deleting
        from one ref will not reflect on the other. By contrast, deleting columns is real.

        Parameters
        ----------
        query : utils.Index
            The rows to be deleted.
        """
        query = query.tonumpy()
        index = self._index.tonumpy()
        self._index = utils.toindex(np.delete(index, query))

    def append(self, other):
        """Append another frame into this one.

        Parameters
        ----------
        other : dict of str to tensor
            The data to be appended.
        """
        old_nrows = self._frame.num_rows
        self._frame.append(other)
        new_nrows = self._frame.num_rows
        # update index
        if (self._index.slice_data() is not None
                and self._index.slice_data().stop == old_nrows):
            # Self index is a slice and index.stop is equal to the size of the
            # underlying frame. Can still use a slice for the new index.
            oldstart = self._index.slice_data().start
            self._index = utils.toindex(slice(oldstart, new_nrows))
        else:
            # convert it to user tensor and concat
            selfidxdata = self._index.tousertensor()
            newdata = F.arange(old_nrows, new_nrows)
            self._index = utils.toindex(F.cat([selfidxdata, newdata], dim=0))

    def clear(self):
        """Clear the frame."""
        self._frame.clear()
        self._index = utils.toindex(slice(0, 0))

    def is_contiguous(self):
        """Return whether this refers to a contiguous range of rows."""
        # NOTE: this check could have false negatives
        return self._index.slice_data() is not None

    def is_span_whole_column(self):
        """Return whether this refers to all the rows."""
        return self.is_contiguous() and self.num_rows == self._frame.num_rows

    def clone(self):
        """Return a new reference to a clone of the underlying frame.

        Returns
        -------
        FrameRef
            A cloned frame reference.

        See Also
        --------
        dgl.Frame.clone
        """
        return FrameRef(self._frame.clone(), self._index)

    def deepclone(self):
        """Return a new reference to a deep clone of the underlying frame.

        Returns
        -------
        FrameRef
            A deep-cloned frame reference.

        See Also
        --------
        dgl.Frame.deepclone
        """
        return FrameRef(self._frame.deepclone(), self._index)

    def _getrows(self, query):
        """Internal function to convert from the local row ids to the row ids of the frame.

        Parameters
        ----------
        query : utils.Index
            The query index.

        Returns
        -------
        utils.Index
            The actual index to the underlying frame.
        """
        return self._index.get_items(query)

def frame_like(other, num_rows=None):
    """Create an empty frame that has the same initializer as the given one.

    Parameters
    ----------
    other : Frame
        The given frame.
    num_rows : int
        The number of rows of the new one. If None, use other.num_rows
        (Default: None)

    Returns
    -------
    Frame
        The new frame.
    """
    num_rows = other.num_rows if num_rows is None else num_rows
    newf = Frame(num_rows=num_rows)
    # set global initializr
    if other.get_initializer() is None:
        other._set_zero_default_initializer()
    sync_frame_initializer(newf, other)
    return newf

def sync_frame_initializer(new_frame, reference_frame):
    """Set the initializers of the new_frame to be the same as the reference_frame,
    for both the default initializer and per-column initializers.

    Parameters
    ----------
    new_frame : Frame
        The frame to set initializers
    reference_frame : Frame
        The frame to copy initializers
    """
    new_frame._default_initializer = reference_frame._default_initializer
    # set per-col initializer
    # TODO(minjie): hack; cannot rely on keys as the _initializers
    #   now supports non-exist columns.
    new_frame._initializers = reference_frame._initializers
