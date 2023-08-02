"""Columnar storage for DGLGraph."""
from __future__ import absolute_import

from collections import namedtuple
from collections.abc import MutableMapping

from . import backend as F
from .base import dgl_warning, DGLError
from .init import zero_initializer
from .storages import TensorStorage
from .utils import gather_pinned_tensor_rows, pin_memory_inplace


class _LazyIndex(object):
    def __init__(self, index):
        if isinstance(index, list):
            self._indices = index
        else:
            self._indices = [index]

    def __len__(self):
        return len(self._indices[-1])

    def slice(self, index):
        """Create a new _LazyIndex object sliced by the given index tensor."""
        # if our indices are in the same context, lets just slice now and free
        # memory, otherwise do nothing until we have to
        if F.context(self._indices[-1]) == F.context(index):
            return _LazyIndex(
                self._indices[:-1] + [F.gather_row(self._indices[-1], index)]
            )
        return _LazyIndex(self._indices + [index])

    def flatten(self):
        """Evaluate the chain of indices, and return a single index tensor."""
        flat_index = self._indices[0]
        # here we actually need to resolve it
        for index in self._indices[1:]:
            if F.context(index) != F.context(flat_index):
                index = F.copy_to(index, F.context(flat_index))
            flat_index = F.gather_row(flat_index, index)
        return flat_index

    def record_stream(self, stream):
        """Record stream for index.

        Parameters
        ----------
        stream : torch.cuda.Stream.
        """
        for index in self._indices:
            if F.context(index) != F.cpu():
                index.record_stream(stream)


class LazyFeature(object):
    """Placeholder for feature prefetching.

    One can assign this object to ``ndata`` or ``edata`` of the graphs returned by various
    samplers' :attr:`sample` method.  When DGL's dataloader receives the subgraphs
    returned by the sampler, it will automatically look up all the ``ndata`` and ``edata``
    whose data is a LazyFeature, replacing them with the actual data of the corresponding
    nodes/edges from the original graph instead.  In particular, for a subgraph returned
    by the sampler has a LazyFeature with name ``k`` in ``subgraph.ndata[key]``:

    .. code:: python

       subgraph.ndata[key] = LazyFeature(k)

    Assuming that ``graph`` is the original graph, DGL's dataloader will perform

    .. code:: python

       subgraph.ndata[key] = graph.ndata[k][subgraph.ndata[dgl.NID]]

    DGL dataloader performs similar replacement for ``edata``.
    For heterogeneous graphs, the replacement is:

    .. code:: python

       subgraph.nodes[ntype].data[key] = graph.nodes[ntype].data[k][
           subgraph.nodes[ntype].data[dgl.NID]]

    For MFGs' ``srcdata`` (and similarly ``dstdata``), the replacement is

    .. code:: python

       mfg.srcdata[key] = graph.ndata[k][mfg.srcdata[dgl.NID]]

    Parameters
    ----------
    name : str
        The name of the data in the original graph.
    id_ : Tensor, optional
        The ID tensor.
    """

    __slots__ = ["name", "id_"]

    def __init__(self, name=None, id_=None):
        self.name = name
        self.id_ = id_

    def to(
        self, *args, **kwargs
    ):  # pylint: disable=invalid-name, unused-argument
        """No-op.  For compatibility of :meth:`Frame.to` method."""
        return self

    @property
    def data(self):
        """No-op.  For compatibility of :meth:`Frame.__repr__` method."""
        return self

    def pin_memory_(self):
        """No-op.  For compatibility of :meth:`Frame.pin_memory_` method."""

    def unpin_memory_(self):
        """No-op.  For compatibility of :meth:`Frame.unpin_memory_` method."""

    def record_stream(self, stream):
        """No-op.  For compatibility of :meth:`Frame.record_stream` method."""


class Scheme(namedtuple("Scheme", ["shape", "dtype"])):
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


class Column(TensorStorage):
    """A column is a compact store of features of multiple nodes/edges.

    It batches all the feature tensors together along the first dimension
    as one dense tensor.

    The column can optionally have an index tensor I.
    In this case, the i^th feature is stored in ``storage[index[i]]``.
    The column class implements a Copy-On-Read semantics -- the index
    select operation happens upon the first read of the feature data.
    This is useful when one extracts a subset of the feature data
    but wishes the actual index select happens on-demand.

    Parameters
    ----------
    storage : Tensor
        The feature data storage.
    scheme : Scheme, optional
        The scheme of the column. Will be inferred if not provided.
    index : Tensor, optional
        The row index to the feature data storage. None means an
        identity mapping.

    Attributes
    ----------
    storage : Tensor
        The storage tensor. The storage tensor may not be the actual data
        tensor of this column when the index tensor is not None.
        This typically happens when the column is extracted from another
        column using the `subcolumn` method.

        It can also be None, which may only happen when transmitting a
        not-yet-materialized subcolumn from a subprocess to the main process.
        In this case, the main process should already maintain the content of
        the storage, and is responsible for restoring the subcolumn's storage pointer.
    data : Tensor
        The actual data tensor of this column.
    scheme : Scheme
        The scheme of the column.
    index : Tensor
        Index tensor
    """

    def __init__(self, storage, *args, **kwargs):
        super().__init__(storage)
        self._init(*args, **kwargs)

    def __len__(self):
        """The number of features (number of rows) in this column."""
        if self.index is None:
            return F.shape(self.storage)[0]
        else:
            return len(self.index)

    @property
    def shape(self):
        """Return the scheme shape (feature shape) of this column."""
        return self.scheme.shape

    @property
    def data(self):
        """Return the feature data. Perform index selecting if needed."""
        if self.index is not None:
            if isinstance(self.index, _LazyIndex):
                self.index = self.index.flatten()

            storage_ctx = F.context(self.storage)
            index_ctx = F.context(self.index)
            # If under the special case where the storage is pinned and the index is on
            # CUDA, directly call UVA slicing (even if they aree not in the same context).
            if (
                storage_ctx != index_ctx
                and storage_ctx == F.cpu()
                and F.is_pinned(self.storage)
            ):
                self.storage = gather_pinned_tensor_rows(
                    self.storage, self.index
                )
            else:
                # If index and storage is not in the same context,
                # copy index to the same context of storage.
                # Copy index is usually cheaper than copy data
                if storage_ctx != index_ctx:
                    kwargs = {}
                    if self.device is not None:
                        kwargs = self.device[1]
                    self.index = F.copy_to(self.index, storage_ctx, **kwargs)
                self.storage = F.gather_row(self.storage, self.index)
            self.index = None

        # move data to the right device
        if self.device is not None:
            self.storage = F.copy_to(
                self.storage, self.device[0], **self.device[1]
            )
            self.device = None

        # convert data to the right type
        if self.deferred_dtype is not None:
            self.storage = F.astype(self.storage, self.deferred_dtype)
            self.deferred_dtype = None
        return self.storage

    @data.setter
    def data(self, val):
        """Update the column data."""
        self.index = None
        self.device = None
        self.deferred_dtype = None
        self.storage = val
        self._data_nd = None  # should unpin data if it was pinned.
        self.pinned_by_dgl = False

    def to(self, device, **kwargs):  # pylint: disable=invalid-name
        """Return a new column with columns copy to the targeted device (cpu/gpu).

        Parameters
        ----------
        device : Framework-specific device context object
            The context to move data to.
        kwargs : Key-word arguments.
            Key-word arguments fed to the framework copy function.

        Returns
        -------
        Column
            A new column
        """
        col = self.clone()
        col.device = (device, kwargs)
        return col

    @property
    def dtype(self):
        """Return the effective data type of this Column"""
        if self.deferred_dtype is not None:
            return self.deferred_dtype
        return self.storage.dtype

    def astype(self, new_dtype):
        """Return a new column such that when its data is requested,
        it will be converted to new_dtype.

        Parameters
        ----------
        new_dtype : Framework-specific type object
            The type to convert the data to.

        Returns
        -------
        Column
            A new column
        """
        col = self.clone()
        if col.dtype != new_dtype:
            # If there is already a pending conversion, ensure that the pending
            # conversion and transfer/sampling are done before this new conversion.
            if col.deferred_dtype is not None:
                _ = col.data

            if (col.device is None) and (col.index is None):
                # Do the conversion immediately if no device transfer or index
                # sampling is pending.  The assumption is that this is most
                # likely to be the desired behaviour, such as converting an
                # entire graph's feature data to float16 (half) before transfer
                # to device when training, or converting back to float32 (float)
                # after fetching the data to a device.
                col.storage = F.astype(col.storage, new_dtype)
            else:
                # Defer the conversion if there is a pending transfer or sampling.
                # This is so that feature data that never gets accessed on the
                # device never needs to be transferred or sampled or converted.
                col.deferred_dtype = new_dtype
        return col

    def __getitem__(self, rowids):
        """Return the feature data given the rowids.

        The operation triggers index selection.

        Parameters
        ----------
        rowids : Tensor
            Row ID tensor.

        Returns
        -------
        Tensor
            The feature data
        """
        return F.gather_row(self.data, rowids)

    def __setitem__(self, rowids, feats):
        """Update the feature data given the index.

        The update is performed out-placely so it can be used in autograd mode.
        The operation triggers index selection.

        Parameters
        ----------
        rowids : Tensor
            Row IDs.
        feats : Tensor
            New features.
        """
        self.update(rowids, feats)

    def update(self, rowids, feats):
        """Update the feature data given the index.

        Parameters
        ----------
        rowids : Tensor
            Row IDs.
        feats : Tensor
            New features.
        """
        feat_scheme = infer_scheme(feats)
        if feat_scheme != self.scheme:
            raise DGLError(
                "Cannot update column of scheme %s using feature of scheme %s."
                % (feat_scheme, self.scheme)
            )
        self.data = F.scatter_row(self.data, rowids, feats)

    def extend(self, feats, feat_scheme=None):
        """Extend the feature data.

        The operation triggers index selection.

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
            raise DGLError(
                "Cannot update column of scheme %s using feature of scheme %s."
                % (feat_scheme, self.scheme)
            )

        self.data = F.cat([self.data, feats], dim=0)

    def clone(self):
        """Return a shallow copy of this column."""
        return Column(
            self.storage,
            self.scheme,
            self.index,
            self.device,
            self.deferred_dtype,
        )

    def deepclone(self):
        """Return a deepcopy of this column.

        The operation triggers index selection.
        """
        return Column(F.clone(self.data), copy.deepcopy(self.scheme))

    def subcolumn(self, rowids):
        """Return a subcolumn.

        The resulting column will share the same storage as this column so this operation
        is quite efficient. If the current column is also a sub-column (i.e.,
        the index tensor is not None), the current index tensor will be sliced
        by 'rowids', if they are on the same context. Otherwise, both index
        tensors are saved, and only applied when the data is accessed.

        Parameters
        ----------
        rowids : Tensor
            Row IDs.

        Returns
        -------
        Column
            Sub-column
        """
        if self.index is None:
            return Column(
                self.storage,
                self.scheme,
                rowids,
                self.device,
                self.deferred_dtype,
            )
        else:
            index = self.index
            if not isinstance(index, _LazyIndex):
                index = _LazyIndex(self.index)
            index = index.slice(rowids)
            return Column(
                self.storage,
                self.scheme,
                index,
                self.device,
                self.deferred_dtype,
            )

    @staticmethod
    def create(data):
        """Create a new column using the given data."""
        if isinstance(data, Column):
            return data.clone()
        else:
            return Column(data)

    def __repr__(self):
        return repr(self.data)

    def __getstate__(self):
        if self.storage is not None:
            # flush any deferred operations
            _ = self.data
        state = self.__dict__.copy()
        # data pinning does not get serialized, so we need to remove that from
        # the state
        state["_data_nd"] = None
        state["pinned_by_dgl"] = False
        return state

    def __setstate__(self, state):
        index = None
        device = None
        if "storage" in state and state["storage"] is not None:
            assert "index" not in state or state["index"] is None
            assert "device" not in state or state["device"] is None
        else:
            # we may have a column with only index information, and that is
            # valid
            index = None if "index" not in state else state["index"]
            device = None if "device" not in state else state["device"]
        assert "deferred_dtype" not in state or state["deferred_dtype"] is None
        assert "pinned_by_dgl" not in state or state["pinned_by_dgl"] is False
        assert "_data_nd" not in state or state["_data_nd"] is None

        self.__dict__ = state
        # properly initialize this object
        self._init(
            self.scheme if hasattr(self, "scheme") else None,
            index=index,
            device=device,
        )

    def _init(self, scheme=None, index=None, device=None, deferred_dtype=None):
        self.scheme = scheme if scheme else infer_scheme(self.storage)
        self.index = index
        self.device = device
        self.deferred_dtype = deferred_dtype
        self.pinned_by_dgl = False
        self._data_nd = None

    def __copy__(self):
        return self.clone()

    def fetch(self, indices, device, pin_memory=False, **kwargs):
        _ = self.data  # materialize in case of lazy slicing & data transfer
        return super().fetch(indices, device, pin_memory=pin_memory, **kwargs)

    def pin_memory_(self):
        """Pin the storage into page-locked memory.

        Does nothing if the storage is already pinned.
        """
        if not self.pinned_by_dgl and not F.is_pinned(self.data):
            self._data_nd = pin_memory_inplace(self.data)
            self.pinned_by_dgl = True

    def unpin_memory_(self):
        """Unpin the storage pinned by ``pin_memory_`` method.

        Does nothing if the storage is not pinned by ``pin_memory_`` method, even if
        it is actually in page-locked memory.
        """
        if self.pinned_by_dgl:
            self._data_nd.unpin_memory_()
            self._data_nd = None
            self.pinned_by_dgl = False

    def record_stream(self, stream):
        """Record stream that is using the storage.
        Does nothing if the backend is not PyTorch.

        Parameters
        ----------
        stream : torch.cuda.Stream.
        """
        if F.get_preferred_backend() != "pytorch":
            raise DGLError("record_stream only supports the PyTorch backend.")
        if self.index is not None and (
            isinstance(self.index, _LazyIndex)
            or F.context(self.index) != F.cpu()
        ):
            self.index.record_stream(stream)
        if F.context(self.storage) != F.cpu():
            self.storage.record_stream(stream)


class Frame(MutableMapping):
    """The columnar storage for node/edge features.

    The frame is a dictionary from feature names to feature columns.
    All columns should have the same number of rows (i.e. the same first dimension).

    Parameters
    ----------
    data : dict-like, optional
        The frame data in dictionary. If the provided data is another frame,
        this frame will NOT share columns with the given frame. So any out-place
        update on one will not reflect to the other.
    num_rows : int, optional
        The number of rows in this frame. If ``data`` is provided and is not empty,
        ``num_rows`` will be ignored and inferred from the given data.
    """

    def __init__(self, data=None, num_rows=None):
        if data is None:
            self._columns = dict()
            self._num_rows = 0 if num_rows is None else num_rows
        else:
            assert not isinstance(data, Frame)  # sanity check for code refactor
            # Note that we always create a new column for the given data.
            # This avoids two frames accidentally sharing the same column.
            self._columns = {
                k: v if isinstance(v, LazyFeature) else Column.create(v)
                for k, v in data.items()
            }
            self._num_rows = num_rows
            # infer num_rows & sanity check
            for name, col in self._columns.items():
                if isinstance(col, LazyFeature):
                    continue
                if self._num_rows is None:
                    self._num_rows = len(col)
                elif len(col) != self._num_rows:
                    raise DGLError(
                        "Expected all columns to have same # rows (%d), "
                        "got %d on %r." % (self._num_rows, len(col), name)
                    )

        # Initializer for empty values. Initializer is a callable.
        # If is none, then a warning will be raised
        # in the first call and zero initializer will be used later.
        self._initializers = {}  # per-column initializers
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

    @property
    def schemes(self):
        """Return a dictionary of column name to column schemes."""
        return {k: col.scheme for k, col in self._columns.items()}

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
        Tensor
            Column data.
        """
        return self._columns[name].data

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
            dgl_warning(
                'Column "%s" already exists. Ignore adding this column again.'
                % name
            )
            return

        if self.get_initializer(name) is None:
            self._set_zero_default_initializer()
        initializer = self.get_initializer(name)
        init_data = initializer(
            (self.num_rows,) + scheme.shape,
            scheme.dtype,
            ctx,
            slice(0, self.num_rows),
        )
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
            new_data = initializer(
                (num_rows,) + scheme.shape,
                scheme.dtype,
                ctx,
                slice(self._num_rows, self._num_rows + num_rows),
            )
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
        if isinstance(data, LazyFeature):
            self._columns[name] = data
            return

        col = Column.create(data)
        if len(col) != self.num_rows:
            raise DGLError(
                "Expected data to have %d rows, got %d."
                % (self.num_rows, len(col))
            )
        self._columns[name] = col

    def update_row(self, rowids, data):
        """Update the feature data of the given rows.

        If the data contains new keys (new columns) that do not exist in
        this frame, add a new column.

        The ``rowids`` shall not contain duplicates. Otherwise, the behavior
        is undefined.

        Parameters
        ----------
        rowids : Tensor
            Row Ids.
        data : dict[str, Tensor]
            Row data.
        """
        for key, val in data.items():
            if key not in self:
                scheme = infer_scheme(val)
                ctx = F.context(val)
                self.add_column(key, scheme, ctx)
        for key, val in data.items():
            self._columns[key].update(rowids, val)

    def _append(self, other):
        """Append ``other`` frame to ``self`` frame."""
        # pad columns that are not provided in the other frame with initial values
        for key, col in self._columns.items():
            if key in other:
                continue
            scheme = col.scheme
            ctx = F.context(col.data)
            if self.get_initializer(key) is None:
                self._set_zero_default_initializer()
            initializer = self.get_initializer(key)
            new_data = initializer(
                (other.num_rows,) + scheme.shape,
                scheme.dtype,
                ctx,
                slice(self._num_rows, self._num_rows + other.num_rows),
            )
            other[key] = new_data
        # append other to self
        for key, col in other._columns.items():
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
        newframe = Frame(
            {k: col.deepclone() for k, col in self._columns.items()},
            self._num_rows,
        )
        newframe._initializers = self._initializers
        newframe._default_initializer = self._default_initializer
        return newframe

    def subframe(self, rowids):
        """Return a new frame whose columns are subcolumns of this frame.

        The given row IDs should be within range [0, self.num_rows), and allow
        duplicate IDs.

        Parameters
        ----------
        rowids : Tensor
            Row IDs

        Returns
        -------
        Frame
            A new subframe.
        """
        subcols = {k: col.subcolumn(rowids) for k, col in self._columns.items()}
        subf = Frame(subcols, len(rowids))
        subf._initializers = self._initializers
        subf._default_initializer = self._default_initializer
        return subf

    def to(self, device, **kwargs):  # pylint: disable=invalid-name
        """Return a new frame with columns copy to the targeted device (cpu/gpu).

        Parameters
        ----------
        device : Framework-specific device context object
            The context to move data to.
        kwargs : Key-word arguments.
            Key-word arguments fed to the framework copy function.

        Returns
        -------
        Frame
            A new frame
        """
        newframe = self.clone()
        new_columns = {
            key: col.to(device, **kwargs)
            for key, col in newframe._columns.items()
        }
        newframe._columns = new_columns
        return newframe

    def __repr__(self):
        return repr(dict(self))

    def pin_memory_(self):
        """Registers the data of every column into pinned memory, materializing them if
        necessary."""
        for column in self._columns.values():
            column.pin_memory_()

    def unpin_memory_(self):
        """Unregisters the data of every column from pinned memory, materializing them
        if necessary."""
        for column in self._columns.values():
            column.unpin_memory_()

    def record_stream(self, stream):
        """Record stream that is using the data of every column, materializing them
        if necessary."""
        for column in self._columns.values():
            column.record_stream(stream)

    def _astype_float(self, new_type):
        assert new_type in [
            F.float64,
            F.float32,
            F.float16,
            F.bfloat16,
        ], "'new_type' must be floating-point type: %s" % str(new_type)
        newframe = self.clone()
        new_columns = {}
        for name, column in self._columns.items():
            dtype = column.dtype
            if dtype != new_type and dtype in [
                F.float64,
                F.float32,
                F.float16,
                F.bfloat16,
            ]:
                new_columns[name] = column.astype(new_type)
            else:
                new_columns[name] = column
        newframe._columns = new_columns
        return newframe

    def bfloat16(self):
        """Return a new frame with all floating-point columns converted
        to bfloat16"""
        return self._astype_float(F.bfloat16)

    def half(self):
        """Return a new frame with all floating-point columns converted
        to half-precision (float16)"""
        return self._astype_float(F.float16)

    def float(self):
        """Return a new frame with all floating-point columns converted
        to single-precision (float32)"""
        return self._astype_float(F.float32)

    def double(self):
        """Return a new frame with all floating-point columns converted
        to double-precision (float64)"""
        return self._astype_float(F.float64)
