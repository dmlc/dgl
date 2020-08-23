"""Internal utilities."""
from __future__ import absolute_import, division

from collections.abc import Mapping, Iterable
from collections import defaultdict
from functools import wraps
import numpy as np

from ..base import DGLError, dgl_warning, NID, EID
from .. import backend as F
from .. import ndarray as nd
from .._ffi.function import _init_api

class InconsistentDtypeException(DGLError):
    """Exception class for inconsistent dtype between graph and tensor"""
    def __init__(self, msg='', *args, **kwargs): #pylint: disable=W1113
        prefix_message = 'DGL now requires the input tensor to have\
            the same dtype as the graph index\'s dtype(which you can get by g.idype). '
        super().__init__(prefix_message + msg, *args, **kwargs)

class Index(object):
    """Index class that can be easily converted to list/tensor."""
    def __init__(self, data, dtype="int64"):
        assert dtype in ['int32', 'int64']
        self.dtype = dtype
        self._initialize_data(data)

    def _initialize_data(self, data):
        self._pydata = None   # a numpy type data
        self._user_tensor_data = dict()  # dictionary of user tensors
        self._dgl_tensor_data = None  # a dgl ndarray
        self._slice_data = None # a slice type data
        self._dispatch(data)

    def __iter__(self):
        for i in self.tonumpy():
            yield int(i)

    def __len__(self):
        if self._slice_data is not None:
            slc = self._slice_data
            return slc.stop - slc.start
        elif self._pydata is not None:
            return len(self._pydata)
        elif len(self._user_tensor_data) > 0:
            data = next(iter(self._user_tensor_data.values()))
            return len(data)
        else:
            return len(self._dgl_tensor_data)

    def __getitem__(self, i):
        return int(self.tonumpy()[i])

    def _dispatch(self, data):
        """Store data based on its type."""
        if F.is_tensor(data):
            if F.dtype(data) != F.data_type_dict[self.dtype]:
                raise InconsistentDtypeException('Index data specified as %s, but got: %s' %
                                                 (self.dtype,
                                                  F.reverse_data_type_dict[F.dtype(data)]))
            if len(F.shape(data)) > 1:
                raise InconsistentDtypeException('Index data must be 1D int32/int64 vector,\
                    but got shape: %s' % str(F.shape(data)))
            if len(F.shape(data)) == 0:
                # a tensor of one int
                self._dispatch(int(data))
            else:
                self._user_tensor_data[F.context(data)] = data
        elif isinstance(data, nd.NDArray):
            if not (data.dtype == self.dtype and len(data.shape) == 1):
                raise InconsistentDtypeException('Index data must be 1D %s vector, but got: %s' %
                                                 (self.dtype, data.dtype))
            self._dgl_tensor_data = data
        elif isinstance(data, slice):
            # save it in the _pydata temporarily; materialize it if `tonumpy` is called
            assert data.step == 1 or data.step is None, \
                "step for slice type must be 1"
            self._slice_data = slice(data.start, data.stop)
        else:
            try:
                data = np.asarray(data, dtype=self.dtype)
            except Exception:  # pylint: disable=broad-except
                raise DGLError('Error index data: %s' % str(data))
            if data.ndim == 0:  # scalar array
                data = np.expand_dims(data, 0)
            elif data.ndim != 1:
                raise DGLError('Index data must be 1D int64 vector,'
                               ' but got: %s' % str(data))
            self._pydata = data
            self._user_tensor_data[F.cpu()] = F.zerocopy_from_numpy(self._pydata)

    def tonumpy(self):
        """Convert to a numpy ndarray."""
        if self._pydata is None:
            if self._slice_data is not None:
                slc = self._slice_data
                self._pydata = np.arange(slc.start, slc.stop).astype(self.dtype)
            elif self._dgl_tensor_data is not None:
                self._pydata = self._dgl_tensor_data.asnumpy()
            else:
                data = self.tousertensor()
                self._pydata = F.zerocopy_to_numpy(data)
        return self._pydata

    def tousertensor(self, ctx=None):
        """Convert to user tensor (defined in `backend`)."""
        if ctx is None:
            ctx = F.cpu()
        if len(self._user_tensor_data) == 0:
            if self._dgl_tensor_data is not None:
                # zero copy from dgl tensor
                dlpack = self._dgl_tensor_data.to_dlpack()
                self._user_tensor_data[F.cpu()] = F.zerocopy_from_dlpack(dlpack)
            else:
                # zero copy from numpy array
                self._user_tensor_data[F.cpu()] = F.zerocopy_from_numpy(self.tonumpy())
        if ctx not in self._user_tensor_data:
            # copy from cpu to another device
            data = next(iter(self._user_tensor_data.values()))
            self._user_tensor_data[ctx] = F.copy_to(data, ctx)
        return self._user_tensor_data[ctx]

    def todgltensor(self):
        """Convert to dgl.NDArray."""
        if self._dgl_tensor_data is None:
            # zero copy from user tensor
            tsor = self.tousertensor()
            dlpack = F.zerocopy_to_dlpack(tsor)
            self._dgl_tensor_data = nd.from_dlpack(dlpack)
        return self._dgl_tensor_data

    def slice_data(self):
        """Return the internal slice data.

        If this index is not initialized from slice, the return will be None.
        """
        return self._slice_data

    def is_slice(self, start, stop):
        """Check if Index wraps a slice data with given start and stop"""
        return self._slice_data == slice(start, stop)

    def __getstate__(self):
        if self._slice_data is not None:
            # the index can be represented by a slice
            return self._slice_data, self.dtype
        else:
            return self.tousertensor(), self.dtype

    def __setstate__(self, state):
        # Pickle compatibility check
        # TODO: we should store a storage version number in later releases.
        if isinstance(state, tuple) and len(state) == 2:
            # post-0.4.4
            data, self.dtype = state
            self._initialize_data(data)
        else:
            # pre-0.4.3
            dgl_warning("The object is pickled before 0.4.3.  Setting dtype of graph to int64")
            self.dtype = 'int64'
            self._initialize_data(state)

    def get_items(self, index):
        """Return values at given positions of an Index

        Parameters
        ----------
        index: utils.Index

        Returns
        -------
        utils.Index
            The values at the given position.
        """
        if self._slice_data is not None and self._slice_data.start == 0:
            # short-cut for identical mapping
            # NOTE: we don't check for out-of-bound error
            return index
        elif index._slice_data is None:
            # the provided index is not a slice
            tensor = self.tousertensor()
            index = index.tousertensor()
            # TODO(Allen): Change F.gather_row to dgl operation
            return Index(F.gather_row(tensor, index), self.dtype)
        elif self._slice_data is None:
            # the current index is not a slice but the provided is a slice
            tensor = self.tousertensor()
            index = index._slice_data
            # TODO(Allen): Change F.narrow_row to dgl operation
            return Index(F.astype(F.narrow_row(tensor, index.start, index.stop),
                                  F.data_type_dict[self.dtype]),
                         self.dtype)
        else:
            # both self and index wrap a slice object, then return another
            # Index wrapping a slice
            start = self._slice_data.start
            index = index._slice_data
            return Index(slice(start + index.start, start + index.stop), self.dtype)

    def set_items(self, index, value):
        """Set values at given positions of an Index. Set is not done in place,
        instead, a new Index object will be returned.

        Parameters
        ----------
        index: utils.Index
            Positions to set values
        value: int or utils.Index
            Values to set. If value is an integer, then all positions are set
            to the same value

        Returns
        -------
        utils.Index
            The new values.
        """
        tensor = self.tousertensor()
        index = index.tousertensor()
        if isinstance(value, int):
            value = F.full_1d(len(index), value, dtype=F.int64, ctx=F.cpu())
        else:
            value = value.tousertensor()
        return Index(F.scatter_row(tensor, index, value), self.dtype)

    def append_zeros(self, num):
        """Append zeros to an Index

        Parameters
        ----------
        num: int
            number of zeros to append
        """
        if num == 0:
            return self
        new_items = F.zeros((num,), dtype=F.int64, ctx=F.cpu())
        if len(self) == 0:
            return Index(new_items, self.dtype)
        else:
            tensor = self.tousertensor()
            tensor = F.cat((tensor, new_items), dim=0)
            return Index(tensor, self.dtype)

    def nonzero(self):
        """Return the nonzero positions"""
        tensor = self.tousertensor()
        mask = F.nonzero_1d(tensor != 0)
        return Index(mask, self.dtype)

    def has_nonzero(self):
        """Check if there is any nonzero value in this Index"""
        tensor = self.tousertensor()
        return F.sum(tensor, 0) > 0

def toindex(data, dtype='int64'):
    """Convert the given data to Index object.

    Parameters
    ----------
    data : index data
        Data to create the index.

    Returns
    -------
    Index
        The index object.

    See Also
    --------
    Index
    """
    return data if isinstance(data, Index) else Index(data, dtype)

def zero_index(size, dtype="int64"):
    """Create a index with provided size initialized to zero

    Parameters
    ----------
    size: int
    """
    return Index(F.zeros((size,), dtype=F.data_type_dict[dtype], ctx=F.cpu()),
                 dtype=dtype)

def set_diff(ar1, ar2):
    """Find the set difference of two index arrays.
    Return the unique values in ar1 that are not in ar2.

    Parameters
    ----------
    ar1: utils.Index
        Input index array.

    ar2: utils.Index
        Input comparison index array.

    Returns
    -------
    setdiff:
        Array of values in ar1 that are not in ar2.
    """
    ar1_np = ar1.tonumpy()
    ar2_np = ar2.tonumpy()
    setdiff = np.setdiff1d(ar1_np, ar2_np)
    setdiff = toindex(setdiff)
    return setdiff

class LazyDict(Mapping):
    """A readonly dictionary that does not materialize the storage."""
    def __init__(self, fn, keys):
        self._fn = fn
        self._keys = keys

    def __getitem__(self, key):
        if key not in self._keys:
            raise KeyError(key)
        return self._fn(key)

    def __contains__(self, key):
        return key in self._keys

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def keys(self):
        return self._keys

class HybridDict(Mapping):
    """A readonly dictonary that merges several dict-like (python dict, LazyDict).

    If there are duplicate keys, early keys have priority over latter ones.
    """
    def __init__(self, *dict_like_list):
        self._dict_like_list = dict_like_list
        self._keys = set()
        for obj in dict_like_list:
            self._keys.update(obj.keys())

    def keys(self):
        return self._keys

    def __getitem__(self, key):
        for obj in self._dict_like_list:
            if key in obj:
                return obj[key]
        raise KeyError(key)

    def __contains__(self, key):
        return key in self.keys()

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

class ReadOnlyDict(Mapping):
    """A readonly dictionary wrapper."""
    def __init__(self, dict_like):
        self._dict_like = dict_like

    def keys(self):
        return self._dict_like.keys()

    def __getitem__(self, key):
        return self._dict_like[key]

    def __contains__(self, key):
        return key in self._dict_like

    def __iter__(self):
        return iter(self._dict_like)

    def __len__(self):
        return len(self._dict_like)

def build_relabel_map(x, is_sorted=False):
    """Relabel the input ids to continuous ids that starts from zero.

    Ids are assigned new ids according to their ascending order.

    Examples
    --------
    >>> x = [1, 5, 3, 6]
    >>> n2o, o2n = build_relabel_map(x)
    >>> n2o
    [1, 3, 5, 6]
    >>> o2n
    [n/a, 0, n/a, 1, n/a, 2, 3]

    "n/a" will be filled with 0

    Parameters
    ----------
    x : Index
        The input ids.
    is_sorted : bool, default=False
        Whether the input has already been unique and sorted.

    Returns
    -------
    new_to_old : tensor
        The mapping from new id to old id.
    old_to_new : tensor
        The mapping from old id to new id. It is a vector of length MAX(x).
        One can use advanced indexing to convert an old id tensor to a
        new id tensor: new_id = old_to_new[old_id]
    """
    x = x.tousertensor()
    if not is_sorted:
        unique_x, _ = F.sort_1d(F.unique(x))
    else:
        unique_x = x
    map_len = int(F.asnumpy(F.max(unique_x, dim=0))) + 1
    old_to_new = F.zeros((map_len,), dtype=F.int64, ctx=F.cpu())
    old_to_new = F.scatter_row(old_to_new, unique_x, F.arange(0, len(unique_x)))
    return unique_x, old_to_new

def build_relabel_dict(x):
    """Relabel the input ids to continuous ids that starts from zero.

    The new id follows the order of the given node id list.

    Parameters
    ----------
    x : list
      The input ids.

    Returns
    -------
    relabel_dict : dict
      Dict from old id to new id.
    """
    relabel_dict = {}
    for i, v in enumerate(x):
        relabel_dict[v] = i
    return relabel_dict

class CtxCachedObject(object):
    """A wrapper to cache object generated by different context.

    Note: such wrapper may incur significant overhead if the wrapped object is very light.

    Parameters
    ----------
    generator : callable
        A callable function that can create the object given ctx as the only argument.
    """
    def __init__(self, generator):
        self._generator = generator
        self._ctx_dict = {}

    def __call__(self, ctx):
        if ctx not in self._ctx_dict:
            self._ctx_dict[ctx] = self._generator(ctx)
        return self._ctx_dict[ctx]

def cached_member(cache, prefix):
    """A member function decorator to memorize the result.

    Note that the member function cannot support kwargs after being decorated.
    The member function must be functional. Otherwise, the behavior is undefined.

    Parameters
    ----------
    cache : str
        The cache name. The cache should be a dictionary attribute
        in the class object.
    prefix : str
        The key prefix to save the result of the function.
    """
    def _creator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            dic = getattr(self, cache)
            key = '%s-%s-%s' % (
                prefix,
                '-'.join([str(a) for a in args]),
                '-'.join([str(k) + ':' + str(v) for k, v in kwargs.items()]))
            if key not in dic:
                dic[key] = func(self, *args, **kwargs)
            return dic[key]
        return wrapper
    return _creator

def is_dict_like(obj):
    """Return true if the object can be treated as a dictionary."""
    return isinstance(obj, Mapping)

def reorder(dict_like, index):
    """Reorder each column in the dict according to the index.

    Parameters
    ----------
    dict_like : dict of tensors
        The dict to be reordered.
    index : dgl.utils.Index
        The reorder index.
    """
    new_dict = {}
    for key, val in dict_like.items():
        idx_ctx = index.tousertensor(F.context(val))
        new_dict[key] = F.gather_row(val, idx_ctx)
    return new_dict

def reorder_index(idx, order):
    """Reorder the idx according to the given order

    Parameters
    ----------
    idx : utils.Index
        The index to be reordered.
    order : utils.Index
        The order to follow.
    """
    idx = idx.tousertensor()
    order = order.tousertensor()
    new_idx = F.gather_row(idx, order)
    return toindex(new_idx)

def is_iterable(obj):
    """Return true if the object is an iterable."""
    return isinstance(obj, Iterable)

def to_dgl_context(ctx):
    """Convert a backend context to DGLContext"""
    device_type = nd.DGLContext.STR2MASK[F.device_type(ctx)]
    device_id = F.device_id(ctx)
    return nd.DGLContext(device_type, device_id)

def to_nbits_int(tensor, nbits):
    """Change the dtype of integer tensor
    The dtype of returned tensor uses nbits, nbits can only be 32 or 64
    """
    assert(nbits in (32, 64)), "nbits can either be 32 or 64"
    if nbits == 32:
        return F.astype(tensor, F.int32)
    else:
        return F.astype(tensor, F.int64)

def make_invmap(array, use_numpy=True):
    """Find the unique elements of the array and return another array with indices
    to the array of unique elements."""
    if use_numpy:
        uniques = np.unique(array)
    else:
        uniques = list(set(array))
    invmap = {x: i for i, x in enumerate(uniques)}
    remapped = np.asarray([invmap[x] for x in array])
    return uniques, invmap, remapped

def expand_as_pair(input_, g=None):
    """Return a pair of same element if the input is not a pair.

    If the graph is a block, obtain the feature of destination nodes from the source nodes.

    Parameters
    ----------
    input_ : Tensor, dict[str, Tensor], or their pairs
        The input features
    g : DGLHeteroGraph or DGLGraph or None
        The graph.

        If None, skip checking if the graph is a block.

    Returns
    -------
    tuple[Tensor, Tensor] or tuple[dict[str, Tensor], dict[str, Tensor]]
        The features for input and output nodes
    """
    if isinstance(input_, tuple):
        return input_
    elif g is not None and g.is_block:
        if isinstance(input_, Mapping):
            input_dst = {
                k: F.narrow_row(v, 0, g.number_of_dst_nodes(k))
                for k, v in input_.items()}
        else:
            input_dst = F.narrow_row(input_, 0, g.number_of_dst_nodes())
        return input_, input_dst
    else:
        return input_, input_

def check_eq_shape(input_):
    """If input_ is a pair of features, check if the feature shape of source
    nodes is equal to the feature shape of destination nodes.
    """
    srcdata, dstdata = expand_as_pair(input_)
    src_feat_shape = tuple(F.shape(srcdata))[1:]
    dst_feat_shape = tuple(F.shape(dstdata))[1:]
    if src_feat_shape != dst_feat_shape:
        raise DGLError("The feature shape of source nodes: {} \
            should be equal to the feature shape of destination \
            nodes: {}.".format(src_feat_shape, dst_feat_shape))

def retry_method_with_fix(fix_method):
    """Decorator that executes a fix method before retrying again when the decorated method
    fails once with any exception.

    If the decorated method fails again, the execution fails with that exception.

    Notes
    -----
    This decorator only works on class methods, and the fix function must also be a class method.
    It would not work on functions.

    Parameters
    ----------
    fix_func : callable
        The fix method to execute.  It should not accept any arguments.  Its return values are
        ignored.
    """
    def _creator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # pylint: disable=W0703,bare-except
            try:
                return func(self, *args, **kwargs)
            except:
                fix_method(self)
                return func(self, *args, **kwargs)

        return wrapper
    return _creator

def group_as_dict(pairs):
    """Combines a list of key-value pairs to a dictionary of keys and value lists.

    Does not require the pairs to be sorted by keys.

    Parameters
    ----------
    pairs : iterable
        Iterable of key-value pairs

    Returns
    -------
    dict
        The dictionary of keys and value lists.
    """
    dic = defaultdict(list)
    for key, value in pairs:
        dic[key].append(value)
    return dic

class FlattenedDict(object):
    """Iterates over each item in a dictionary of groups.

    Parameters
    ----------
    groups : dict
        The item groups.

    Examples
    --------
    >>> groups = FlattenedDict({'a': [1, 3], 'b': [2, 5, 8], 'c': [7]})
    >>> list(groups)
    [('a', 1), ('a', 3), ('b', 2), ('b', 5), ('b', 8), ('c', 7)]
    >>> groups[2]
    ('b', 2)
    >>> len(groups)
    6
    """
    def __init__(self, groups):
        self._groups = groups
        group_sizes = {k: len(v) for k, v in groups.items()}
        self._group_keys, self._group_sizes = zip(*group_sizes.items())
        self._group_offsets = np.insert(np.cumsum(self._group_sizes), 0, 0)
        # TODO: this is faster (37s -> 21s per epoch compared to searchsorted in GCMC) but takes
        # O(E) memory.
        self._idx_to_group = np.zeros(self._group_offsets[-1], dtype='int32')
        for i in range(len(self._groups)):
            self._idx_to_group[self._group_offsets[i]:self._group_offsets[i + 1]] = i

    def __len__(self):
        """Return the total number of items."""
        return self._group_offsets[-1]

    def __iter__(self):
        """Return the iterator of all items with the key of its original group."""
        for i, k in enumerate(self._group_keys):
            for j in range(self._group_sizes[i]):
                yield k, self._groups[k][j]

    def __getitem__(self, idx):
        """Return the item at the given position with the key of its original group."""
        i = self._idx_to_group[idx]
        k = self._group_keys[i]
        j = idx - self._group_offsets[i]
        g = self._groups[k]
        return k, g[j]

def compensate(ids, origin_ids):
    """computing the compensate set of ids from origin_ids

    Note: ids should be a subset of origin_ids.
    Any of ids and origin_ids can be non-consecutive,
    and origin_ids should be sorted.

    Example:
    >>> ids = th.Tensor([0, 2, 4])
    >>> origin_ids = th.Tensor([0, 1, 2, 4, 5])
    >>> compensate(ids, origin_ids)
    th.Tensor([1, 5])
    """
    # trick here, eid_0 or nid_0 can be 0.
    mask = F.scatter_row(origin_ids,
                         F.copy_to(F.tensor(0, dtype=F.int64),
                                   F.context(origin_ids)),
                         F.copy_to(F.tensor(1, dtype=F.dtype(origin_ids)),
                                   F.context(origin_ids)))
    mask = F.scatter_row(mask,
                         ids,
                         F.full_1d(len(ids), 0, F.dtype(ids), F.context(ids)))
    return F.tensor(F.nonzero_1d(mask), dtype=F.dtype(ids))

def relabel(x):
    """Relabel the input ids to continuous ids that starts from zero.

    Ids are assigned new ids according to their ascending order.

    Examples
    --------
    >>> x = [1, 5, 3, 6]
    >>> n2o, o2n = build_relabel_map(x)
    >>> n2o
    [1, 3, 5, 6]
    >>> o2n
    [n/a, 0, n/a, 1, n/a, 2, 3]

    "n/a" will be filled with 0

    Parameters
    ----------
    x : Tensor
        ID tensor.

    Returns
    -------
    new_to_old : Tensor
        The mapping from new id to old id.
    old_to_new : Tensor
        The mapping from old id to new id. It is a vector of length MAX(x).
        One can use advanced indexing to convert an old id tensor to a
        new id tensor: new_id = old_to_new[old_id]
    """
    unique_x = F.unique(x)
    map_len = F.as_scalar(F.max(unique_x, dim=0)) + 1
    ctx = F.context(x)
    dtype = F.dtype(x)
    old_to_new = F.zeros((map_len,), dtype=dtype, ctx=ctx)
    old_to_new = F.scatter_row(old_to_new, unique_x,
                               F.copy_to(F.arange(0, len(unique_x), dtype), ctx))
    return unique_x, old_to_new

def extract_node_subframes(graph, nodes):
    """Extract node features of the given nodes from :attr:`graph`
    and return them in frames.

    Note that this function does not perform actual tensor memory copy but using `Frame.subframe`
    to get the features. If :attr:`nodes` is None, it performs a shallow copy of the
    original node frames that only copies the dictionary structure but not the tensor
    contents.

    Parameters
    ----------
    graph : DGLGraph
        The graph to extract features from.
    nodes : list[Tensor] or None
        Node IDs. If not None, the list length must be equal to the number of node types
        in the graph. The returned frames store the node IDs in the ``dgl.NID`` field
        unless it is None, which means the whole frame is shallow-copied.

    Returns
    -------
    list[Frame]
        Extracted node frames.
    """
    if nodes is None:
        node_frames = [nf.clone() for nf in graph._node_frames]
    else:
        node_frames = []
        for i, ind_nodes in enumerate(nodes):
            subf = graph._node_frames[i].subframe(ind_nodes)
            subf[NID] = ind_nodes
            node_frames.append(subf)
    return node_frames

def extract_node_subframes_for_block(graph, srcnodes, dstnodes):
    """Extract the input node features and output node features of the given nodes from
    :attr:`graph` and return them in frames ready for a block.

    Note that this function does not perform actual tensor memory copy but using `Frame.subframe`
    to get the features. If :attr:`srcnodes` or :attr:`dstnodes` is None, it performs a
    shallow copy of the original node frames that only copies the dictionary structure
    but not the tensor contents.

    Parameters
    ----------
    graph : DGLGraph
        The graph to extract features from.
    srcnodes : list[Tensor]
        Input node IDs. The list length must be equal to the number of node types
        in the graph. The returned frames store the node IDs in the ``dgl.NID`` field.
    dstnodes : list[Tensor]
        Output node IDs. The list length must be equal to the number of node types
        in the graph. The returned frames store the node IDs in the ``dgl.NID`` field.

    Returns
    -------
    list[Frame]
        Extracted node frames.
    """
    node_frames = []
    for i, ind_nodes in enumerate(srcnodes):
        subf = graph._node_frames[i].subframe(ind_nodes)
        subf[NID] = ind_nodes
        node_frames.append(subf)
    for i, ind_nodes in enumerate(dstnodes):
        subf = graph._node_frames[i].subframe(ind_nodes)
        subf[NID] = ind_nodes
        node_frames.append(subf)
    return node_frames

def extract_edge_subframes(graph, edges):
    """Extract edge features of the given edges from :attr:`graph`
    and return them in frames.

    Note that this function does not perform actual tensor memory copy but using `Frame.subframe`
    to get the features. If :attr:`edges` is None, it performs a shallow copy of the
    original edge frames that only copies the dictionary structure but not the tensor
    contents.

    Parameters
    ----------
    graph : DGLGraph
        The graph to extract features from.
    edges : list[Tensor] or None
        Edge IDs. If not None, the list length must be equal to the number of edge types
        in the graph. The returned frames store the edge IDs in the ``dgl.NID`` field
        unless it is None, which means the whole frame is shallow-copied.

    Returns
    -------
    list[Frame]
        Extracted edge frames.
    """
    if edges is None:
        edge_frames = [nf.clone() for nf in graph._edge_frames]
    else:
        edge_frames = []
        for i, ind_edges in enumerate(edges):
            subf = graph._edge_frames[i].subframe(ind_edges)
            subf[EID] = ind_edges
            edge_frames.append(subf)
    return edge_frames

def set_new_frames(graph, *, node_frames=None, edge_frames=None):
    """Set the node and edge frames of a given graph to new ones.

    Parameters
    ----------
    graph : DGLGraph
        The graph whose node and edge frames are to be updated.
    node_frames : list[Frame], optional
        New node frames.

        Default is None, where the node frames are not updated.
    edge_frames : list[Frame], optional
        New edge frames

        Default is None, where the edge frames are not updated.
    """
    if node_frames is not None:
        assert len(node_frames) == len(graph.ntypes), \
            "[BUG] number of node frames different from number of node types"
        graph._node_frames = node_frames
    if edge_frames is not None:
        assert len(edge_frames) == len(graph.etypes), \
            "[BUG] number of edge frames different from number of edge types"
        graph._edge_frames = edge_frames

def set_num_threads(num_threads):
    """Set the number of OMP threads in the process.

    Parameters
    ----------
    num_threads : int
        The number of OMP threads in the process.
    """
    _CAPI_DGLSetOMPThreads(num_threads)

def alias_func(func):
    """Return an alias function with proper docstring."""
    @wraps(func)
    def _fn(*args, **kwargs):
        return func(*args, **kwargs)
    _fn.__doc__ = """Alias of :func:`dgl.{}`.""".format(func.__name__)
    return _fn

_init_api("dgl.utils.internal")
