"""Utility module."""
from __future__ import absolute_import, division

from collections.abc import Mapping, Iterable
from functools import wraps
import numpy as np

from .base import DGLError
from . import backend as F
from . import ndarray as nd

class Index(object):
    """Index class that can be easily converted to list/tensor."""
    def __init__(self, data):
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
            if F.dtype(data) != F.int64:
                raise DGLError('Index data must be an int64 vector, but got: %s' % str(data))
            if len(F.shape(data)) > 1:
                raise DGLError('Index data must be 1D int64 vector, but got: %s' % str(data))
            if len(F.shape(data)) == 0:
                # a tensor of one int
                self._dispatch(int(data))
            else:
                self._user_tensor_data[F.context(data)] = data
        elif isinstance(data, nd.NDArray):
            if not (data.dtype == 'int64' and len(data.shape) == 1):
                raise DGLError('Index data must be 1D int64 vector, but got: %s' % str(data))
            self._dgl_tensor_data = data
        elif isinstance(data, slice):
            # save it in the _pydata temporarily; materialize it if `tonumpy` is called
            assert data.step == 1 or data.step is None, \
                "step for slice type must be 1"
            self._slice_data = slice(data.start, data.stop)
        else:
            try:
                data = np.array(data).astype(np.int64)
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
                self._pydata = np.arange(slc.start, slc.stop).astype(np.int64)
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
            return self._slice_data
        else:
            return self.tousertensor()

    def __setstate__(self, state):
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
            return Index(F.gather_row(tensor, index))
        elif self._slice_data is None:
            # the current index is not a slice but the provided is a slice
            tensor = self.tousertensor()
            index = index._slice_data
            return Index(F.narrow_row(tensor, index.start, index.stop))
        else:
            # both self and index wrap a slice object, then return another
            # Index wrapping a slice
            start = self._slice_data.start
            index = index._slice_data
            return Index(slice(start + index.start, start + index.stop))

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
        return Index(F.scatter_row(tensor, index, value))

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
            return Index(new_items)
        else:
            tensor = self.tousertensor()
            tensor = F.cat((tensor, new_items), dim=0)
            return Index(tensor)

    def nonzero(self):
        """Return the nonzero positions"""
        tensor = self.tousertensor()
        mask = F.nonzero_1d(tensor != 0)
        return Index(mask)

    def has_nonzero(self):
        """Check if there is any nonzero value in this Index"""
        tensor = self.tousertensor()
        return F.sum(tensor, 0) > 0

def toindex(data):
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
    return data if isinstance(data, Index) else Index(data)

def zero_index(size):
    """Create a index with provided size initialized to zero

    Parameters
    ----------
    size: int
    """
    return Index(F.zeros((size,), dtype=F.int64, ctx=F.cpu()))

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
    F.scatter_row_inplace(old_to_new, unique_x, F.arange(0, len(unique_x)))
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
        def wrapper(self, *args):
            dic = getattr(self, cache)
            key = '%s-%s' % (prefix, '-'.join([str(a) for a in args]))
            if key not in dic:
                dic[key] = func(self, *args)
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
    remapped = np.array([invmap[x] for x in array])
    return uniques, invmap, remapped
