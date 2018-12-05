"""Utility module."""
from __future__ import absolute_import, division

from collections import Mapping, Iterable
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
        self._pydata = None   # a numpy type data or a slice
        self._user_tensor_data = dict()  # dictionary of user tensors
        self._dgl_tensor_data = None  # a dgl ndarray
        self._dispatch(data)

    def __iter__(self):
        for i in self.tonumpy():
            yield int(i)

    def __len__(self):
        if self._pydata is not None and isinstance(self._pydata, slice):
            slc = self._pydata
            if slc.step is None:
                return slc.stop - slc.start
            else:
                return (slc.stop - slc.start) // slc.step
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
            if not (F.dtype(data) == F.int64):
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
            self._pydata = data
        else:
            try:
                self._pydata = np.array([int(data)]).astype(np.int64)
            except:
                try:
                    data = np.array(data).astype(np.int64)
                    if data.ndim != 1:
                        raise DGLError('Index data must be 1D int64 vector,'
                                       ' but got: %s' % str(data))
                    self._pydata = data
                except:
                    raise DGLError('Error index data: %s' % str(data))
            self._user_tensor_data[F.cpu()] = F.zerocopy_from_numpy(self._pydata)

    def tonumpy(self):
        """Convert to a numpy ndarray."""
        if self._pydata is None:
            if self._dgl_tensor_data is not None:
                self._pydata = self._dgl_tensor_data.asnumpy()
            else:
                data = self.tousertensor()
                self._pydata = F.zerocopy_to_numpy(data)
        elif isinstance(self._pydata, slice):
            # convert it to numpy array
            slc = self._pydata
            self._pydata = np.arange(slc.start, slc.stop, slc.step).astype(np.int64)
        return self._pydata

    def tousertensor(self, ctx=None):
        """Convert to user tensor (defined in `backend`)."""
        if ctx is None:
            ctx = F.cpu()
        if len(self._user_tensor_data) == 0:
            if self._dgl_tensor_data is not None:
                # zero copy from dgl tensor
                dl = self._dgl_tensor_data.to_dlpack()
                self._user_tensor_data[F.cpu()] = F.zerocopy_from_dlpack(dl)
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
            dl = F.zerocopy_to_dlpack(tsor)
            self._dgl_tensor_data = nd.from_dlpack(dl)
        return self._dgl_tensor_data

    def is_slice(self, start, stop, step=None):
        return (isinstance(self._pydata, slice)
                and self._pydata == slice(start, stop, step))

    def __getstate__(self):
        return self.tousertensor()

    def __setstate__(self, state):
        self._initialize_data(state)

def toindex(x):
    return x if isinstance(x, Index) else Index(x)

class LazyDict(Mapping):
    """A readonly dictionary that does not materialize the storage."""
    def __init__(self, fn, keys):
        self._fn = fn
        self._keys = keys

    def __getitem__(self, key):
        if not key in self._keys:
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
       If there are duplicate keys, early keys have priority over latter ones
    """
    def __init__(self, *dict_like_list):
        self._dict_like_list = dict_like_list
        self._keys = set()
        for d in dict_like_list:
            self._keys.update(d.keys())

    def keys(self):
        return self._keys

    def __getitem__(self, key):
        for d in self._dict_like_list:
            if key in d:
                return d[key]
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

def build_relabel_map(x, sorted=False):
    """Relabel the input ids to continuous ids that starts from zero.

    Ids are assigned new ids according to their ascending order.

    Examples
    --------
    >>> x = [1, 5, 3, 6]
    >>> n2o, o2n = build_relabel_map(x)
    >>> n2o
    [1, 3, 5, 6]
    >>> o2n
    [n/a, 0, n/a, 2, n/a, 3, 4]

    "n/a" will be filled with 0

    Parameters
    ----------
    x : Index
        The input ids.
    sorted : bool, default=False
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
    if not sorted:
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

    def get(self, ctx):
        if not ctx in self._ctx_dict:
            self._ctx_dict[ctx] = self._generator(ctx)
        return self._ctx_dict[ctx]

def ctx_cached_member(func):
    """Convenient class member function wrapper to cache the function result.

    The wrapped function must only have two arguments: `self` and `ctx`. The former is the
    class object and the later is the context. It will check whether the class object is
    freezed (by checking the `_freeze` member). If yes, it caches the function result in
    the field prefixed by '_CACHED_' before the function name.
    """
    cache_name = '_CACHED_' + func.__name__
    @wraps(func)
    def wrapper(self, ctx):
        if self._freeze:
            # cache
            if getattr(self, cache_name, None) is None:
                bind_func = lambda _ctx : func(self, _ctx)
                setattr(self, cache_name, CtxCachedObject(bind_func))
            return getattr(self, cache_name).get(ctx)
        else:
            return func(self, ctx)
    return wrapper

def cached_member(func):
    cache_name = '_CACHED_' + func.__name__
    @wraps(func)
    def wrapper(self):
        if self._freeze:
            # cache
            if getattr(self, cache_name, None) is None:
                setattr(self, cache_name, func(self))
            return getattr(self, cache_name)
        else:
            return func(self)
    return wrapper

def is_dict_like(obj):
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
