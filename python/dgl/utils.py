"""Utility module."""
from __future__ import absolute_import

from collections import Mapping
from functools import wraps
import numpy as np

from . import backend as F
from .backend import Tensor, SparseTensor
from . import ndarray as nd

class Index(object):
    """Index class that can be easily converted to list/tensor."""
    def __init__(self, data):
        self._list_data = None  # a numpy type data
        self._user_tensor_data = dict()  # dictionary of user tensors
        self._dgl_tensor_data = None  # a dgl ndarray
        self._dispatch(data)

    def _dispatch(self, data):
        """Store data based on its type."""
        if isinstance(data, Tensor):
            if not (F.dtype(data) == F.int64 and len(F.shape(data)) == 1):
                raise ValueError('Index data must be 1D int64 vector, but got: %s' % str(data))
            self._user_tensor_data[F.get_context(data)] = data
        elif isinstance(data, nd.NDArray): 
            if not (data.dtype == 'int64' and len(data.shape) == 1):
                raise ValueError('Index data must be 1D int64 vector, but got: %s' % str(data))
            self._dgl_tensor_data = data
        else:
            try:
                self._list_data = np.array([int(data)]).astype(np.int64)
            except:
                try:
                    self._list_data = np.array(data).astype(np.int64)
                except:
                    raise ValueError('Error index data: %s' % str(data))
            self._user_tensor_data[nd.cpu()] = F.zerocopy_from_numpy(self._list_data)

    def tolist(self):
        """Convert to a python-list compatible object."""
        if self._list_data is None:
            if self._dgl_tensor_data is not None:
                self._list_data = self._dgl_tensor_data.asnumpy()
            else:
                data = self.tousertensor()
                self._list_data = F.zerocopy_to_numpy(data)
        return self._list_data

    def tousertensor(self, ctx=None):
        """Convert to user tensor (defined in `backend`)."""
        if ctx is None:
            ctx = nd.cpu()
        if len(self._user_tensor_data) == 0:
            # zero copy from dgl tensor
            dl = self._dgl_tensor_data.to_dlpack()
            self._user_tensor_data[nd.cpu()] = F.zerocopy_from_dlpack(dl)
        if ctx not in self._user_tensor_data:
            # copy from cpu to another device
            data = next(iter(self._user_tensor_data.values()))
            self._user_tensor_data[ctx] = F.to_context(data, ctx)
        return self._user_tensor_data[ctx]

    def todgltensor(self):
        """Convert to dgl.NDArray."""
        if self._dgl_tensor_data is None:
            # zero copy from user tensor
            tsor = self.tousertensor()
            dl = F.zerocopy_to_dlpack(tsor)
            self._dgl_tensor_data = nd.from_dlpack(dl)
        return self._dgl_tensor_data

    def __iter__(self):
        return iter(self.tolist())

    def __len__(self):
        if self._list_data is not None:
            return len(self._list_data)
        elif len(self._user_tensor_data) > 0:
            data = next(iter(self._user_tensor_data.values()))
            return len(data)
        else:
            return len(self._dgl_tensor_data)

    def __getitem__(self, i):
        return self.tolist()[i]

def toindex(x):
    return x if isinstance(x, Index) else Index(x)

def node_iter(n):
    """Return an iterator that loops over the given nodes.

    Parameters
    ----------
    n : iterable
        The node ids.
    """
    return iter(n)

def edge_iter(u, v):
    """Return an iterator that loops over the given edges.

    Parameters
    ----------
    u : iterable
        The src ids.
    v : iterable
        The dst ids.
    """
    if len(u) == len(v):
        # many-many
        for uu, vv in zip(u, v):
            yield uu, vv
    elif len(v) == 1:
        # many-one
        for uu in u:
            yield uu, v[0]
    elif len(u) == 1:
        # one-many
        for vv in v:
            yield u[0], vv
    else:
        raise ValueError('Error edges:', u, v)

def edge_broadcasting(u, v):
    """Convert one-many and many-one edges to many-many.

    Parameters
    ----------
    u : Index
        The src id(s)
    v : Index
        The dst id(s)

    Returns
    -------
    uu : Index
        The src id(s) after broadcasting
    vv : Index
        The dst id(s) after broadcasting
    """
    if len(u) != len(v) and len(u) == 1:
        u = toindex(F.broadcast_to(u.tousertensor(), v.tousertensor()))
    elif len(u) != len(v) and len(v) == 1:
        v = toindex(F.broadcast_to(v.tousertensor(), u.tousertensor()))
    else:
        assert len(u) == len(v)
    return u, v

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

def build_relabel_map(x):
    """Relabel the input ids to continuous ids that starts from zero.

    Parameters
    ----------
    x : Index
      The input ids.

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
    unique_x, _ = F.sort(F.unique(x))
    map_len = int(F.max(unique_x)) + 1
    old_to_new = F.zeros(map_len, dtype=F.int64)
    # TODO(minjie): should not directly use []
    old_to_new[unique_x] = F.astype(F.arange(len(unique_x)), F.int64)
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

def pack2(a, b):
    if a is None:
        return b
    elif b is None:
        return a
    else:
        if isinstance(a, dict):
            return {k: F.pack([a[k], b[k]]) for k in a}
        else:
            return F.pack([a, b])

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
        idx_ctx = index.tousertensor(F.get_context(val))
        new_dict[key] = F.gather_row(val, idx_ctx)
    return new_dict
