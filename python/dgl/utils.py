"""Utility module."""
from __future__ import absolute_import

from collections import Mapping
import dgl.backend as F
from dgl.backend import Tensor, SparseTensor

def is_id_tensor(u):
    """Return whether the input is a supported id tensor."""
    return isinstance(u, Tensor) and F.isinteger(u) and len(F.shape(u)) == 1

def is_id_container(u):
    """Return whether the input is a supported id container."""
    return isinstance(u, list)

def node_iter(n):
    """Return an iterator that loops over the given nodes."""
    n = convert_to_id_container(n)
    for nn in n:
        yield nn

def edge_iter(u, v):
    """Return an iterator that loops over the given edges."""
    u = convert_to_id_container(u)
    v = convert_to_id_container(v)
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

def convert_to_id_container(x):
    """Convert the input to id container."""
    if is_id_container(x):
        return x
    elif is_id_tensor(x):
        return F.asnumpy(x)
    else:
        try:
            return [int(x)]
        except:
            raise TypeError('Error node: %s' % str(x))
    return None

def convert_to_id_tensor(x, ctx=None):
    """Convert the input to id tensor."""
    if is_id_container(x):
        ret = F.tensor(x, dtype=F.int64)
    elif is_id_tensor(x):
        ret = x
    else:
        try:
            ret = F.tensor([int(x)], dtype=F.int64)
        except:
            raise TypeError('Error node: %s' % str(x))
    ret = F.to_context(ret, ctx)
    return ret

class LazyDict(Mapping):
    """A readonly dictionary that does not materialize the storage."""
    def __init__(self, fn, keys):
        self._fn = fn
        self._keys = keys

    def keys(self):
        return self._keys

    def __getitem__(self, key):
        assert key in self._keys
        return self._fn(key)

    def __contains__(self, key):
        return key in self._keys

    def __iter__(self):
        for key in self._keys:
            yield key, self._fn(key)

    def __len__(self):
        return len(self._keys)

def build_relabel_map(x):
    """Relabel the input ids to continuous ids that starts from zero.

    Parameters
    ----------
    x : int, tensor or container
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
    x = convert_to_id_tensor(x)
    unique_x, _ = F.sort(F.unique(x))
    map_len = int(F.max(unique_x)) + 1
    old_to_new = F.zeros(map_len, dtype=F.int64)
    # TODO(minjie): should not directly use []
    old_to_new[unique_x] = F.astype(F.arange(len(unique_x)), F.int64)
    return unique_x, old_to_new

def edge_broadcasting(u, v):
    """Convert one-many and many-one edges to many-many."""
    if len(u) != len(v) and len(u) == 1:
        u = F.broadcast_to(u, v)
    elif len(u) != len(v) and len(v) == 1:
        v = F.broadcast_to(v, u)
    else:
        assert len(u) == len(v)
    return u, v
