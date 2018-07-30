"""Utility module."""
from __future__ import absolute_import

from collections import Mapping
import dgl.backend as F
from dgl.backend import Tensor, SparseTensor

def is_id_tensor(u):
    return isinstance(u, Tensor) and F.isinteger(u) and len(F.shape(u)) == 1

def is_id_container(u):
    return isinstance(u, list)

def node_iter(n):
    if is_id_tensor(n):
        n = list(F.asnumpy(n))
    if is_id_container(n):
        for nn in n:
            yield nn
    else:
        yield n

def edge_iter(u, v):
    u_is_container = is_id_container(u)
    v_is_container = is_id_container(v)
    u_is_tensor = is_id_tensor(u)
    v_is_tensor = is_id_tensor(v)
    if u_is_tensor:
        u = F.asnumpy(u)
        u_is_tensor = False
        u_is_container = True
    if v_is_tensor:
        v = F.asnumpy(v)
        v_is_tensor = False
        v_is_container = True
    if u_is_container and v_is_container:
        # many-many
        for uu, vv in zip(u, v):
            yield uu, vv
    elif u_is_container and not v_is_container:
        # many-one
        for uu in u:
            yield uu, v
    elif not u_is_container and v_is_container:
        # one-many
        for vv in v:
            yield u, vv
    else:
        yield u, v

def homogeneous(x_list, type_x=None):
    type_x = type_x if type_x else type(x_list[0])
    return all(type(x) == type_x for x in x_list)

def convert_to_id_tensor(x):
    if is_id_container(x):
        assert homogeneous(x, int)
        return F.tensor(x)
    elif is_id_tensor(x):
        return x
    elif isinstance(x, int):
        x = F.tensor([x])
        return x
    else:
        raise TypeError('Error node: %s' % str(x))
    return None

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
