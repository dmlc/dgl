from collections import defaultdict, MutableMapping
from functools import reduce
import dgl.backend as F
from dgl.backend import Tensor

iscontainer = lambda x: isinstance(x, list)

class NodeDict(MutableMapping):
    """dict: node -> attr_dict"""
    def __init__(self):
        self._row_dict = defaultdict(dict) # dict: node -> attr_dict

    def __delitem__(self, n):
        n_is_container = iscontainer(n)
        n_is_tensor = isinstance(n, Tensor)
        if n_is_container:
            for nn in n:
                del self._row_dict[nn]
        if n_is_tensor:
            raise NotImplementedError()
        else:
            del self._row_dict[n]

    def __getitem__(self, n):
        n_is_container = iscontainer(n)
        n_is_tensor = isinstance(n, Tensor)
        if n_is_container:
            for nn in n:
                self._row_dict[nn]
            return LazyContainerNodeAttrDict(self, n)
        elif n_is_tensor:
            raise NotImplementedError()
        else:
            return self._row_dict[n]

    def __iter__(self):
        return iter(self._row_dict)

    def __len__(self):
        return len(self._row_dict)

    def __setitem__(self, n, attr_dict):
        n_is_container = iscontainer(n)
        n_is_tensor = isinstance(n, Tensor)
        if n_is_container:
            for nn in n:
                self._row_dict[nn] = {}
            lazy_dict = LazyContainerNodeAttrDict(self, n)
            lazy_dict.update(attr_dict)
        if n_is_tensor:
            raise NotImplementedError()
        else:
            self._row_dict[n] = attr_dict

class LazyContainerNodeAttrDict(MutableMapping):
    """dict: attr_name -> attr"""
    def __init__(self, node_dict, n):
        self._node_dict = node_dict
        self._n = n
        if len(n) == 1:
            self._keys = node_dict._row_dict[n[0]]
        else:
            keys_list = [set(node_dict._row_dict[nn]) for nn in n]
            self._keys = reduce(set.intersection, keys_list)

    def __delitem__(self, attr_name):
        for nn in self._n:
            del self._node_dict._row_dict[attr_name]

    def __getitem__(self, attr_name):
        attr_list = [self._node_dict._row_dict[nn][attr_name] for nn in self._n]
        if isinstance(items[0], Tensor):
            return F.pack(attr_list)
        else:
            return attr_list

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def __setitem__(self, attr_name, attr):
        attr_is_container = iscontainer(attr)
        attr_is_tensor = isinstance(attr, Tensor)
        if attr_is_container:
            for nn, a in zip(self._n, attr):
                self._node_dict._row_dict[nn][attr_name] = a
        elif attr_is_tensor:
            for nn, a in zip(self._n, F.unpack(attr)):
                self._node_dict._row_dict[nn][attr_name] = a
        else:
            for nn in self._n:
                self._node_dict._row_dict[nn][attr_name] = attr

class AdjOuterDict(MutableMapping):
    """dict: node -> adj inner dict"""
    def __init__(self):
        self._outer_dict = defaultdict(lambda: defaultdict(dict))

    def __delitem__(self, u):
        u_is_container = iscontainer(u)
        u_is_tensor = isinstance(u, Tensor)
        if u_is_container:
            for uu in u:
                del self._outer_dict[uu]
        elif u_is_tensor:
            raise NotImplementedError()
        else:
            del self._outer_dict[u]

    def __iter__(self):
        return iter(self._outer_dict)

    def __len__(self):
        return len(self._outer_dict)

    def __getitem__(self, u):
        u_is_container = iscontainer(u)
        u_is_tensor = isinstance(u, Tensor)
        if u_is_container:
            assert all(uu in self._outer_dict for uu in u)
        elif n_is_tensor:
            raise NotImplementedError()
        else:
            assert u in self._outer_dict
        return LazyAdjInnerDict(self, u, u_is_container, u_is_tensor)

    def __setitem__(self, u, inner_dict):
        u_is_container = iscontainer(n)
        u_is_tensor = isinstance(u, Tensor)
        if u_is_container:
            for uu in u:
                self._outer_dict[uu] = inner_dict
        elif u_is_tensor:
            raise NotImplementedError()
        else:
            self._outer_dict[u] = inner_dict

class LazyAdjInnerDict:
    def __init__(self, store, u, u_is_container, u_is_tensor):
        self._store = store
        self._u = u
        self.u_is_container = u_is_container
        self.u_is_tensor = u_is_tensor

    def __getitem__(self, v):
        v_is_container = iscontainer(v)
        v_is_tensor = isinstance(v, Tensor)
        if v_is_container:
            assert all(
        if v_is_tensor:
            raise NotImplementedError()
        else:
            return self._store._row_dict[self._n][v]

    def __iter__(self):

    def __len__(self):

    def __setitem__(self, v, attrs):
        v_is_container = iscontainer(v)
        v_is_tensor = isinstance(v, Tensor)
        if v_is_container:
            assert not self.u_is_tensor
            for vv in v:
                self._store._row_dict[self._n][vv] = {}
            attr_dict = LazyEdgeAttrDict(self._store, self._n, v)
            for name, attr in attrs.items():
                attr_dict[name] = attr
        elif v_is_tensor:
            assert self.u_is_tensor
            raise NotImplementedError()
        else:
            self._store._row_dict[self._n][v] = attrs

class LazyContainerEdgeAttrDict:
    """dict: attr_name -> attr"""
    pass
