from collections import defaultdict, MutableMapping
import dgl.backend as F
import dgl.utils as utils

class DGLNodeTensor:
    def __init__(self, idx, dat):
        """
        Parameters
        ----------
        idx : Tensor (N,)
        dat : Tensor (N, D, ...)
        """
        assert isinstance(idx, F.Tensor) and isinstance(dat, F.Tensor)
        assert F.isinteger(idx) and F.prod(idx >= 0)
        # TODO(gaiyu): device
        assert len(F.shape(idx)) == 1 and F.shape(idx)[0] == F.shape(dat)[0]
        self._idx = idx
        self._dat = dat

    def __getitem__(self, u):
        """
        Parameters
        ----------
        u : Tensor
        """
        assert isinstance(u, F.Tensor)
        idx = self._idx[u]
        dat = self._dat[u]
        return DGLNodeTensor(idx, dat)

    @property
    def idx(self):
        return self._idx

    @property
    def dat(self):
        return self._dat

class DGLEdgeTensor:
    def __init__(self, src, dst, dat):
        """
        Parameters
        ----------
        src : Tensor (N,)
        dst : Tensor (N,)
        dat : Tensor (N, D, ...)
        """
        assert all(isinstance(x, F.Tensor) for x in [src, dst, dat])
        assert F.isinteger(src) and F.prod(src >= 0)
        assert F.isinteger(dst) and F.prod(dst >= 0)
        assert len(F.shape(src)) == 1 and len(F.shape(dst)) == 1
        assert F.shape(src)[0] == F.shape(dat)[0] and F.shape(dst)[0] == F.shape(dat)[0]
        self._src = src
        self._dst = dst
        self._dat = dat

    @staticmethod
    def _complete(self, src=None, dst=None):
        """
        Parameters
        ----------
        src :
        dst :
        """
        if src:
            assert not dst
            x, self_x = src, self._src
        elif dst:
            assert not src
            x, self_x = dst, self._dst
        else:
            raise RuntimeError()

        y = F.expand_dims(x, 1) == F.expand_dims(self_x, 0)
        return F.nonzero(y)[0]

    @staticmethod
    def _index(self, idx):
        """
        Parameters
        ----------
        idx : Tensor (N,)
        """
        assert isinstance(idx, F.Tensor) and len(F.shape(idx)) == 1
        src = self._src[idx]
        dst = self._dst[idx]
        dat = self._dat[idx]
        return DGLEdgeTensor(src, dst, dat)

    def __getitem__(self, u, v):
        """
        Parameters
        ----------
        u : Tensor (N,) or slice(None, None, None)
        v : Tensor (N,) or slice(None, None, None)
        """
        if isinstance(u, F.Tensor):
            assert len(F.shape(u)) == 1
        if isinstance(v, F.Tensor):
            assert len(F.shape(v)) == 1
        if u == slice(None, None, None) and slice(None, None, None):
            return self
        elif u == slice(None, None, None) and isinstance(v, F.Tensor):
            return self._index(self._complete(v=v))
        elif isinstance(u, F.Tensor) and v == slice(None, None, None):
            return self._index(self._complete(u=u))
        elif isinstance(u, F.Tensor) and isinstance(v, F.Tensor):
            assert F.shape(u)[0] == F.shape(v)[0]
            return self._index((self._src == u) * (self._dst == v))
        else:
            raise RuntimeError()

    @property
    def src(self):
        return self._src

    @property
    def dst(self):
        return self._dst

    @property
    def dat(self):
        return self._dat

    @property
    def complete(self, u=None, v=None):
        idx = self._complete(u, v)
        return self._src[idx], self._dst[idx]

def concatenate(tensors):
    if all(isinstance(x, DGLNodeTensor) for x in tensors):
        idx = F.concatenate([x.idx for x in tensors])
        dat = F.concatenate([x.dat for x in tensors])
        return DGLNodeTensor(idx, dat)
    elif all(isinstance(x, DGLEdgeTensor) for x in tensors):
        src = F.concatenate([x.src for x in tensors])
        dst = F.concatenate([x.dst for x in tensors])
        dat = F.concatenate([x.dat for x in tensors])
        return DGLNodeTensor(src, dst, dat)
    else:
        raise RuntimeError()

class NodeDict(MutableMapping):
    def __init__(self):
        self._node = set()
        self._attrs = defaultdict(dict)

    @staticmethod
    def _deltensor(attr_value, u):
        """
        Parameters
        ----------
        u : Tensor
        """
        isin = F.isin(attr_value.idx, u)
        if F.sum(isin):
            if F.prod(isin):
                return DGLNodeTensor
            else:
                return attr_value[1 - isin]

    @staticmethod
    def _delitem(attrs, attr_name, u, uu):
        """
        Parameters
        ----------
        attrs :
        """
        attr_value = attrs[attr_name]
        deltensor = NodeDict._deltensor
        if isinstance(attr_value, dict):
            if isinstance(u, list):
                for x in u:
                    attr_value.pop(x, None)
            elif isinstance(u, F.Tensor):
                uu = uu if uu else map(F.item, u)
                for x in uu:
                    attr_value.pop(x, None)
            elif u == slice(None, None, None):
                assert not uu
                attrs[attr_name] = {}
            else:
                raise RuntimeError()
        elif isinstance(attr_value, DGLNodeTensor):
            if isinstance(u, list):
                uu = uu if uu else F.tensor(u) # TODO(gaiyu): device, dtype, shape
                attrs[attr_name] = deltensor(attr_value, uu)
            elif isinstance(u, Tensor):
                attrs[attr_name] = deltensor(attr_value, u)
            elif u == slice(None, None, None):
                assert not uu
                attrs[attr_name] = DGLNodeTensor
            else:
                raise RuntimeError()
        elif attr_value != DGLNodeTensor:
            raise RuntimeError()

    def __delitem__(self, u):
        """
        Parameters
        ----------
        """
        if isinstance(u, list):
            assert utils.homogeneous(u, int)
            if all(x not in self._adj for x in u):
                raise KeyError()
            self._node = self._node.difference(set(u))
            uu = None
        elif isinstance(u, F.Tensor):
            assert len(F.shape(u)) == 1 \
                and F.isinteger(u) \
                and F.prod(u >= 0) \
                and F.unpackable(u)
            uu = F.unpackable(u)
            self._node = self._node.difference(set(uu))
        elif u == slice(None, None, None):
            uu = None
        else:
            assert isinstance(u, int) and u >= 0
            self._node.remove(u)
            u, uu = [u], None

        for attr_name in self._attrs:
            self._delitem(self._attrs, attr_name, u, uu)

    def __getitem__(self, u):
        """
        Parameters
        ----------
        u :
        """
        if isinstance(u, list):
            assert utils.homogeneous(u, int) and all(x >= 0 for x in u)
            if all(x not in self._node for x in u):
                raise KeyError()
            uu = None
        elif isinstance(u, F.Tensor):
            assert len(F.shape(u)) == 1 and F.unpackable(u)
            uu = list(map(F.item, F.unpack(u)))
            assert utils.homogeneous(uu, int) and all(x >= 0 for x in uu)
            if all(x not in self._node for x in uu):
                raise KeyError()
        elif u == slice(None, None, None):
            uu = None
        elif isinstance(u, int):
            assert u >= 0
            if u not in self._node:
                raise KeyError()
            uu = None
        else:
            raise KeyError()
        return LazyNodeAttrDict(u, uu, self._node, self._attrs)

    def __iter__(self):
        return iter(self._node)

    def __len__(self):
        return len(self._node)

    @staticmethod
    def _settensor(attrs, attr_name, u, uu, attr_value):
        """
        Parameters
        ----------
        attrs :
        attr_name :
        u : Tensor or slice(None, None, None) or None
        uu : list or None
        attr_value : Tensor
        """
        x = attrs[attr_name]
        if isinstance(x, dict):
            if isinstance(u, list):
                for y, z in zip(u, F.unpack(attr_value)):
                    x[y] = z
            elif isinstance(u, F.Tensor):
                uu = uu if uu else map(F.item, F.unpack(u))
                assert F.unpackable(attr_value)
                for y, z in zip(uu, F.unpack(attr_value)):
                    x[y] = z
            elif u == slice(None, None, None):
                assert not uu
                attrs[attr_name] = self._dictize(attr_value)
            else:
                raise RuntimeError()
        elif isinstance(x, DGLNodeTensor):
            u = u if u else F.tensor(uu)
            isin = F.isin(x.idx, u)
            if F.sum(isin):
                if F.prod(isin):
                    attrs[attr_name] = DGLEdgeTensor(u, attr_value)
                else:
                    y = attr_value[1 - isin]
                    z = DGLNodeTensor(u, attr_value)
                    attrs[attr_name] = concatenate([y, z])
        elif x == DGLNodeTensor:
            attrs[attr_name] = DGLEdgeTensor(F.tensor(u), attr_value)

    @staticmethod
    def _setitem(node, attrs, attr_name, u, uu, attr_value):
        def valid(x):
            return isinstance(attr_value, F.Tensor) \
                and F.shape(attr_value)[0] == x \
                and F.unpackable(attr_value)

        settensor = NodeDict._settensor

        if isinstance(u, list):
            assert valid(len(u))
            settensor(attrs, attr_name, u, None, attr_value)
        elif isinstance(u, F.Tensor):
            assert valid(F.shape(u)[0])
            settensor(attrs, attr_name, u, uu, attr_value)
        elif u == slice(None, None, None):
            assert valid(len(node))
            settensor(attrs, attr_name, u, None, attr_value)
        elif isinstance(u, int):
            assert u >= 0
            if isinstance(attr_value, F.Tensor):
                assert valid(1)
                settensor(attrs, attr_name, [u], None, attr_value)
            else:
                attrs[attr_name][u] = attr_value
        else:
            raise RuntimeError()

    def __setitem__(self, u, attrs):
        """
        Parameters
        ----------
        u :
        attrs : dict
        """
        if isinstance(u, list):
            assert utils.homogeneous(u, int) and all(x >= 0 for x in u)
            self._node.update(u)
            uu = None
        elif isinstance(u, F.Tensor):
            assert len(F.shape(u)) == 1 and F.isinteger(u) and F.prod(u >= 0)
            uu = list(map(F.item, F.unpack(u)))
            self._node.update(uu)
        elif u == slice(None, None, None):
            uu = None
        elif isinstance(u, int):
            assert u >= 0
            self._node.add(u)
            uu = None
        else:
            raise RuntimeError()

        for attr_name, attr_value in attrs.items():
            self._setitem(self._node, self._attrs, attr_name, u, uu, attr_value)

    @staticmethod
    def _tensorize(attr_value):
        assert isinstance(attr_value, dict)
        if attr_value:
            assert F.packable([x for x in attr_value.values()])
            keys, values = map(list, zip(*attr_value.items()))
            assert utils.homoegeneous(keys, int) and all(x >= 0 for x in keys)
            assert F.packable(values)
            idx = F.tensor(keys) # TODO(gaiyu): device, dtype, shape
            dat = F.pack(values) # TODO(gaiyu): device, dtype, shape
            return DGLNodeTensor(idx, dat)
        else:
            return DGLNodeTensor

    def tensorize(self, attr_name):
        self._attrs[attr_name] = self._tensorize(self.attrs[attr_name])

    def istensorized(self, attr_name):
        attr_value = self._attrs[attr_name]
        return isinstance(attr_value, DGLNodeTensor) or attr_value == DGLNodeTensor

    @staticmethod
    def _dictize(attr_value):
        assert isinstance(attr_value, DGLNodeTensor)
        keys = map(F.item, F.unpack(attr_value.idx))
        values = F.unpack(attr_value.dat)
        return dict(zip(keys, values))

    def dictize(self, attr_name):
        self._attrs[attr_name] = self._dictize(attr_name)

    def isdictized(self, attr_name):
        return isinstance(self._attrs[attr_name], dict)

    def purge(self):
        predicate = lambda x: (isinstance(x, dict) and x) or isinstance(x, DGLNodeTensor)
        self._attrs = {k : v for k, v in self._attrs.items() if predicate(v)}

class LazyNodeAttrDict(MutableMapping):
    """
    `__iter__` and `__len__` are undefined for list.
    """
    def __init__(self, u, uu, node, attrs):
        self._u = u
        self._uu = uu
        self._node = node
        self._attrs = attrs

    def __delitem__(self, attr_name):
        NodeDict._delitem(self._attrs, self._u, attr_name)

    def __getitem__(self, attr_name):
        attr_value = self._attrs[attr_name]
        if isinstance(self._u, list):
            if all(x not in self._node for x in self._u):
                raise KeyError()
            if isinstance(attr_value, dict):
                y = [attr_value[x] for x in self._u]
                assert F.packable(y)
                return F.pack(y)
            elif isinstance(attr_value, DGLNodeTensor):
                uu = self._uu if self._uu else F.tensor(u)
                isin = F.isin(attr_value.idx, uu)
                return attr_value[isin].dat
            else:
                raise KeyError()
        elif isinstance(self._u, F.Tensor):
            uu = self._uu if self._uu else list(map(F.item, F.unpack(self._u)))
            if all(x not in self._node for x in uu):
                raise KeyError()
            if isinstance(attr_value, dict):
                y_list = [attr_value[x] for x in uu]
                assert F.packable(y_list)
                return F.pack(y_list)
            elif isinstance(attr_value, DGLNodeTensor):
                isin = F.isin(attr_value.idx, self._u)
                return attr_value[isin].dat
            else:
                raise KeyError()
        elif self._u == slice(None, None, None):
            assert not self._uu
            if isinstance(attr_value, dict) and attr_value:
                return NodeDict._tensorize(attr_value).dat
            elif isinstance(attr_value, DGLNodeTensor):
                return attr_value.dat
            else:
                raise KeyError()
        elif isinstance(self._u, int):
            assert not self._uu
            if isinstance(attr_value, dict):
                return attr_value[self._u]
            elif isinstance(attr_value, DGLNodeTensor):
                try: # TODO(gaiyu)
                    return attr_value.dat[self._u]
                except:
                    raise KeyError()
            else:
                raise KeyError()
        else:
            raise KeyError()

    def __iter__(self):
        if isinstance(self._u, int):
            for key, value in self._attrs.items():
                if (isinstance(value, dict) and self._u in value) or \
                    (isinstance(value, DGLNodeTensor) and F.sum(value.idx == self._u)):
                    yield key
        else:
            raise RuntimeError()

    def __len__(self):
        return sum(1 for x in self)

    def __setitem__(self, attr_name, attr_value):
        """
        Parameters
        ----------
        """
        setitem = NodeDict._setitem
        if isinstance(self._u, int):
            assert self._u in self._node
            if isinstance(attr_value, F.Tensor):
                setitem(self._node, self._attrs, attr_name, self._u, None, attr_value)
            else:
                self._attrs[self._u][attr_name] = attr_value
        else:
            if all(x not in self._node for x in self._u):
                raise KeyError()
            setitem(self._node, self._attrs, self._u, self._uu, attr_name)

    def materialized(self):
        attrs = {}
        for key in self._attrs:
            try:
                attrs[key] = self[key]
            except:
                KeyError()
        return attrs

class AdjOuterDict(MutableMapping):
    def __init__(self):
        self._adj = defaultdict(lambda: defaultdict(dict))
        self._attrs = defaultdict(dict)

    @staticmethod
    def _delitem(attrs, attr_name, u, uu):
        attr_value = attrs[attr_name]
        if isinstance(attr_value, dict):
            if u == slice(None, None, None):
                assert not uu
                attrs[attr_name] = {}
            else:
                uu = uu if uu else map(F.item, u)
                for x in uu:
                    attr_value.pop(x, None)
        elif isinstance(attr_value, DGLNodeTensor):
            if u == slice(None, None, None):
                assert not uu
                attrs[attr_name] = DGLEdgeTensor
            else:
                u = u if u else F.tensor(uu) # TODO(gaiyu): device, dtype, shape
                isin = F.isin(attr_value.idx, u)
                if F.sum(isin):
                    if F.prod(isin):
                        attrs[attr_name] = DGLEdgeTensor
                    else:
                        attrs[attr_name] = attr_value[1 - isin]
        elif attr_value != DGLEdgeTensor:
            raise RuntimeError()

    def __delitem__(self, u):
        if isinstance(u, list):
            assert utils.homogeneous(u, int) and all(x >= 0 for x in u)
            if all(x not in self._attrs for x in u):
                raise KeyError()
            for x in u:
                self._attrs.pop(x, None)
        elif isinstance(u, F.Tensor):
            pass

        for attr_name in self._attrs:
            self._delitem(self._attrs, attr_name, u, uu)

    def __iter__(self):
        return iter(self._adj)

    def __len__(self):
        return len(self._adj)

    def __getitem__(self, u):
        if isinstance(u, list):
            assert utils.homogeneous(u, int)
            if all(x not in self._adj for x in u):
                raise KeyError()
        elif isinstance(u, slice):
            assert u == slice(None, None, None)
        elif u not in self._adj:
            raise KeyError()
        return LazyAdjInnerDict(u, self._adj, self._attrs)

    def __setitem__(self, u, attrs):
        pass

    def uv(self, attr_name, u=None, v=None):
        if u:
            assert not v
            assert (isinstance(u, list) and utils.homogeneous(u, int)) or \
                (isinstance(u, F.Tensor) and F.isinteger(u) and len(F.shape(u)) == 1)
        elif v:
            assert not u
            assert (isinstance(v, list) and utils.homogeneous(v, int)) or \
                (isinstance(v, F.Tensor) and F.isinteger(v) and len(F.shape(v)) == 1)
        else:
            raise RuntimeError()

        attr_value = self._attrs[attr_name]
        if isinstance(attr_value, dict):
            if u:
                v = [[src, dst] for dst in attr_value.get(src, {}) for src in u]
            elif v:
                pass
        elif isinstance(attr_value, DGLEdgeTensor):
            u, v = attr_value._complete(u, v)

        return u, v

class LazyAdjInnerDict(MutableMapping):
    def __init__(self, u, uu, adj, attrs):
        self._u = u
        self._uu = uu
        self._adj = adj
        self._attrs = attrs

    def __getitem__(self, v):
        pass

    def __iter__(self):
        if isinstance(self._u, int):
            pass
        else:
            raise RuntimeError()

    def __len__(self):
        if not isinstance(self._u, [list, slice]):
            return len(self._adj[u])
        else:
            raise RuntimeError()

    def __setitem__(self, v, attr_dict):
        pass

class LazyEdgeAttrDict(MutableMapping):
    """dict: attr_name -> attr"""
    def __init__(self, u, v, uu, vv, adj, attrs):
        self._u = u
        self._v = v
        self._uu = uu
        self._vv = vv
        self._adj = adj
        self._attrs = attrs

    def __getitem__(self, attr_name):
        edge_iter = utils.edge_iter(self._u, self._v)
        attr_list = [self._outer_dict[uu, vv][attr_name] for uu, vv in edge_iter]
        return F.pack(attr_list) if F.packable(attr_list) else attr_list
    
    def __iter__(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __setitem__(self, attr_name, attr):
        if F.unpackable(attr):
            for [uu, vv], a in zip(utils.edge_iter(self._u, self._v), F.unpack(attr)):
                self._outer_dict[uu][vv][attr_name] = a
        else:
            for uu, vv in utils.edge_iter(self._u, self._v):
                self._outer_dict[uu][vv][attr_name] = attr
        
AdjInnerDict = dict
EdgeAttrDict = dict
