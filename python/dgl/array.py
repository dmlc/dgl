from collections import MutableMapping
import dgl.backend as F

class DGLArray(MutableMapping):
    def __init__(self):
        pass

    def __delitem__(self, key, value):
        raise NotImplementedError()

    def __getitem__(self, key):
        """
        If the key is an DGLArray of identical length, this function performs a
        logical filter: i.e. it subselects all the elements in this array
        where the corresponding value in the other array evaluates to true.
        If the key is an integer this returns a single row of
        the DGLArray. If the key is a slice, this returns an DGLArray with the
        sliced rows. See the Turi Create User Guide for usage examples.
        """
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __setitem__(self, key, value):
        raise NotImplementedError()

class DGLDenseArray(DGLArray):
    def __init__(self, data, applicable=None):
        """
        Parameters
        ----------
        data : list or tensor
        """
        if type(data) is list:
            raise NotImplementedError()
        elif isinstance(data, F.Tensor):
            self._data = data
            if applicable is None:
                self._applicable = F.ones(F.shape(data)[0], dtype=F.bool) # TODO: device
            else:
                assert isinstance(applicable, F.Tensor)
                assert F.device(applicable) == F.device(data)
                assert F.isboolean(applicable)
                a_shape = F.shape(applicable)
                assert len(a_shape) == 1
                assert a_shape[0] == F.shape(data)[0]
                self._applicable = applicable

    def __getitem__(self, key):
        """
        If the key is an DGLDenseArray of identical length, this function performs a
        logical filter: i.e. it subselects all the elements in this array
        where the corresponding value in the other array evaluates to true.
        If the key is an integer this returns a single row of
        the DGLArray. If the key is a slice, this returns an DGLArray with the
        sliced rows. See the Turi Create User Guide for usage examples.
        """
        if type(key) is DGLDenseArray:
            if type(key._data) is list:
                raise NotImplementedError()
            elif type(key._data) is F.Tensor:
                if type(self._data) is F.Tensor:
                    shape = F.shape(key._data)
                    assert len(shape) == 1
                    assert shape[0] == F.shape(self._data)[0]
                    assert F.dtype(key._data) is F.bool
                    data = self._data[key._data]
                    return DGLDenseArray(data)
                else:
                    raise NotImplementedError()
            else:
                raise RuntimeError()
        elif type(key) is int:
            return self._data[key]
        elif type(key) is slice:
            raise NotImplementedError()
        else:
            raise RuntimeError()

    def __iter__(self):
        return iter(range(len(self)))

    def __len__(self):
        if type(self._data) is F.Tensor:
            return F.shape(self._data)[0]
        elif type(self._data) is list:
            return len(self._data)
        else:
            raise RuntimeError()

    def __setitem__(self, key, value):
        if type(key) is int:
            if type(self._data) is list:
                raise NotImplementedError()
            elif type(self._data) is F.Tensor:
                assert isinstance(value, F.Tensor)
                assert F.device(value) == F.device(self._data)
                assert F.dtype(value) == F.dtype(self._data)
                # TODO(gaiyu): shape
                x = []
                if key > 0:
                    x.append(self._data[:key])
                x.append(F.expand_dims(value, 0))
                if key < F.shape(self._data)[0] - 1:
                    x.append(self._data[key + 1:])
                self._data = F.concatenate(x)
            else:
                raise RuntimeError()
        elif type(key) is DGLDenseArray:
            shape = F.shape(key._data)
            assert len(shape) == 1
            assert shape[0] == F.shape(self._data)[0]
            assert F.isboolean(key._data)
            data = self._data[key._data]
        elif type(key) is DGLSparseArray:
            raise NotImplementedError()
        else:
            raise RuntimeError()

    def _listize(self):
        raise NotImplementedError()

    def _tensorize(self):
        raise NotImplementedError()

    def append(self, other):
        assert type(other, DGLDenseArray)
        if self.shape is None:
            return other
        elif other.shape is None:
            return self
        else:
            assert self.shape[1:] == other.shape[1:]
            data = F.concatenate([self.data, other.data])
            return DGLDenseArray(data)

    @property
    def applicable(self):
        return self._applicable

    @property
    def data(self):
        return self._data

    def dropna(self):
        if type(self._data) is list:
            raise NotImplementedError()
        elif isinstance(self._data, F.Tensor):
            data = F.index_by_bool(self._data, self._applicable)
            return DGLDenseArray(data)
        else:
            raise RuntimeError()

class DGLSparseArray(DGLArray):
    def __init__(self):
        raise NotImplementedError()
