import re
from collections import defaultdict
import torch

from ..storages import TensorStorage

_storage_pattern = re.compile('__(.*)_storages__')

class SamplerMeta(type):
    """``SamplerMeta`` is a metaclass that looks for the methods look like
    ``__xxx_storages__`` from the classes it creates.  They indicate where the prefetched
    features will be stored into.  For each method named ``__foo_storages__``,
    ``SamplerMeta`` will create a method with the following signature:

    .. code:: python

       def add_foo(self, name: str, storage: Tensor | FeatureStorage):
           pass.

    See also
    --------
    Sampler
    """
    def __new__(cls, clsname, bases, attrs):
        new_func_names = []
        for key, value in list(attrs.items()):
            match = _storage_pattern.fullmatch(key)
            if not (match is not None and callable(value)):
                continue

            storage_key = match.group(1)
            def _add_func(self, name, storage, _storage_key=storage_key):
                # _storages is initialized in Sampler class.
                if torch.is_tensor(storage):
                    self._storages[_storage_key][name] = TensorStorage(storage)
                else:
                    self._storages[_storage_key][name] = storage

            new_func_name = f'add_{storage_key}'
            _add_func.__doc__ = \
                f"""Adds a feature storage to prefetch from for ``{storage_key}``.

                The prefetched features will be stored into the place indicated by the method
                :attr:`__{storage_key}_storages__` with the name given in argument
                :attr:`name`.

                Parameters
                ----------
                name : str
                    The feature name.
                storage : Tensor or FeatureStorage or dict
                    The tensor or the :class:`FeatureStorage` object to prefetch features
                    from.

                    If a dictionary is given, the dictionary's keys will be node/edge
                    types and the values will be tensors or :class:`FeatureStorage`
                    objects.  Whether the keys shall be node types or edge types depends
                    on the return value of the sampler's :attr:`__{storage_key}_storages__`
                    method.
                """
            _add_func.__name__ = new_func_name
            new_func_names.append(new_func_name)
            attrs[new_func_name] = _add_func

        # The correct values for __qualname__ and __module__ of the new functions depend
        # on the class *after creation*, so we first create the class before updating.
        newcls = super().__new__(cls, clsname, bases, attrs)
        for new_func_name in new_func_names:
            func = getattr(newcls, new_func_name)
            func.__qualname__ = f'{newcls.__qualname__}.{new_func_name}'
            func.__module__ = newcls.__module__

        return newcls


class Sampler(object, metaclass=SamplerMeta):
    """Sampler base class. All samplers used by :class:`NodeDataLoader` and
    :class:`EdgeDataLoader` inherits from this class.

    All subclasses should implement the :attr:`sample` method, which takes in 

    When feature prefetching is necessary for a sampler, one could define a method
    named ``__foo_storages__`` where ``foo`` could be replaced by any name you want.
    The method should take in a single argument which is returned by its
    :attr:`sample` method, and returns a list of either :class:`HeteroNodeDataView`
    or :class:`HeteroEdgeDataView` objects that are part of the feature storage
    of the sampled subgraphs or blocks.  DGL will automatically generate an
    :attr:`add_foo` method that takes in a name and a tensor or :class:`FeatureStorage`
    object, which can be called later to register the actual feature storage to prefetch
    from.
    """
    def __init__(self):
        self._storages = defaultdict(dict)
        super().__init__()
