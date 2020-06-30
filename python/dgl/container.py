"""Container data structures used in DGL runtime.
reference: tvm/python/tvm/collections.py
"""
from __future__ import absolute_import as _abs
from ._ffi.object import ObjectBase, register_object
from ._ffi.object_generic import convert_to_object
from . import _api_internal


@register_object
class List(ObjectBase):
    """List container of DGL.

    You do not need to create List explicitly.
    Normally python list and tuple will be converted automatically
    to List during dgl function call.
    You may get List in return values of DGL function call.
    """

    def __getitem__(self, i):
        if isinstance(i, slice):
            start = i.start if i.start is not None else 0
            stop = i.stop if i.stop is not None else len(self)
            step = i.step if i.step is not None else 1
            if start < 0:
                start += len(self)
            if stop < 0:
                stop += len(self)
            return [self[idx] for idx in range(start, stop, step)]

        if i < -len(self) or i >= len(self):
            raise IndexError("List index out of range. List size: {}, got index {}"
                             .format(len(self), i))
        if i < 0:
            i += len(self)
        ret = _api_internal._ListGetItem(self, i)
        if isinstance(ret, Value):
            ret = ret.data
        return ret

    def __len__(self):
        return _api_internal._ListSize(self)


@register_object
class Map(ObjectBase):
    """Map container of DGL.

    You do not need to create Map explicitly.
    Normally python dict will be converted automaticall to Map during dgl function call.
    You can use convert to create a dict[ObjectBase-> ObjectBase] into a Map
    """

    def __getitem__(self, k):
        return _api_internal._MapGetItem(self, k)

    def __contains__(self, k):
        return _api_internal._MapCount(self, k) != 0

    def items(self):
        """Get the items from the map"""
        akvs = _api_internal._MapItems(self)
        return [(akvs[i], akvs[i+1]) for i in range(0, len(akvs), 2)]

    def __len__(self):
        return _api_internal._MapSize(self)


@register_object
class StrMap(Map):
    """A special map container that has str as key.

    You can use convert to create a dict[str->ObjectBase] into a Map.
    """

    def items(self):
        """Get the items from the map"""
        akvs = _api_internal._MapItems(self)
        return [(akvs[i], akvs[i+1]) for i in range(0, len(akvs), 2)]


@register_object
class Value(ObjectBase):
    """Object wrapper for various values."""
    @property
    def data(self):
        """Return the value data."""
        return _api_internal._ValueGet(self)


def convert_to_strmap(value):
    """Convert a python dictionary to a dgl.contrainer.StrMap"""
    assert isinstance(value, dict), "Only support dict"
    if len(value) == 0:
        return _api_internal._EmptyStrMap()
    else:
        return convert_to_object(value)
