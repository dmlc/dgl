"""Common implementation of Object generic related logic"""
# pylint: disable=unused-import
from __future__ import absolute_import

from numbers import Integral, Number

from .. import _api_internal
from .base import string_types

# Object base class
_CLASS_OBJECT_BASE = None


def _set_class_object_base(cls):
    global _CLASS_OBJECT_BASE
    _CLASS_OBJECT_BASE = cls


class ObjectGeneric(object):
    """Base class for all classes that can be converted to object."""

    def asobject(self):
        """Convert value to object"""
        raise NotImplementedError()


def convert_to_object(value):
    """Convert a python value to corresponding object type.

    Parameters
    ----------
    value : str
        The value to be inspected.

    Returns
    -------
    object : Object
        The corresponding object value.
    """
    if isinstance(value, _CLASS_OBJECT_BASE):
        return value
    if isinstance(value, (list, tuple)):
        value = [convert_to_object(x) for x in value]
        return _api_internal._List(*value)
    if isinstance(value, dict):
        vlist = []
        for item in value.items():
            if not isinstance(item[0], _CLASS_OBJECT_BASE) and not isinstance(
                item[0], string_types
            ):
                raise ValueError(
                    "key of map must already been a container type"
                )
            vlist.append(item[0])
            vlist.append(convert_to_object(item[1]))
        return _api_internal._Map(*vlist)
    if isinstance(value, ObjectGeneric):
        return value.asobject()
    return _api_internal._Value(value)
