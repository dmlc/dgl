"""For HeteroGraph Serialization"""
from __future__ import absolute_import
from ..graph import DGLGraph
from .._ffi.object import ObjectBase, register_object
from .._ffi.function import _init_api
from .. import backend as F

_init_api("dgl.data.heterograph_serialize")

@register_object("heterograph_serialize.HeteroGraphData")
class HeteroGraphData(ObjectBase):
    pass