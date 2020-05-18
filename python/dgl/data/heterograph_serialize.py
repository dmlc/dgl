"""For HeteroGraph Serialization"""
from __future__ import absolute_import
from ..heterograph import DGLHeteroGraph
from .._ffi.object import ObjectBase, register_object
from .._ffi.function import _init_api
from .. import backend as F
from ..container import convert_to_strmap

_init_api("dgl.data.heterograph_serialize")


def tensor_dict_to_ndarray_dict(tensor_dict):
    ndarray_dict = {}
    for key, value in tensor_dict.items():
        ndarray_dict[key] = F.zerocopy_to_dgl_ndarray(value)
    return convert_to_strmap(ndarray_dict)

def save_heterographs(filename, g_list):    
    if isinstance(g_list, DGLHeteroGraph):
        g_list = [g_list]
    gdata_list = [HeteroGraphData.create(g) for g in g_list]
    _CAPI_SaveHeteroGraphData(filename, gdata_list)

@register_object("heterograph_serialize.HeteroGraphData")
class HeteroGraphData(ObjectBase):

    def create(g: DGLHeteroGraph):
        edata_list = []
        ndata_list = []
        for etype in g.etypes:
            edata_list.append(tensor_dict_to_ndarray_dict(g.edges[etype].data))
        for ntype in g.ntypes:
            ndata_list.append(tensor_dict_to_ndarray_dict(g.nodes[ntype].data))
        return _CAPI_MakeHeteroGraphData(g._graph, ndata_list, edata_list, g.ntypes, g.etypes)

