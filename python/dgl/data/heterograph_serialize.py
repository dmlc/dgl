"""For HeteroGraph Serialization"""
from __future__ import absolute_import
from ..heterograph import DGLHeteroGraph
from ..frame import Frame, FrameRef
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


def load_heterographs(filename):
    metadata = _CAPI_LoadDGLGraphFiles(filename, [], False)
    return metadata


@register_object("heterograph_serialize.HeteroGraphData")
class HeteroGraphData(ObjectBase):

    @staticmethod
    def create(g: DGLHeteroGraph):
        edata_list = []
        ndata_list = []
        for etype in g.etypes:
            edata_list.append(tensor_dict_to_ndarray_dict(g.edges[etype].data))
        for ntype in g.ntypes:
            ndata_list.append(tensor_dict_to_ndarray_dict(g.nodes[ntype].data))
        return _CAPI_MakeHeteroGraphData(g._graph, ndata_list, edata_list, g.ntypes, g.etypes)

    def get_graph(self):
        ntensor_list = list(_CAPI_GetNDataFromHeteroGraphData(self))
        etensor_list = list(_CAPI_GetEDataFromHeteroGraphData(self))
        ntype_names = list(_CAPI_GetNtypesFromHeteroGraphData(self))
        etype_names = list(_CAPI_GetEtypesFromHeteroGraphData(self))
        gidx = _CAPI_GetGindexFromHeteroGraphData(self)
        nframes = []
        eframes = []
        for ntensor in ntensor_list:
            ndict = {ntensor[0]: ntensor[1] for i in range(0, len(ntensor), 2)}
            nframes.append(FrameRef(Frame(ndict)))
        
        for etensor in etensor_list:
            edict = {etensor[0]: etensor[1] for i in range(0, len(etensor), 2)}
            eframes.append(FrameRef(Frame(edict)))
        
        return DGLHeteroGraph(gidx, ntype_names, etype_names, nframes, eframes)

    #     return DGLHeteroGraph()
