"""For Graph Serialization"""
from __future__ import absolute_import
from .graph import DGLGraph
from ._ffi.object import ObjectBase, register_object
# from ._ffi.base import c_array
from . import _api_internal
from ._ffi.function import _init_api
from . import backend as F

_init_api("dgl.graph_serialize")


@register_object("graph_serialize.GraphData")
class GraphData(ObjectBase):
    """GraphData Object"""

    @staticmethod
    def create(g: DGLGraph):
        """Create GraphData"""
        ghandle = g._graph
        if len(g.ndata) != 0:
            node_tensors = dict()
            for key, value in g.ndata.items():
                node_tensors[key] = F.zerocopy_to_dgl_ndarray(value)
        else:
            node_tensors = None

        if len(g.edata) != 0:
            edge_tensors = dict()
            for key, value in g.edata.items():
                edge_tensors[key] = F.zerocopy_to_dgl_ndarray(value)
        else:
            edge_tensors = None

        return _CAPI_MakeGraphData(ghandle, node_tensors, edge_tensors)

    def get_graph(self):
        """Get DGLGraph from GraphData"""
        ghandle = _CAPI_GDataGraphHandle(self)
        g = DGLGraph(graph_data=ghandle, readonly=True)
        node_tensors_items = _CAPI_GDataNodeTensors(self).items()
        edge_tensors_items = _CAPI_GDataEdgeTensors(self).items()
        for k, v in node_tensors_items:
            g.ndata[k] = F.zerocopy_from_dgl_ndarray(_api_internal._ValueGet(v))
        for k, v in edge_tensors_items:
            g.edata[k] = F.zerocopy_from_dgl_ndarray(_api_internal._ValueGet(v))
        return g


def save_graphs(filename, g_list):
    """
    Save DGLGraphs to file
    :param filename: file to store DGLGraphs
    :param g_list: DGLGraph or list of DGLGraph
    :return:
    """
    if isinstance(g_list, DGLGraph):
        g_list = [g_list]
    gdata_list = [GraphData.create(g) for g in g_list]
    _CAPI_DGLSaveGraphs(filename, gdata_list)


def load_graphs(filename, idx_list=None):
    """
    Load DGLGraphs from file
    :param filename: file to load DGLGraphs
    :param idx_list: list of index of graph to be loaded. If not specified, will load all graphs
    from file
    :return: list of immutable DGLGraphs
    """
    assert isinstance(idx_list, list)
    if idx_list is None:
        idx_list = []
    gdata_list = _CAPI_DGLLoadGraphs(filename, idx_list)
    return [gdata.get_graph() for gdata in gdata_list]
