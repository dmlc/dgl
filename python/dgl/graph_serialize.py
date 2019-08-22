"""For Graph Serialization"""
from __future__ import absolute_import
from .graph import DGLGraph
from ._ffi.object import ObjectBase, register_object
# from ._ffi.base import c_array
from ._ffi.function import _init_api
from . import backend as F

_init_api("dgl.graph_serialize")


@register_object("graph_serialize.StorageMetaData")
class MetaData(ObjectBase):
    """MetaData Object
    attributes available:
      num_graph [int]: return numbers of graphs
      nodes_num_list [list of int]: return number of nodes for each graph
      edges_num_list [list of int]: return number of edges for each graph
      labels [dict of backend tensors]: return dict of labels
      graph_data [list of GraphData]: return list of GraphData Object
    """


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
            g.ndata[k] = F.zerocopy_from_dgl_ndarray(v.data)
        for k, v in edge_tensors_items:
            g.edata[k] = F.zerocopy_from_dgl_ndarray(v.data)
        return g


def save_graphs(filename, g_list, labels=None):
    """
    Save DGLGraphs to file
    :param filename: file to store DGLGraphs
    :param g_list: DGLGraph or list of DGLGraph
    :return:
    """
    if isinstance(g_list, DGLGraph):
        g_list = [g_list]
    if (labels is not None) and (len(labels) != 0):
        label_dict = dict()
        for key, value in labels.items():
            label_dict[key] = F.zerocopy_to_dgl_ndarray(value)
    else:
        label_dict = None
    gdata_list = [GraphData.create(g) for g in g_list]
    _CAPI_DGLSaveGraphs(filename, gdata_list, label_dict)


def load_graphs(filename, idx_list=None):
    """
    Load DGLGraphs from file
    :param filename: file to load DGLGraphs
    :param idx_list: list of index of graph to be loaded. If not specified, will load all graphs
    from file
    :return: list of immutable DGLGraphs, and labels stored in files (empty dict returned if no
    label stored)
    """
    assert isinstance(idx_list, list)
    if idx_list is None:
        idx_list = []
    metadata = _CAPI_DGLLoadGraphs(filename, idx_list, False)
    label_dict = {}
    for k, v in metadata.labels.items():
        label_dict[k] = F.zerocopy_from_dgl_ndarray(v.data)

    return [gdata.get_graph() for gdata in metadata.graph_data], label_dict


def load_labels(filename):
    """
    Load labels from file
    :param filename: file to load DGLGraphs
    :return: label_dict: dict of tensors
    """
    metadata = _CAPI_DGLLoadGraphs(filename, [], True)
    label_dict = {}
    for k, v in metadata.labels.items():
        label_dict[k] = F.zerocopy_from_dgl_ndarray(v.data)
    return label_dict
