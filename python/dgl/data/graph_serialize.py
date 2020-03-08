"""For Graph Serialization"""
from __future__ import absolute_import
from ..graph import DGLGraph
from .._ffi.object import ObjectBase, register_object
from .._ffi.function import _init_api
from .. import backend as F

_init_api("dgl.data.graph_serialize")

__all__ = ['save_graphs', "load_graphs", "load_labels"]

@register_object("graph_serialize.StorageMetaData")
class StorageMetaData(ObjectBase):
    """StorageMetaData Object
    attributes available:
      num_graph [int]: return numbers of graphs
      nodes_num_list Value of NDArray: return number of nodes for each graph
      edges_num_list Value of NDArray: return number of edges for each graph
      labels [dict of backend tensors]: return dict of labels
      graph_data [list of GraphData]: return list of GraphData Object
    """


@register_object("graph_serialize.GraphData")
class GraphData(ObjectBase):
    """GraphData Object"""

    @staticmethod
    def create(g: DGLGraph):
        """Create GraphData"""
        # TODO(zihao): support serialize batched graph in the future.
        assert g.batch_size == 1, "Batched DGLGraph is not supported for serialization"
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
    r"""
    Save DGLGraphs and graph labels to file

    Parameters
    ----------
    filename : str
        File name to store DGLGraphs. 
    g_list: list
        DGLGraph or list of DGLGraph
    labels: dict (Default: None)
        labels should be dict of tensors/ndarray, with str as keys

    Examples
    ----------
    >>> import dgl
    >>> import torch as th

    Create :code:`DGLGraph` objects and initialize node and edge features.

    >>> g1 = dgl.DGLGraph()
    >>> g1.add_nodes(3)
    >>> g1.add_edges([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
    >>> g1.ndata["e"] = th.ones(3, 5)
    >>> g2 = dgl.DGLGraph()
    >>> g2.add_nodes(3)
    >>> g2.add_edges([0, 1, 2], [1, 2, 1])
    >>> g2.edata["e"] = th.ones(3, 4)

    Save Graphs into file

    >>> from dgl.data.utils import save_graphs
    >>> graph_labels = {"glabel": th.tensor([0, 1])}
    >>> save_graphs("./data.bin", [g1, g2], graph_labels)

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

    Parameters
    ----------
    filename: str
        filename to load DGLGraphs
    idx_list: list of int
        list of index of graph to be loaded. If not specified, will
        load all graphs from file

    Returns
    ----------
    graph_list: list of immutable DGLGraphs
    labels: dict of labels stored in file (empty dict returned if no
    label stored)

    Examples
    ----------
    Following the example in save_graphs.

    >>> from dgl.data.utils import load_graphs
    >>> glist, label_dict = load_graphs("./data.bin") # glist will be [g1, g2]
    >>> glist, label_dict = load_graphs("./data.bin", [0]) # glist will be [g1]

    """
    if idx_list is None:
        idx_list = []
    assert isinstance(idx_list, list)
    metadata = _CAPI_DGLLoadGraphs(filename, idx_list, False)
    label_dict = {}
    for k, v in metadata.labels.items():
        label_dict[k] = F.zerocopy_from_dgl_ndarray(v.data)

    return [gdata.get_graph() for gdata in metadata.graph_data], label_dict


def load_labels(filename):
    """
    Load label dict from file

    Parameters
    ----------
    filename: str
        filename to load DGLGraphs

    Returns
    ----------
    labels: dict
        dict of labels stored in file (empty dict returned if no
        label stored)

    Examples
    ----------
    Following the example in save_graphs.

    >>> from dgl.data.utils import load_labels
    >>> label_dict = load_graphs("./data.bin")

    """
    metadata = _CAPI_DGLLoadGraphs(filename, [], True)
    label_dict = {}
    for k, v in metadata.labels.items():
        label_dict[k] = F.zerocopy_from_dgl_ndarray(v.data)
    return label_dict
