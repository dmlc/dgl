"""For Graph Serialization"""
from __future__ import absolute_import
import os
from ..base import dgl_warning, DGLError
from ..heterograph import DGLHeteroGraph
from .._ffi.object import ObjectBase, register_object
from .._ffi.function import _init_api
from .. import backend as F
from .heterograph_serialize import save_heterographs

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


def is_local_path(filepath):
    return not (filepath.startswith("hdfs://") or
                filepath.startswith("viewfs://") or
                filepath.startswith("s3://"))


def check_local_file_exists(filename):
    if is_local_path(filename) and not os.path.exists(filename):
        raise DGLError("File {} does not exist.".format(filename))

@register_object("graph_serialize.GraphData")
class GraphData(ObjectBase):
    """GraphData Object"""

    @staticmethod
    def create(g):
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
        """Get DGLHeteroGraph from GraphData"""
        ghandle = _CAPI_GDataGraphHandle(self)
        hgi =_CAPI_DGLAsHeteroGraph(ghandle)
        g = DGLHeteroGraph(hgi, ['_U'], ['_E'])
        node_tensors_items = _CAPI_GDataNodeTensors(self).items()
        edge_tensors_items = _CAPI_GDataEdgeTensors(self).items()
        for k, v in node_tensors_items:
            g.ndata[k] = F.zerocopy_from_dgl_ndarray(v)
        for k, v in edge_tensors_items:
            g.edata[k] = F.zerocopy_from_dgl_ndarray(v)
        return g


def save_graphs(filename, g_list, labels=None):
    r"""Save graphs and optionally their labels to file.

    Besides saving to local files, DGL supports writing the graphs directly
    to S3 (by providing a ``"s3://..."`` path) or to HDFS (by providing
    ``"hdfs://..."`` a path).

    The function saves both the graph structure and node/edge features to file
    in DGL's own binary format. For graph-level features, pass them via
    the :attr:`labels` argument.

    Parameters
    ----------
    filename : str
        The file name to store the graphs and labels.
    g_list: list
        The graphs to be saved.
    labels: dict[str, Tensor]
        labels should be dict of tensors, with str as keys

    Examples
    ----------
    >>> import dgl
    >>> import torch as th

    Create :class:`DGLGraph` objects and initialize node
    and edge features.

    >>> g1 = dgl.graph(([0, 1, 2], [1, 2, 3]))
    >>> g2 = dgl.graph(([0, 2], [2, 3]))
    >>> g2.edata["e"] = th.ones(2, 4)

    Save Graphs into file

    >>> from dgl.data.utils import save_graphs
    >>> graph_labels = {"glabel": th.tensor([0, 1])}
    >>> save_graphs("./data.bin", [g1, g2], graph_labels)

    See Also
    --------
    load_graphs
    """
    # if it is local file, do some sanity check
    if is_local_path(filename):
        if os.path.isdir(filename):
            raise DGLError("Filename {} is an existing directory.".format(filename))
        f_path = os.path.dirname(filename)
        if f_path and not os.path.exists(f_path):
            os.makedirs(f_path)

    g_sample = g_list[0] if isinstance(g_list, list) else g_list
    if type(g_sample) == DGLHeteroGraph:  # Doesn't support DGLHeteroGraph's derived class
        save_heterographs(filename, g_list, labels)
    else:
        raise DGLError(
            "Invalid argument g_list. Must be a DGLGraph or a list of DGLGraphs.")



def load_graphs(filename, idx_list=None):
    """Load graphs and optionally their labels from file saved by :func:`save_graphs`.

    Besides loading from local files, DGL supports loading the graphs directly
    from S3 (by providing a ``"s3://..."`` path) or from HDFS (by providing
    ``"hdfs://..."`` a path).

    Parameters
    ----------
    filename: str
        The file name to load graphs from.
    idx_list: list[int], optional
        The indices of the graphs to be loaded if the file contains multiple graphs.
        Default is loading all the graphs stored in the file.

    Returns
    --------
    graph_list: list[DGLGraph]
        The loaded graphs.
    labels: dict[str, Tensor]
        The graph labels stored in file. If no label is stored, the dictionary is empty.
        Regardless of whether the ``idx_list`` argument is given or not,
        the returned dictionary always contains the labels of all the graphs.

    Examples
    ----------
    Following the example in :func:`save_graphs`.

    >>> from dgl.data.utils import load_graphs
    >>> glist, label_dict = load_graphs("./data.bin") # glist will be [g1, g2]
    >>> glist, label_dict = load_graphs("./data.bin", [0]) # glist will be [g1]

    See Also
    --------
    save_graphs
    """
    # if it is local file, do some sanity check
    check_local_file_exists(filename)
    version = _CAPI_GetFileVersion(filename)
    if version == 1:
        dgl_warning(
            "You are loading a graph file saved by old version of dgl.  \
            Please consider saving it again with the current format.")
        return load_graph_v1(filename, idx_list)
    elif version == 2:
        return load_graph_v2(filename, idx_list)
    else:
        raise DGLError("Invalid DGL Version Number.")


def load_graph_v2(filename, idx_list=None):
    """Internal functions for loading DGLHeteroGraphs."""
    if idx_list is None:
        idx_list = []
    assert isinstance(idx_list, list)
    heterograph_list = _CAPI_LoadGraphFiles_V2(filename, idx_list)
    label_dict = load_labels_v2(filename)
    return [gdata.get_graph() for gdata in heterograph_list], label_dict


def load_graph_v1(filename, idx_list=None):
    """"Internal functions for loading DGLGraphs (V0)."""
    if idx_list is None:
        idx_list = []
    assert isinstance(idx_list, list)
    metadata = _CAPI_LoadGraphFiles_V1(filename, idx_list, False)
    label_dict = {}
    for k, v in metadata.labels.items():
        label_dict[k] = F.zerocopy_from_dgl_ndarray(v)

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
    # if it is local file, do some sanity check
    check_local_file_exists(filename)

    version = _CAPI_GetFileVersion(filename)
    if version == 1:
        return load_labels_v1(filename)
    elif version == 2:
        return load_labels_v2(filename)
    else:
        raise Exception("Invalid DGL Version Number")


def load_labels_v2(filename):
    """Internal functions for loading labels from V2 format"""
    label_dict = {}
    nd_dict = _CAPI_LoadLabels_V2(filename)
    for k, v in nd_dict.items():
        label_dict[k] = F.zerocopy_from_dgl_ndarray(v)
    return label_dict


def load_labels_v1(filename):
    """Internal functions for loading labels from V1 format"""
    metadata = _CAPI_LoadGraphFiles_V1(filename, [], True)
    label_dict = {}
    for k, v in metadata.labels.items():
        label_dict[k] = F.zerocopy_from_dgl_ndarray(v)
    return label_dict
