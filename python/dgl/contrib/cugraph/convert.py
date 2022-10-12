import dgl
from .utils.cugraph_conversion_utils import (
    add_nodes_from_dgl_heteroGraph,
    add_edges_from_dgl_heteroGraph,
)
from dgl.contrib.cugraph.cugraph_storage import CuGraphStorage


def cugraph_storage_from_heterograph(g: dgl.DGLHeteroGraph, single_gpu=True) -> CuGraphStorage:
    """
    Convert DGL Graph to CuGraphStorage graph
    """
    gs = dgl.contrib.cugraph.CuGraphStorage(single_gpu=single_gpu, idtype=g.idtype)
    num_nodes_dict = {ntype: g.num_nodes(ntype) for ntype in g.ntypes}
    add_nodes_from_dgl_heteroGraph(gs, g, num_nodes_dict)
    add_edges_from_dgl_heteroGraph(gs, g, num_nodes_dict)
    return gs
