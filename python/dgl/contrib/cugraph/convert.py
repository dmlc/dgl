import dgl
from .utils.cugraph_conversion_utils import (
    add_nodes_from_dgl_heteroGraph,
    add_edges_from_dgl_heteroGraph,
)
from dgl.contrib.cugraph.cugraph_storage import CuGraphStorage


def cugraph_storage_from_heterograph(
    g: dgl.DGLHeteroGraph, single_gpu=True
) -> CuGraphStorage:
    """
    Convert DGL Graph to CuGraphStorage graph
    """
    num_nodes_dict = {ntype: g.num_nodes(ntype) for ntype in g.ntypes}
    gs = dgl.contrib.cugraph.CuGraphStorage(
        single_gpu=single_gpu, num_nodes_dict=num_nodes_dict, idtype=g.idtype
    )
    add_nodes_from_dgl_heteroGraph(gs, g)
    add_edges_from_dgl_heteroGraph(gs, g)
    return gs
