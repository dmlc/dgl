"""Factory routines from/to DGLGraphs and DGLHeteroGraphs."""
from .heterograph import DGLBaseBipartite

def bipartite_from_edge_list(src_type, dst_type, edge_type, u, v,
                             num_src=None, num_dst=None):
    """Create a bipartite graph component of a heterogeneous graph with a
    list of edges.

    Parameters
    ----------
    src_type : str
        The source type name.
    dst_type : str
        The destination type name.
    edge_type : str
        The edge type name.
    u, v : list[int]
        List of source and destination node IDs.
    num_src : int, optional
        The number of nodes of source type.

        By default, the value is the maximum of the source node IDs in the
        edge list plus 1.
    num_dst : int, optional
        The number of nodes of destination type.

        By default, the value is the maximum of the destination node IDs in
        the edge list plus 1.
    """
    num_src = num_src or (max(u) + 1)
    num_dst = num_dst or (max(v) + 1)
    return DGLBaseBipartite.from_coo(
        src_type, dst_type, edge_type, num_src, num_dst, u, v)

def bipartite_from_scipy(src_type, dst_type, edge_type, spmat, with_edge_id=False):
    """Create a bipartite graph component of a heterogeneous graph with a
    scipy sparse matrix.

    Parameters
    ----------
    src_type : str
        The source type name.
    dst_type : str
        The destination type name.
    edge_type : str
        The edge typ3e name.
    spmat : scipy sparse matrix
        The bipartite graph matrix whose rows represent sources and columns
        represent destinations.
    with_edge_id : bool
        If True, the entries in the sparse matrix are treated as edge IDs.
        Otherwise, the entries are ignored and edges will be added in
        (source, destination) order.
    """
    spmat = spmat.tocsr()
    num_src, num_dst = spmat.shape
    indptr = spmat.indptr
    indices = spmat.indices
    data = spmat.data if with_edge_id else list(range(len(indices)))
    return DGLBaseBipartite.from_csr(
        src_type, dst_type, edge_type, num_src, num_dst, indptr, indices, data)
