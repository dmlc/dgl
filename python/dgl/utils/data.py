"""Data utilities."""

import scipy as sp
import networkx as nx

from ..base import DGLError
from .. import backend as F

def elist2tensor(elist, idtype):
    """Function to convert an edge list to edge tensors.

    Parameters
    ----------
    elist : iterable of int pairs
        List of (src, dst) node ID pairs.
    idtype : int32, int64, optional
        Integer ID type. Must be int32 or int64.

    Returns
    -------
    (Tensor, Tensor)
        Edge tensors.
    """
    if len(elist) == 0:
        u, v = [], []
    else:
        u, v = zip(*elist)
        u = list(u)
        v = list(v)
    return F.tensor(u, idtype), F.tensor(v, idtype)

def scipy2tensor(spmat, idtype):
    """Function to convert a scipy matrix to edge tensors.

    Parameters
    ----------
    spmat : scipy.sparse.spmatrix
        SciPy sparse matrix.
    idtype : int32, int64, optional
        Integer ID type. Must be int32 or int64.

    Returns
    -------
    (Tensor, Tensor)
        Edge tensors.
    """
    spmat = spmat.tocoo()
    row = F.tensor(spmat.row, idtype)
    col = F.tensor(spmat.col, idtype)
    return row, col

def networkx2tensor(nx_graph, idtype, edge_id_attr_name='id'):
    """Function to convert a networkx graph to edge tensors.

    Parameters
    ----------
    nx_graph : nx.Graph
        NetworkX graph.
    idtype : int32, int64, optional
        Integer ID type. Must be int32 or int64.
    edge_id_attr_name : str, optional
        Key name for edge ids in the NetworkX graph. If not found, we
        will consider the graph not to have pre-specified edge ids. (Default: 'id')

    Returns
    -------
    (Tensor, Tensor)
        Edge tensors.
    """
    if not nx_graph.is_directed():
        nx_graph = nx_graph.to_directed()

    # Relabel nodes using consecutive integers
    nx_graph = nx.convert_node_labels_to_integers(nx_graph, ordering='sorted')

    # nx_graph.edges(data=True) returns src, dst, attr_dict
    if nx_graph.number_of_edges() > 0:
        has_edge_id = edge_id_attr_name in next(iter(nx_graph.edges(data=True)))[-1]
    else:
        has_edge_id = False

    if has_edge_id:
        num_edges = nx_graph.number_of_edges()
        src = [0] * num_edges
        dst = [0] * num_edges
        for u, v, attr in nx_graph.edges(data=True):
            eid = attr[edge_id_attr_name]
            src[eid] = u
            dst[eid] = v
    else:
        src = []
        dst = []
        for e in nx_graph.edges:
            src.append(e[0])
            dst.append(e[1])
    src = F.tensor(src, idtype)
    dst = F.tensor(dst, idtype)
    return src, dst

def graphdata2tensors(data, idtype=None, bipartite=False):
    """Function to convert various types of data to edge tensors and infer
    the number of nodes.

    Parameters
    ----------
    data : graph data
        Various kinds of graph data.
    idtype : int32, int64, optional
        Integer ID type. If None, try infer from the data and if fail use
        int64.
    bipartite : bool, optional
        Whether infer number of nodes of a bipartite graph --
        num_src and num_dst can be different.

    Returns
    -------
    src : Tensor
        Src nodes.
    dst : Tensor
        Dst nodes.
    num_src : int
        Number of source nodes
    num_dst : int
        Number of destination nodes.
    """
    if idtype is None and not (isinstance(data, tuple) and F.is_tensor(data[0])):
        # preferred default idtype is int64
        # if data is tensor and idtype is None, infer the idtype from tensor
        idtype = F.int64
    if isinstance(data, tuple):
        src, dst = F.tensor(data[0], idtype), F.tensor(data[1], idtype)
    elif isinstance(data, list):
        src, dst = elist2tensor(data, idtype)
    elif isinstance(data, sp.sparse.spmatrix):
        src, dst = scipy2tensor(data, idtype)
    elif isinstance(data, nx.Graph):
        if bipartite:
            src, dst = networkxbipartite2tensors(data, idtype)
        else:
            src, dst = networkx2tensor(data, idtype)
    else:
        raise DGLError('Unsupported graph data type:', type(data))
    infer_from_raw = infer_num_nodes(data, bipartite=bipartite)
    if infer_from_raw is None:
        num_src, num_dst = infer_num_nodes((src, dst), bipartite=bipartite)
    else:
        num_src, num_dst = infer_from_raw
    return src, dst, num_src, num_dst

def networkxbipartite2tensors(nx_graph, idtype, edge_id_attr_name='id'):
    """Function to convert a networkx bipartite to edge tensors.

    Parameters
    ----------
    nx_graph : nx.Graph
        NetworkX graph. It must follow the bipartite graph convention of networkx.
        Each node has an attribute ``bipartite`` with values 0 and 1 indicating
        which set it belongs to. Only edges from node set 0 to node set 1 are
        added to the returned graph.
    idtype : int32, int64, optional
        Integer ID type. Must be int32 or int64.
    edge_id_attr_name : str, optional
        Key name for edge ids in the NetworkX graph. If not found, we
        will consider the graph not to have pre-specified edge ids. (Default: 'id')

    Returns
    -------
    (Tensor, Tensor)
        Edge tensors.
    """
    if not nx_graph.is_directed():
        nx_graph = nx_graph.to_directed()

    top_nodes = {n for n, d in nx_graph.nodes(data=True) if d['bipartite'] == 0}
    bottom_nodes = set(nx_graph) - top_nodes
    top_nodes = sorted(top_nodes)
    bottom_nodes = sorted(bottom_nodes)
    top_map = {n : i for i, n in enumerate(top_nodes)}
    bottom_map = {n : i for i, n in enumerate(bottom_nodes)}

    if nx_graph.number_of_edges() > 0:
        has_edge_id = edge_id_attr_name in next(iter(nx_graph.edges(data=True)))[-1]
    else:
        has_edge_id = False

    if has_edge_id:
        num_edges = nx_graph.number_of_edges()
        src = [0] * num_edges
        dst = [0] * num_edges
        for u, v, attr in nx_graph.edges(data=True):
            eid = attr[edge_id_attr_name]
            src[eid] = top_map[u]
            dst[eid] = bottom_map[v]
    else:
        src = []
        dst = []
        for e in nx_graph.edges:
            if e[0] in top_map:
                src.append(top_map[e[0]])
                dst.append(bottom_map[e[1]])
    src = F.tensor(src, dtype=idtype)
    dst = F.tensor(dst, dtype=idtype)
    return src, dst

def infer_num_nodes(data, bipartite=False):
    """Function for inferring the number of nodes.

    Parameters
    ----------
    data : graph data
        Supported types are:
        * Tensor pair (u, v)
        * SciPy matrix
        * NetworkX graph
    bipartite : bool, optional
        Whether infer number of nodes of a bipartite graph --
        num_src and num_dst can be different.

    Returns
    -------
    num_src : int
        Number of source nodes.
    num_dst : int
        Number of destination nodes.

    or

    None
        If the inference failed.
    """
    if isinstance(data, tuple) and len(data) == 2 and F.is_tensor(data[0]):
        u, v = data
        nsrc = F.as_scalar(F.max(u, dim=0)) + 1 if len(u) > 0 else 0
        ndst = F.as_scalar(F.max(v, dim=0)) + 1 if len(v) > 0 else 0
    elif isinstance(data, sp.sparse.spmatrix):
        nsrc, ndst = data.shape[0], data.shape[1]
    elif isinstance(data, nx.Graph):
        if data.number_of_nodes() == 0:
            nsrc = ndst = 0
        elif not bipartite:
            nsrc = ndst = data.number_of_nodes()
        else:
            nsrc = len({n for n, d in data.nodes(data=True) if d['bipartite'] == 0})
            ndst = data.number_of_nodes() - nsrc
    else:
        return None
    if not bipartite:
        nsrc = ndst = max(nsrc, ndst)
    return nsrc, ndst
