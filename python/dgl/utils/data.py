"""Data utilities."""

import scipy as sp
import networkx as nx

from ..base import DGLError
from .. import backend as F
from . import checks

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

def networkx2tensor(nx_graph, idtype, edge_id_attr_name=None):
    """Function to convert a networkx graph to edge tensors.

    Parameters
    ----------
    nx_graph : nx.Graph
        NetworkX graph.
    idtype : int32, int64, optional
        Integer ID type. Must be int32 or int64.
    edge_id_attr_name : str, optional
        Key name for edge ids in the NetworkX graph. If not found, we
        will consider the graph not to have pre-specified edge ids. (Default: None)

    Returns
    -------
    (Tensor, Tensor)
        Edge tensors.
    """
    if not nx_graph.is_directed():
        nx_graph = nx_graph.to_directed()

    # Relabel nodes using consecutive integers
    nx_graph = nx.convert_node_labels_to_integers(nx_graph, ordering='sorted')
    has_edge_id = edge_id_attr_name is not None

    if has_edge_id:
        num_edges = nx_graph.number_of_edges()
        src = [0] * num_edges
        dst = [0] * num_edges
        for u, v, attr in nx_graph.edges(data=True):
            eid = int(attr[edge_id_attr_name])
            if eid < 0 or eid >= nx_graph.number_of_edges():
                raise DGLError('Expect edge IDs to be a non-negative integer smaller than {:d}, '
                               'got {:d}'.format(num_edges, eid))
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

def graphdata2tensors(data, idtype=None, bipartite=False, **kwargs):
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
    kwargs

        - edge_id_attr_name : The name (str) of the edge attribute that stores the edge
          IDs in the NetworkX graph.
        - top_map : The dictionary mapping the original IDs of the source nodes to the
          new ones.
        - bottom_map : The dictionary mapping the original IDs of the destination nodes
          to the new ones.

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
    checks.check_valid_idtype(idtype)

    if isinstance(data, tuple) and (not F.is_tensor(data[0]) or not F.is_tensor(data[1])):
        # (Iterable, Iterable) type data, convert it to (Tensor, Tensor)
        if len(data[0]) == 0:
            # force idtype for empty list
            data = F.tensor(data[0], idtype), F.tensor(data[1], idtype)
        else:
            # convert the iterable to tensor and keep its native data type so we can check
            # its validity later
            data = F.tensor(data[0]), F.tensor(data[1])

    if isinstance(data, tuple):
        # (Tensor, Tensor) type data
        src, dst = data
        # sanity checks
        # TODO(minjie): move these checks to C for faster graph construction.
        if F.dtype(src) != F.dtype(dst):
            raise DGLError('Expect the source and destination node IDs to have the same type,'
                           ' but got {} and {}.'.format(F.dtype(src), F.dtype(dst)))
        if F.context(src) != F.context(dst):
            raise DGLError('Expect the source and destination node IDs to be on the same device,'
                           ' but got {} and {}.'.format(F.context(src), F.context(dst)))
        if F.dtype(src) not in (F.int32, F.int64):
            raise DGLError('Expect the source ID tensor to have data type int32 or int64,'
                           ' but got {}.'.format(F.dtype(src)))
        if F.dtype(dst) not in (F.int32, F.int64):
            raise DGLError('Expect the destination ID tensor to have data type int32 or int64,'
                           ' but got {}.'.format(F.dtype(dst)))
        if idtype is not None:
            src, dst = F.astype(src, idtype), F.astype(dst, idtype)
    elif isinstance(data, list):
        src, dst = elist2tensor(data, idtype)
    elif isinstance(data, sp.sparse.spmatrix):
        src, dst = scipy2tensor(data, idtype)
    elif isinstance(data, nx.Graph):
        edge_id_attr_name = kwargs.get('edge_id_attr_name', None)
        if bipartite:
            top_map = kwargs.get('top_map')
            bottom_map = kwargs.get('bottom_map')
            src, dst = networkxbipartite2tensors(
                data, idtype, top_map=top_map,
                bottom_map=bottom_map, edge_id_attr_name=edge_id_attr_name)
        else:
            src, dst = networkx2tensor(
                data, idtype, edge_id_attr_name=edge_id_attr_name)
    else:
        raise DGLError('Unsupported graph data type:', type(data))

    if len(src) != len(dst):
        raise DGLError('Expect the source and destination ID tensors to have the same length,'
                       ' but got {} and {}.'.format(len(src), len(dst)))
    if len(src) > 0 and (F.as_scalar(F.min(src, 0)) < 0 or F.as_scalar(F.min(dst, 0)) < 0):
        raise DGLError('All IDs must be non-negative integers.')

    # infer number of nodes
    infer_from_raw = infer_num_nodes(data, bipartite=bipartite)
    if infer_from_raw is None:
        num_src, num_dst = infer_num_nodes((src, dst), bipartite=bipartite)
    else:
        num_src, num_dst = infer_from_raw
    return src, dst, num_src, num_dst

def networkxbipartite2tensors(nx_graph, idtype, top_map, bottom_map, edge_id_attr_name=None):
    """Function to convert a networkx bipartite to edge tensors.

    Parameters
    ----------
    nx_graph : nx.Graph
        NetworkX graph. It must follow the bipartite graph convention of networkx.
        Each node has an attribute ``bipartite`` with values 0 and 1 indicating
        which set it belongs to.
    top_map : dict
        The dictionary mapping the original node labels to the node IDs for the source type.
    bottom_map : dict
        The dictionary mapping the original node labels to the node IDs for the destination type.
    idtype : int32, int64, optional
        Integer ID type. Must be int32 or int64.
    edge_id_attr_name : str, optional
        Key name for edge ids in the NetworkX graph. If not found, we
        will consider the graph not to have pre-specified edge ids. (Default: None)

    Returns
    -------
    (Tensor, Tensor)
        Edge tensors.
    """
    has_edge_id = edge_id_attr_name is not None

    if has_edge_id:
        num_edges = nx_graph.number_of_edges()
        src = [0] * num_edges
        dst = [0] * num_edges
        for u, v, attr in nx_graph.edges(data=True):
            if u not in top_map:
                raise DGLError('Expect the node {} to have attribute bipartite=0 '
                               'with edge {}'.format(u, (u, v)))
            if v not in bottom_map:
                raise DGLError('Expect the node {} to have attribute bipartite=1 '
                               'with edge {}'.format(v, (u, v)))
            eid = int(attr[edge_id_attr_name])
            if eid < 0 or eid >= nx_graph.number_of_edges():
                raise DGLError('Expect edge IDs to be a non-negative integer smaller than {:d}, '
                               'got {:d}'.format(num_edges, eid))
            src[eid] = top_map[u]
            dst[eid] = bottom_map[v]
    else:
        src = []
        dst = []
        for e in nx_graph.edges:
            u, v = e[0], e[1]
            if u not in top_map:
                raise DGLError('Expect the node {} to have attribute bipartite=0 '
                               'with edge {}'.format(u, (u, v)))
            if v not in bottom_map:
                raise DGLError('Expect the node {} to have attribute bipartite=1 '
                               'with edge {}'.format(v, (u, v)))
            src.append(top_map[u])
            dst.append(bottom_map[v])
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
