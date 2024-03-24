"""Data utilities."""

from collections import namedtuple

import networkx as nx
import scipy as sp

from .. import backend as F
from ..base import DGLError
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
    """Function to convert a scipy matrix to a sparse adjacency matrix tuple.

    Note that the data array of the scipy matrix is discarded.

    Parameters
    ----------
    spmat : scipy.sparse.spmatrix
        SciPy sparse matrix.
    idtype : int32, int64, optional
        Integer ID type. Must be int32 or int64.

    Returns
    -------
    (str, tuple[Tensor])
        A tuple containing the format as well as the list of tensors representing
        the sparse matrix.
    """
    if spmat.format in ["csr", "csc"]:
        indptr = F.tensor(spmat.indptr, idtype)
        indices = F.tensor(spmat.indices, idtype)
        data = F.tensor([], idtype)
        return SparseAdjTuple(spmat.format, (indptr, indices, data))
    else:
        spmat = spmat.tocoo()
        row = F.tensor(spmat.row, idtype)
        col = F.tensor(spmat.col, idtype)
        return SparseAdjTuple("coo", (row, col))


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
    nx_graph = nx.convert_node_labels_to_integers(nx_graph, ordering="sorted")
    has_edge_id = edge_id_attr_name is not None

    if has_edge_id:
        num_edges = nx_graph.number_of_edges()
        src = [0] * num_edges
        dst = [0] * num_edges
        for u, v, attr in nx_graph.edges(data=True):
            eid = int(attr[edge_id_attr_name])
            if eid < 0 or eid >= nx_graph.number_of_edges():
                raise DGLError(
                    "Expect edge IDs to be a non-negative integer smaller than {:d}, "
                    "got {:d}".format(num_edges, eid)
                )
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


SparseAdjTuple = namedtuple("SparseAdjTuple", ["format", "arrays"])


def graphdata2tensors(
    data, idtype=None, bipartite=False, infer_node_count=True, **kwargs
):
    """Function to convert various types of data to edge tensors and infer
    the number of nodes.

    Parameters
    ----------
    data : graph data
        Various kinds of graph data.  Possible data types are:

        - ``(row, col)``
        - ``('coo', (row, col))``
        - ``('csr', (indptr, indices, edge_ids))``
        - ``('csc', (indptr, indices, edge_ids))``
        - SciPy sparse matrix
        - NetworkX graph
    idtype : int32, int64, optional
        Integer ID type. If None, try infer from the data and if fail use
        int64.
    bipartite : bool, optional
        Whether infer number of nodes of a bipartite graph --
        num_src and num_dst can be different.
    infer_node_count : bool, optional
        Whether infer number of nodes at all. If False, num_src and num_dst
        are returned as None.
    kwargs

        - edge_id_attr_name : The name (str) of the edge attribute that stores the edge
          IDs in the NetworkX graph.
        - top_map : The dictionary mapping the original IDs of the source nodes to the
          new ones.
        - bottom_map : The dictionary mapping the original IDs of the destination nodes
          to the new ones.

    Returns
    -------
    data : SparseAdjTuple
        A tuple with the sparse matrix format and the adjacency matrix tensors.
    num_src : int
        Number of source nodes.
    num_dst : int
        Number of destination nodes.
    """
    # Convert tuple to SparseAdjTuple
    if isinstance(data, tuple):
        if not isinstance(data[0], str):
            # (row, col) format, convert to ('coo', (row, col))
            data = ("coo", data)
        data = SparseAdjTuple(*data)

    if idtype is None and not (
        isinstance(data, SparseAdjTuple) and F.is_tensor(data.arrays[0])
    ):
        # preferred default idtype is int64
        # if data is tensor and idtype is None, infer the idtype from tensor
        idtype = F.int64
    checks.check_valid_idtype(idtype)

    if isinstance(data, SparseAdjTuple) and (
        not all(F.is_tensor(a) for a in data.arrays)
    ):
        # (Iterable, Iterable) type data, convert it to (Tensor, Tensor)
        if len(data.arrays[0]) == 0:
            # force idtype for empty list
            data = SparseAdjTuple(
                data.format, tuple(F.tensor(a, idtype) for a in data.arrays)
            )
        else:
            # convert the iterable to tensor and keep its native data type so we can check
            # its validity later
            data = SparseAdjTuple(
                data.format, tuple(F.tensor(a) for a in data.arrays)
            )

    num_src, num_dst = None, None
    if isinstance(data, SparseAdjTuple):
        if idtype is not None:
            data = SparseAdjTuple(
                data.format, tuple(F.astype(a, idtype) for a in data.arrays)
            )
        if infer_node_count:
            num_src, num_dst = infer_num_nodes(data, bipartite=bipartite)
    elif isinstance(data, list):
        src, dst = elist2tensor(data, idtype)
        data = SparseAdjTuple("coo", (src, dst))
        if infer_node_count:
            num_src, num_dst = infer_num_nodes(data, bipartite=bipartite)
    elif isinstance(data, sp.sparse.spmatrix):
        # We can get scipy matrix's number of rows and columns easily.
        if infer_node_count:
            num_src, num_dst = infer_num_nodes(data, bipartite=bipartite)
        data = scipy2tensor(data, idtype)
    elif isinstance(data, nx.Graph):
        # We can get networkx graph's number of sources and destinations easily.
        if infer_node_count:
            num_src, num_dst = infer_num_nodes(data, bipartite=bipartite)
        edge_id_attr_name = kwargs.get("edge_id_attr_name", None)
        if bipartite:
            top_map = kwargs.get("top_map")
            bottom_map = kwargs.get("bottom_map")
            src, dst = networkxbipartite2tensors(
                data,
                idtype,
                top_map=top_map,
                bottom_map=bottom_map,
                edge_id_attr_name=edge_id_attr_name,
            )
        else:
            src, dst = networkx2tensor(
                data, idtype, edge_id_attr_name=edge_id_attr_name
            )
        data = SparseAdjTuple("coo", (src, dst))
    else:
        raise DGLError("Unsupported graph data type:", type(data))

    return data, num_src, num_dst


def networkxbipartite2tensors(
    nx_graph, idtype, top_map, bottom_map, edge_id_attr_name=None
):
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
                raise DGLError(
                    "Expect the node {} to have attribute bipartite=0 "
                    "with edge {}".format(u, (u, v))
                )
            if v not in bottom_map:
                raise DGLError(
                    "Expect the node {} to have attribute bipartite=1 "
                    "with edge {}".format(v, (u, v))
                )
            eid = int(attr[edge_id_attr_name])
            if eid < 0 or eid >= nx_graph.number_of_edges():
                raise DGLError(
                    "Expect edge IDs to be a non-negative integer smaller than {:d}, "
                    "got {:d}".format(num_edges, eid)
                )
            src[eid] = top_map[u]
            dst[eid] = bottom_map[v]
    else:
        src = []
        dst = []
        for e in nx_graph.edges:
            u, v = e[0], e[1]
            if u not in top_map:
                raise DGLError(
                    "Expect the node {} to have attribute bipartite=0 "
                    "with edge {}".format(u, (u, v))
                )
            if v not in bottom_map:
                raise DGLError(
                    "Expect the node {} to have attribute bipartite=1 "
                    "with edge {}".format(v, (u, v))
                )
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

        * SparseTuple ``(sparse_fmt, arrays)`` where ``arrays`` can be either ``(src, dst)`` or
          ``(indptr, indices, data)``.
        * SciPy matrix.
        * NetworkX graph.
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
    if isinstance(data, tuple) and len(data) == 2:
        if not isinstance(data[0], str):
            raise TypeError(
                "Expected sparse format as a str, but got %s" % type(data[0])
            )

        if data[0] == "coo":
            # ('coo', (src, dst)) format
            u, v = data[1]
            nsrc = F.as_scalar(F.max(u, dim=0)) + 1 if len(u) > 0 else 0
            ndst = F.as_scalar(F.max(v, dim=0)) + 1 if len(v) > 0 else 0
        elif data[0] == "csr":
            # ('csr', (indptr, indices, eids)) format
            indptr, indices, _ = data[1]
            nsrc = F.shape(indptr)[0] - 1
            ndst = (
                F.as_scalar(F.max(indices, dim=0)) + 1
                if len(indices) > 0
                else 0
            )
        elif data[0] == "csc":
            # ('csc', (indptr, indices, eids)) format
            indptr, indices, _ = data[1]
            ndst = F.shape(indptr)[0] - 1
            nsrc = (
                F.as_scalar(F.max(indices, dim=0)) + 1
                if len(indices) > 0
                else 0
            )
        else:
            raise ValueError("unknown format %s" % data[0])
    elif isinstance(data, sp.sparse.spmatrix):
        nsrc, ndst = data.shape[0], data.shape[1]
    elif isinstance(data, nx.Graph):
        if data.number_of_nodes() == 0:
            nsrc = ndst = 0
        elif not bipartite:
            nsrc = ndst = data.number_of_nodes()
        else:
            nsrc = len(
                {n for n, d in data.nodes(data=True) if d["bipartite"] == 0}
            )
            ndst = data.number_of_nodes() - nsrc
    else:
        return None
    if not bipartite:
        nsrc = ndst = max(nsrc, ndst)
    return nsrc, ndst


def to_device(data, device):
    """Transfer the tensor or dictionary of tensors to the given device.

    Nothing will happen if the device of the original tensor is the same as target device.

    Parameters
    ----------
    data : Tensor or dict[str, Tensor]
        The data.
    device : device
        The target device.

    Returns
    -------
    Tensor or dict[str, Tensor]
        The output data.
    """
    if isinstance(data, dict):
        return {k: F.copy_to(v, device) for k, v in data.items()}
    else:
        return F.copy_to(data, device)
