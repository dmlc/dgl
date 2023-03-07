"""Python interfaces to DGL farthest point sampler."""
import numpy as np

from .. import backend as F, ndarray as nd
from .._ffi.base import DGLError
from .._ffi.function import _init_api


def _farthest_point_sampler(
    data, batch_size, sample_points, dist, start_idx, result
):
    r"""Farthest Point Sampler

    Parameters
    ----------
    data : tensor
        A tensor of shape (N, d) where N is the number of points and d is the dimension.
    batch_size : int
        The number of batches in the ``data``. N should be divisible by batch_size.
    sample_points : int
        The number of points to sample in each batch.
    dist : tensor
        Pre-allocated tensor of shape (N, ) for to-sample distance.
    start_idx : tensor of int
        Pre-allocated tensor of shape (batch_size, ) for the starting sample in each batch.
    result : tensor of int
        Pre-allocated tensor of shape (sample_points * batch_size, ) for the sampled index.

    Returns
    -------
    No return value. The input variable ``result`` will be overwriten with sampled indices.

    """
    assert F.shape(data)[0] >= sample_points * batch_size
    assert F.shape(data)[0] % batch_size == 0

    _CAPI_FarthestPointSampler(
        F.zerocopy_to_dgl_ndarray(data),
        batch_size,
        sample_points,
        F.zerocopy_to_dgl_ndarray(dist),
        F.zerocopy_to_dgl_ndarray(start_idx),
        F.zerocopy_to_dgl_ndarray(result),
    )


def _neighbor_matching(
    graph_idx, num_nodes, edge_weights=None, relabel_idx=True
):
    """
    Description
    -----------
    The neighbor matching procedure of edge coarsening used in
    `Metis <http://cacs.usc.edu/education/cs653/Karypis-METIS-SIAMJSC98.pdf>`__
    and
    `Graclus <https://www.cs.utexas.edu/users/inderjit/public_papers/multilevel_pami.pdf>`__
    for homogeneous graph coarsening. This procedure keeps picking an unmarked
    vertex and matching it with one its unmarked neighbors (that maximizes its
    edge weight) until no match can be done.

    If no edge weight is given, this procedure will randomly pick neighbor for each
    vertex.

    The GPU implementation is based on `A GPU Algorithm for Greedy Graph Matching
    <http://www.staff.science.uu.nl/~bisse101/Articles/match12.pdf>`__

    NOTE: The input graph must be bi-directed (undirected) graph. Call :obj:`dgl.to_bidirected`
    if you are not sure your graph is bi-directed.

    Parameters
    ----------
    graph : HeteroGraphIndex
        The input homogeneous graph.
    num_nodes : int
        The number of nodes in this homogeneous graph.
    edge_weight : tensor, optional
        The edge weight tensor holding non-negative scalar weight for each edge.
        default: :obj:`None`
    relabel_idx : bool, optional
        If true, relabel resulting node labels to have consecutive node ids.
        default: :obj:`True`

    Returns
    -------
    a 1-D tensor
        A vector with each element that indicates the cluster ID of a vertex.
    """
    edge_weight_capi = nd.NULL["int64"]
    if edge_weights is not None:
        edge_weight_capi = F.zerocopy_to_dgl_ndarray(edge_weights)
    node_label = F.full_1d(
        num_nodes,
        -1,
        getattr(F, graph_idx.dtype),
        F.to_backend_ctx(graph_idx.ctx),
    )
    node_label_capi = F.zerocopy_to_dgl_ndarray_for_write(node_label)
    _CAPI_NeighborMatching(graph_idx, edge_weight_capi, node_label_capi)
    if F.reduce_sum(node_label < 0).item() != 0:
        raise DGLError("Find unmatched node")

    # reorder node id
    # TODO: actually we can add `return_inverse` option for `unique`
    #       function in backend for efficiency.
    if relabel_idx:
        node_label_np = F.zerocopy_to_numpy(node_label)
        _, node_label_np = np.unique(node_label_np, return_inverse=True)
        return F.tensor(node_label_np)
    else:
        return node_label


_init_api("dgl.geometry", __name__)
