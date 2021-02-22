"""Python interfaces to DGL farthest point sampler."""
import numpy as np
from .._ffi.function import _init_api
from .. import backend as F
from .. import ndarray as nd

__all__ = ["edge_coarsening"]


def farthest_point_sampler(data, batch_size, sample_points, dist, start_idx, result):
    """Farthest Point Sampler

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

    _CAPI_FarthestPointSampler(F.zerocopy_to_dgl_ndarray(data),
                               batch_size, sample_points,
                               F.zerocopy_to_dgl_ndarray(dist),
                               F.zerocopy_to_dgl_ndarray(start_idx),
                               F.zerocopy_to_dgl_ndarray(result))


def edge_coarsening(graph, edge_weights=None, relabel_idx=True):
    """
    Description
    -----------
    The edge coarsening procedure used in 
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
    graph : DGLGraph
        The input homogeneous graph.
    edge_weight : torch.Tensor, optional
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
    assert graph.is_homogeneous, \
    "The graph used in graph node matching must be homogeneous"
    
    # currently only int64 are supported
    if graph.idtype == F.int32:
        graph = graph.long()

    num_nodes = graph.num_nodes()
    src, dst = graph.edges()

    # remove self-loops
    mask = src != dst
    src, dst = src[mask], dst[mask]
    if edge_weights is not None:
        edge_weights = edge_weights[mask]

    # randomly shuffle if no edge weight is given
    # thus we can randomly pick neighbors for each node
    if edge_weights is None:
        perm = F.rand_shuffle(F.arange(0, src.size(0), F.dtype(src), F.context(src)))
        src, dst = src[perm], dst[perm]

    # convert to CSR
    perm = F.argsort(src, 0, False)
    src, dst = src[perm], dst[perm]
    deg = F.zeros(num_nodes, F.dtype(src), F.context(src))
    F.index_add_inplace(deg, src, F.full_1d(F.shape(src)[0], 1, F.dtype(src), F.context(src)))
    indptr = F.zeros(num_nodes + 1, F.dtype(src), F.context(src))
    indptr[1:] = F.cumsum(deg, dim=0)

    node_label = F.full_1d(num_nodes, -1, F.dtype(src), F.context(src))

    _CAPI_EdgeCoarsening(F.zerocopy_to_dgl_ndarray(indptr),
                         F.zerocopy_to_dgl_ndarray(dst),
                         nd.NULL["int64"] if edge_weights is None else F.zerocopy_to_dgl_ndarray(edge_weights),
                         F.zerocopy_to_dgl_ndarray_for_write(node_label))
    assert F.reduce_sum(node_label < 0).item() == 0, "Find unmatched node"

    # reorder node id
    if relabel_idx:
        node_label_np = F.zerocopy_to_numpy(node_label)
        _, node_label_np = np.unique(node_label_np, return_inverse=True)
        return F.tensor(node_label_np)
    else:
        return node_label


_init_api('dgl.geometry', __name__)
