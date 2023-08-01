##
#   Copyright 2019-2021 Contributors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
"""Functional interface for transform"""
# pylint: disable= too-many-lines

import copy
from collections.abc import Iterable, Mapping

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
from packaging.version import Version

try:
    import torch as th
except ImportError:
    pass

from .. import (
    backend as F,
    batch,
    convert,
    function,
    ndarray as nd,
    subgraph,
    utils,
)
from .._ffi.function import _init_api
from ..base import dgl_warning, DGLError, EID, NID
from ..frame import Frame
from ..heterograph import DGLGraph
from ..heterograph_index import (
    create_heterograph_from_relations,
    create_metagraph_index,
)
from ..partition import (
    metis_partition,
    metis_partition_assignment,
    partition_graph_with_halo,
)
from ..sampling.neighbor import sample_neighbors

__all__ = [
    "line_graph",
    "khop_adj",
    "khop_graph",
    "reverse",
    "to_bidirected",
    "add_reverse_edges",
    "laplacian_lambda_max",
    "knn_graph",
    "segmented_knn_graph",
    "add_edges",
    "add_nodes",
    "remove_edges",
    "remove_nodes",
    "add_self_loop",
    "remove_self_loop",
    "metapath_reachable_graph",
    "compact_graphs",
    "to_simple",
    "to_simple_graph",
    "sort_csr_by_tag",
    "sort_csc_by_tag",
    "metis_partition_assignment",
    "partition_graph_with_halo",
    "metis_partition",
    "adj_product_graph",
    "adj_sum_graph",
    "reorder_graph",
    "norm_by_dst",
    "radius_graph",
    "random_walk_pe",
    "laplacian_pe",
    "lap_pe",
    "to_bfloat16",
    "to_half",
    "to_float",
    "to_double",
    "double_radius_node_labeling",
    "shortest_dist",
    "svd_pe",
]


def pairwise_squared_distance(x):
    """
    x : (n_samples, n_points, dims)
    return : (n_samples, n_points, n_points)
    """
    x2s = F.sum(x * x, -1, True)
    # assuming that __matmul__ is always implemented (true for PyTorch, MXNet and Chainer)
    return x2s + F.swapaxes(x2s, -1, -2) - 2 * x @ F.swapaxes(x, -1, -2)


# pylint: disable=invalid-name
def knn_graph(
    x, k, algorithm="bruteforce-blas", dist="euclidean", exclude_self=False
):
    r"""Construct a graph from a set of points according to k-nearest-neighbor (KNN)
    and return.

    The function transforms the coordinates/features of a point set
    into a directed homogeneous graph. The coordinates of the point
    set is specified as a matrix whose rows correspond to points and
    columns correspond to coordinate/feature dimensions.

    The nodes of the returned graph correspond to the points, where the predecessors
    of each point are its k-nearest neighbors measured by the chosen distance.

    If :attr:`x` is a 3D tensor, then each submatrix will be transformed
    into a separate graph. DGL then composes the graphs into a large batched
    graph of multiple (:math:`shape(x)[0]`) connected components.

    See :doc:`the benchmark <../api/python/knn_benchmark>` for a complete benchmark result.

    Parameters
    ----------
    x : Tensor
        The point coordinates. It can be either on CPU or GPU.

        * If is 2D, ``x[i]`` corresponds to the i-th node in the KNN graph.

        * If is 3D, ``x[i]`` corresponds to the i-th KNN graph and
          ``x[i][j]`` corresponds to the j-th node in the i-th KNN graph.
    k : int
        The number of nearest neighbors per node.
    algorithm : str, optional
        Algorithm used to compute the k-nearest neighbors.

        * 'bruteforce-blas' will first compute the distance matrix
          using BLAS matrix multiplication operation provided by
          backend frameworks. Then use topk algorithm to get
          k-nearest neighbors. This method is fast when the point
          set is small but has :math:`O(N^2)` memory complexity where
          :math:`N` is the number of points.

        * 'bruteforce' will compute distances pair by pair and
          directly select the k-nearest neighbors during distance
          computation. This method is slower than 'bruteforce-blas'
          but has less memory overhead (i.e., :math:`O(Nk)` where :math:`N`
          is the number of points, :math:`k` is the number of nearest
          neighbors per node) since we do not need to store all distances.

        * 'bruteforce-sharemem' (CUDA only) is similar to 'bruteforce'
          but use shared memory in CUDA devices for buffer. This method is
          faster than 'bruteforce' when the dimension of input points
          is not large. This method is only available on CUDA device.

        * 'kd-tree' will use the kd-tree algorithm (CPU only).
          This method is suitable for low-dimensional data (e.g. 3D
          point clouds)

        * 'nn-descent' is an approximate approach from paper
          `Efficient k-nearest neighbor graph construction for generic similarity
          measures <https://www.cs.princeton.edu/cass/papers/www11.pdf>`_. This method
          will search for nearest neighbor candidates in "neighbors' neighbors".

        (default: 'bruteforce-blas')
    dist : str, optional
        The distance metric used to compute distance between points. It can be the following
        metrics:
        * 'euclidean': Use Euclidean distance (L2 norm) :math:`\sqrt{\sum_{i} (x_{i} - y_{i})^{2}}`.
        * 'cosine': Use cosine distance.
        (default: 'euclidean')
    exclude_self : bool, optional
        If True, the output graph will not contain self loop edges, and each node will not
        be counted as one of its own k neighbors.  If False, the output graph will contain
        self loop edges, and a node will be counted as one of its own k neighbors.

    Returns
    -------
    DGLGraph
        The constructed graph. The node IDs are in the same order as :attr:`x`.

    Examples
    --------

    The following examples use PyTorch backend.

    >>> import dgl
    >>> import torch

    When :attr:`x` is a 2D tensor, a single KNN graph is constructed.

    >>> x = torch.tensor([[0.0, 0.0, 1.0],
    ...                   [1.0, 0.5, 0.5],
    ...                   [0.5, 0.2, 0.2],
    ...                   [0.3, 0.2, 0.4]])
    >>> knn_g = dgl.knn_graph(x, 2)  # Each node has two predecessors
    >>> knn_g.edges()
    (tensor([0, 1, 2, 2, 2, 3, 3, 3]), tensor([0, 1, 1, 2, 3, 0, 2, 3]))

    When :attr:`x` is a 3D tensor, DGL constructs multiple KNN graphs and
    and then composes them into a graph of multiple connected components.

    >>> x1 = torch.tensor([[0.0, 0.0, 1.0],
    ...                    [1.0, 0.5, 0.5],
    ...                    [0.5, 0.2, 0.2],
    ...                    [0.3, 0.2, 0.4]])
    >>> x2 = torch.tensor([[0.0, 1.0, 1.0],
    ...                    [0.3, 0.3, 0.3],
    ...                    [0.4, 0.4, 1.0],
    ...                    [0.3, 0.8, 0.2]])
    >>> x = torch.stack([x1, x2], dim=0)
    >>> knn_g = dgl.knn_graph(x, 2)  # Each node has two predecessors
    >>> knn_g.edges()
    (tensor([0, 1, 2, 2, 2, 3, 3, 3, 4, 5, 5, 5, 6, 6, 7, 7]),
     tensor([0, 1, 1, 2, 3, 0, 2, 3, 4, 5, 6, 7, 4, 6, 5, 7]))
    """
    if exclude_self:
        # add 1 to k, for the self edge, since it will be removed
        k = k + 1

    # check invalid k
    if k <= 0:
        raise DGLError("Invalid k value. expect k > 0, got k = {}".format(k))

    # check empty point set
    x_size = tuple(F.shape(x))
    if x_size[0] == 0:
        raise DGLError("Find empty point set")

    d = F.ndim(x)
    x_seg = x_size[0] * [x_size[1]] if d == 3 else [x_size[0]]
    if algorithm == "bruteforce-blas":
        result = _knn_graph_blas(x, k, dist=dist)
    else:
        if d == 3:
            x = F.reshape(x, (x_size[0] * x_size[1], x_size[2]))
        out = knn(k, x, x_seg, algorithm=algorithm, dist=dist)
        row, col = out[1], out[0]
        result = convert.graph((row, col))

    if d == 3:
        # set batch information if x is 3D
        num_nodes = F.tensor(x_seg, dtype=F.int64).to(F.context(x))
        result.set_batch_num_nodes(num_nodes)
        # if any segment is too small for k, all algorithms reduce k for all segments
        clamped_k = min(k, np.min(x_seg))
        result.set_batch_num_edges(clamped_k * num_nodes)

    if exclude_self:
        # remove_self_loop will update batch_num_edges as needed
        result = remove_self_loop(result)

        # If there were more than k(+1) coincident points, there may not have been self loops on
        # all nodes, in which case there would still be one too many out edges on some nodes.
        # However, if every node had a self edge, the common case, every node would still have the
        # same degree as each other, so we can check that condition easily.
        # The -1 is for the self edge removal.
        clamped_k = min(k, np.min(x_seg)) - 1
        if result.num_edges() != clamped_k * result.num_nodes():
            # edges on any nodes with too high degree should all be length zero,
            # so pick an arbitrary one to remove from each such node
            degrees = result.in_degrees()
            node_indices = F.nonzero_1d(degrees > clamped_k)
            edges_to_remove_graph = sample_neighbors(
                result, node_indices, 1, edge_dir="in"
            )
            edge_ids = edges_to_remove_graph.edata[EID]
            result = remove_edges(result, edge_ids)

    return result


def _knn_graph_blas(x, k, dist="euclidean"):
    r"""Construct a graph from a set of points according to k-nearest-neighbor (KNN).

    This function first compute the distance matrix using BLAS matrix multiplication
    operation provided by backend frameworks. Then use topk algorithm to get
    k-nearest neighbors.

    Parameters
    ----------
    x : Tensor
        The point coordinates. It can be either on CPU or GPU.

        * If is 2D, ``x[i]`` corresponds to the i-th node in the KNN graph.

        * If is 3D, ``x[i]`` corresponds to the i-th KNN graph and
          ``x[i][j]`` corresponds to the j-th node in the i-th KNN graph.
    k : int
        The number of nearest neighbors per node.
    dist : str, optional
        The distance metric used to compute distance between points. It can be the following
        metrics:
        * 'euclidean': Use Euclidean distance (L2 norm) :math:`\sqrt{\sum_{i} (x_{i} - y_{i})^{2}}`.
        * 'cosine': Use cosine distance.
        (default: 'euclidean')
    """
    if F.ndim(x) == 2:
        x = F.unsqueeze(x, 0)
    n_samples, n_points, _ = F.shape(x)

    if k > n_points:
        dgl_warning(
            "'k' should be less than or equal to the number of points in 'x'"
            "expect k <= {0}, got k = {1}, use k = {0}".format(n_points, k)
        )
        k = n_points

    # if use cosine distance, normalize input points first
    # thus we can use euclidean distance to find knn equivalently.
    if dist == "cosine":
        l2_norm = lambda v: F.sqrt(F.sum(v * v, dim=2, keepdims=True))
        x = x / (l2_norm(x) + 1e-5)

    ctx = F.context(x)
    dist = pairwise_squared_distance(x)
    k_indices = F.astype(F.argtopk(dist, k, 2, descending=False), F.int64)
    # index offset for each sample
    offset = F.arange(0, n_samples, ctx=ctx) * n_points
    offset = F.unsqueeze(offset, 1)
    src = F.reshape(k_indices, (n_samples, n_points * k))
    src = F.unsqueeze(src, 0) + offset
    dst = F.repeat(F.arange(0, n_points, ctx=ctx), k, dim=0)
    dst = F.unsqueeze(dst, 0) + offset
    return convert.graph((F.reshape(src, (-1,)), F.reshape(dst, (-1,))))


# pylint: disable=invalid-name
def segmented_knn_graph(
    x,
    k,
    segs,
    algorithm="bruteforce-blas",
    dist="euclidean",
    exclude_self=False,
):
    r"""Construct multiple graphs from multiple sets of points according to
    k-nearest-neighbor (KNN) and return.

    Compared with :func:`dgl.knn_graph`, this allows multiple point sets with
    different capacity. The points from different sets are stored contiguously
    in the :attr:`x` tensor.
    :attr:`segs` specifies the number of points in each point set. The
    function constructs a KNN graph for each point set, where the predecessors
    of each point are its k-nearest neighbors measured by the Euclidean distance.
    DGL then composes all KNN graphs
    into a batched graph with multiple (:math:`len(segs)`) connected components.

    Parameters
    ----------
    x : Tensor
        Coordinates/features of points. Must be 2D. It can be either on CPU or GPU.
    k : int
        The number of nearest neighbors per node.
    segs : list[int]
        Number of points in each point set. The numbers in :attr:`segs`
        must sum up to the number of rows in :attr:`x`.
    algorithm : str, optional
        Algorithm used to compute the k-nearest neighbors.

        * 'bruteforce-blas' will first compute the distance matrix
          using BLAS matrix multiplication operation provided by
          backend frameworks. Then use topk algorithm to get
          k-nearest neighbors. This method is fast when the point
          set is small but has :math:`O(N^2)` memory complexity where
          :math:`N` is the number of points.

        * 'bruteforce' will compute distances pair by pair and
          directly select the k-nearest neighbors during distance
          computation. This method is slower than 'bruteforce-blas'
          but has less memory overhead (i.e., :math:`O(Nk)` where :math:`N`
          is the number of points, :math:`k` is the number of nearest
          neighbors per node) since we do not need to store all distances.

        * 'bruteforce-sharemem' (CUDA only) is similar to 'bruteforce'
          but use shared memory in CUDA devices for buffer. This method is
          faster than 'bruteforce' when the dimension of input points
          is not large. This method is only available on CUDA device.

        * 'kd-tree' will use the kd-tree algorithm (CPU only).
          This method is suitable for low-dimensional data (e.g. 3D
          point clouds)

        * 'nn-descent' is an approximate approach from paper
          `Efficient k-nearest neighbor graph construction for generic similarity
          measures <https://www.cs.princeton.edu/cass/papers/www11.pdf>`_. This method
          will search for nearest neighbor candidates in "neighbors' neighbors".

        (default: 'bruteforce-blas')
    dist : str, optional
        The distance metric used to compute distance between points. It can be the following
        metrics:
        * 'euclidean': Use Euclidean distance (L2 norm) :math:`\sqrt{\sum_{i} (x_{i} - y_{i})^{2}}`.
        * 'cosine': Use cosine distance.
        (default: 'euclidean')
    exclude_self : bool, optional
        If True, the output graph will not contain self loop edges, and each node will not
        be counted as one of its own k neighbors.  If False, the output graph will contain
        self loop edges, and a node will be counted as one of its own k neighbors.

    Returns
    -------
    DGLGraph
        The batched graph. The node IDs are in the same order as :attr:`x`.

    Examples
    --------

    The following examples use PyTorch backend.

    >>> import dgl
    >>> import torch

    In the example below, the first point set has three points
    and the second point set has four points.

    >>> # Features/coordinates of the first point set
    >>> x1 = torch.tensor([[0.0, 0.5, 0.2],
    ...                    [0.1, 0.3, 0.2],
    ...                    [0.4, 0.2, 0.2]])
    >>> # Features/coordinates of the second point set
    >>> x2 = torch.tensor([[0.3, 0.2, 0.1],
    ...                    [0.5, 0.2, 0.3],
    ...                    [0.1, 0.1, 0.2],
    ...                    [0.6, 0.3, 0.3]])
    >>> x = torch.cat([x1, x2], dim=0)
    >>> segs = [x1.shape[0], x2.shape[0]]
    >>> knn_g = dgl.segmented_knn_graph(x, 2, segs)
    >>> knn_g.edges()
    (tensor([0, 0, 1, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6]),
     tensor([0, 1, 0, 1, 2, 2, 3, 5, 4, 6, 3, 5, 4, 6]))
    """
    if exclude_self:
        # add 1 to k, for the self edge, since it will be removed
        k = k + 1

    # check invalid k
    if k <= 0:
        raise DGLError("Invalid k value. expect k > 0, got k = {}".format(k))

    # check empty point set
    if F.shape(x)[0] == 0:
        raise DGLError("Find empty point set")

    if algorithm == "bruteforce-blas":
        result = _segmented_knn_graph_blas(x, k, segs, dist=dist)
    else:
        out = knn(k, x, segs, algorithm=algorithm, dist=dist)
        row, col = out[1], out[0]
        result = convert.graph((row, col))

    num_nodes = F.tensor(segs, dtype=F.int64).to(F.context(x))
    result.set_batch_num_nodes(num_nodes)
    # if any segment is too small for k, all algorithms reduce k for all segments
    clamped_k = min(k, np.min(segs))
    result.set_batch_num_edges(clamped_k * num_nodes)

    if exclude_self:
        # remove_self_loop will update batch_num_edges as needed
        result = remove_self_loop(result)

        # If there were more than k(+1) coincident points, there may not have been self loops on
        # all nodes, in which case there would still be one too many out edges on some nodes.
        # However, if every node had a self edge, the common case, every node would still have the
        # same degree as each other, so we can check that condition easily.
        # The -1 is for the self edge removal.
        clamped_k = min(k, np.min(segs)) - 1
        if result.num_edges() != clamped_k * result.num_nodes():
            # edges on any nodes with too high degree should all be length zero,
            # so pick an arbitrary one to remove from each such node
            degrees = result.in_degrees()
            node_indices = F.nonzero_1d(degrees > clamped_k)
            edges_to_remove_graph = sample_neighbors(
                result, node_indices, 1, edge_dir="in"
            )
            edge_ids = edges_to_remove_graph.edata[EID]
            result = remove_edges(result, edge_ids)

    return result


def _segmented_knn_graph_blas(x, k, segs, dist="euclidean"):
    r"""Construct multiple graphs from multiple sets of points according to
    k-nearest-neighbor (KNN).

    This function first compute the distance matrix using BLAS matrix multiplication
    operation provided by backend frameworks. Then use topk algorithm to get
    k-nearest neighbors.

    Parameters
    ----------
    x : Tensor
        Coordinates/features of points. Must be 2D. It can be either on CPU or GPU.
    k : int
        The number of nearest neighbors per node.
    segs : list[int]
        Number of points in each point set. The numbers in :attr:`segs`
        must sum up to the number of rows in :attr:`x`.
    dist : str, optional
        The distance metric used to compute distance between points. It can be the following
        metrics:
        * 'euclidean': Use Euclidean distance (L2 norm) :math:`\sqrt{\sum_{i} (x_{i} - y_{i})^{2}}`.
        * 'cosine': Use cosine distance.
        (default: 'euclidean')
    """
    # if use cosine distance, normalize input points first
    # thus we can use euclidean distance to find knn equivalently.
    if dist == "cosine":
        l2_norm = lambda v: F.sqrt(F.sum(v * v, dim=1, keepdims=True))
        x = x / (l2_norm(x) + 1e-5)

    n_total_points, _ = F.shape(x)
    offset = np.insert(np.cumsum(segs), 0, 0)
    min_seg_size = np.min(segs)
    if k > min_seg_size:
        dgl_warning(
            "'k' should be less than or equal to the number of points in 'x'"
            "expect k <= {0}, got k = {1}, use k = {0}".format(min_seg_size, k)
        )
        k = min_seg_size

    h_list = F.split(x, segs, 0)
    src = [
        F.argtopk(pairwise_squared_distance(h_g), k, 1, descending=False)
        + int(offset[i])
        for i, h_g in enumerate(h_list)
    ]
    src = F.cat(src, 0)
    ctx = F.context(x)
    dst = F.repeat(F.arange(0, n_total_points, ctx=ctx), k, dim=0)
    return convert.graph((F.reshape(src, (-1,)), F.reshape(dst, (-1,))))


def _nndescent_knn_graph(
    x,
    k,
    segs,
    num_iters=None,
    max_candidates=None,
    delta=0.001,
    sample_rate=0.5,
    dist="euclidean",
):
    r"""Construct multiple graphs from multiple sets of points according to
    **approximate** k-nearest-neighbor using NN-descent algorithm from paper
    `Efficient k-nearest neighbor graph construction for generic similarity
    measures <https://www.cs.princeton.edu/cass/papers/www11.pdf>`_.

    Parameters
    ----------
    x : Tensor
        Coordinates/features of points. Must be 2D. It can be either on CPU or GPU.
    k : int
        The number of nearest neighbors per node.
    segs : list[int]
        Number of points in each point set. The numbers in :attr:`segs`
        must sum up to the number of rows in :attr:`x`.
    num_iters : int, optional
        The maximum number of NN-descent iterations to perform. A value will be
        chosen based on the size of input by default.
        (Default: None)
    max_candidates : int, optional
        The maximum number of candidates to be considered during one iteration.
        Larger values will provide more accurate search results later, but
        potentially at non-negligible computation cost. A value will be chosen
        based on the number of neighbors by default.
        (Default: None)
    delta : float, optional
        A value controls the early abort. This function will abort if
        :math:`k * N * delta > c`, where :math:`N` is the number of points,
        :math:`c` is the number of updates during last iteration.
        (Default: 0.001)
    sample_rate : float, optional
        A value controls how many candidates sampled. It should be a float value
        between 0 and 1. Larger values will provide higher accuracy and converge
        speed but with higher time cost.
        (Default: 0.5)
    dist : str, optional
        The distance metric used to compute distance between points. It can be the following
        metrics:
        * 'euclidean': Use Euclidean distance (L2 norm) :math:`\sqrt{\sum_{i} (x_{i} - y_{i})^{2}}`.
        * 'cosine': Use cosine distance.
        (default: 'euclidean')

    Returns
    -------
    DGLGraph
        The graph. The node IDs are in the same order as :attr:`x`.
    """
    num_points, _ = F.shape(x)
    if isinstance(segs, (tuple, list)):
        segs = F.tensor(segs)
    segs = F.copy_to(segs, F.context(x))

    if max_candidates is None:
        max_candidates = min(60, k)
    if num_iters is None:
        num_iters = max(10, int(round(np.log2(num_points))))
    max_candidates = int(sample_rate * max_candidates)

    # if use cosine distance, normalize input points first
    # thus we can use euclidean distance to find knn equivalently.
    if dist == "cosine":
        l2_norm = lambda v: F.sqrt(F.sum(v * v, dim=1, keepdims=True))
        x = x / (l2_norm(x) + 1e-5)

    # k must less than or equal to min(segs)
    if k > F.min(segs, dim=0):
        raise DGLError(
            "'k' must be less than or equal to the number of points in 'x'"
            "expect 'k' <= {}, got 'k' = {}".format(F.min(segs, dim=0), k)
        )
    if delta < 0 or delta > 1:
        raise DGLError("'delta' must in [0, 1], got 'delta' = {}".format(delta))

    offset = F.zeros((F.shape(segs)[0] + 1,), F.dtype(segs), F.context(segs))
    offset[1:] = F.cumsum(segs, dim=0)
    out = F.zeros((2, num_points * k), F.dtype(segs), F.context(segs))

    # points, offsets, out, k, num_iters, max_candidates, delta
    _CAPI_DGLNNDescent(
        F.to_dgl_nd(x),
        F.to_dgl_nd(offset),
        F.zerocopy_to_dgl_ndarray_for_write(out),
        k,
        num_iters,
        max_candidates,
        delta,
    )
    return out


def knn(
    k, x, x_segs, y=None, y_segs=None, algorithm="bruteforce", dist="euclidean"
):
    r"""For each element in each segment in :attr:`y`, find :attr:`k` nearest
    points in the same segment in :attr:`x`. If :attr:`y` is None, perform a self-query
    over :attr:`x`.

    This function allows multiple point sets with different capacity. The points
    from different sets are stored contiguously in the :attr:`x` and :attr:`y` tensor.
    :attr:`x_segs` and :attr:`y_segs` specifies the number of points in each point set.

    Parameters
    ----------
    k : int
        The number of nearest neighbors per node.
    x : Tensor
        The point coordinates in x. It can be either on CPU or GPU (must be the
        same as :attr:`y`). Must be 2D.
    x_segs : Union[List[int], Tensor]
        Number of points in each point set in :attr:`x`. The numbers in :attr:`x_segs`
        must sum up to the number of rows in :attr:`x`.
    y : Tensor, optional
        The point coordinates in y. It can be either on CPU or GPU (must be the
        same as :attr:`x`). Must be 2D.
        (default: None)
    y_segs : Union[List[int], Tensor], optional
        Number of points in each point set in :attr:`y`. The numbers in :attr:`y_segs`
        must sum up to the number of rows in :attr:`y`.
        (default: None)
    algorithm : str, optional
        Algorithm used to compute the k-nearest neighbors.

        * 'bruteforce' will compute distances pair by pair and
          directly select the k-nearest neighbors during distance
          computation. This method is slower than 'bruteforce-blas'
          but has less memory overhead (i.e., :math:`O(Nk)` where :math:`N`
          is the number of points, :math:`k` is the number of nearest
          neighbors per node) since we do not need to store all distances.

        * 'bruteforce-sharemem' (CUDA only) is similar to 'bruteforce'
          but use shared memory in CUDA devices for buffer. This method is
          faster than 'bruteforce' when the dimension of input points
          is not large. This method is only available on CUDA device.

        * 'kd-tree' will use the kd-tree algorithm (CPU only).
          This method is suitable for low-dimensional data (e.g. 3D
          point clouds)

        * 'nn-descent' is an approximate approach from paper
          `Efficient k-nearest neighbor graph construction for generic similarity
          measures <https://www.cs.princeton.edu/cass/papers/www11.pdf>`_. This method
          will search for nearest neighbor candidates in "neighbors' neighbors".

        Note: Currently, 'nn-descent' only supports self-query cases, i.e. :attr:`y` is None.
        (default: 'bruteforce')
    dist : str, optional
        The distance metric used to compute distance between points. It can be the following
        metrics:
        * 'euclidean': Use Euclidean distance (L2 norm) :math:`\sqrt{\sum_{i} (x_{i} - y_{i})^{2}}`.
        * 'cosine': Use cosine distance.
        (default: 'euclidean')

    Returns
    -------
    Tensor
        Tensor with size `(2, k * num_points(y))`
        The first subtensor contains point indexs in :attr:`y`. The second subtensor contains
        point indexs in :attr:`x`
    """
    # TODO(lygztq) add support for querying different point sets using nn-descent.
    if algorithm == "nn-descent":
        if y is not None or y_segs is not None:
            raise DGLError(
                "Currently 'nn-descent' only supports self-query cases."
            )
        return _nndescent_knn_graph(x, k, x_segs, dist=dist)

    # self query
    if y is None:
        y = x
        y_segs = x_segs

    assert F.context(x) == F.context(y)
    if isinstance(x_segs, (tuple, list)):
        x_segs = F.tensor(x_segs)
    if isinstance(y_segs, (tuple, list)):
        y_segs = F.tensor(y_segs)
    x_segs = F.copy_to(x_segs, F.context(x))
    y_segs = F.copy_to(y_segs, F.context(y))

    # k shoule be less than or equal to min(x_segs)
    min_num_points = F.min(x_segs, dim=0)
    if k > min_num_points:
        dgl_warning(
            "'k' should be less than or equal to the number of points in 'x'"
            "expect k <= {0}, got k = {1}, use k = {0}".format(
                min_num_points, k
            )
        )
        k = F.as_scalar(min_num_points)

    # invalid k
    if k <= 0:
        raise DGLError("Invalid k value. expect k > 0, got k = {}".format(k))

    # empty point set
    if F.shape(x)[0] == 0 or F.shape(y)[0] == 0:
        raise DGLError("Find empty point set")

    dist = dist.lower()
    dist_metric_list = ["euclidean", "cosine"]
    if dist not in dist_metric_list:
        raise DGLError(
            "Only {} are supported for distance"
            "computation, got {}".format(dist_metric_list, dist)
        )

    x_offset = F.zeros(
        (F.shape(x_segs)[0] + 1,), F.dtype(x_segs), F.context(x_segs)
    )
    x_offset[1:] = F.cumsum(x_segs, dim=0)
    y_offset = F.zeros(
        (F.shape(y_segs)[0] + 1,), F.dtype(y_segs), F.context(y_segs)
    )
    y_offset[1:] = F.cumsum(y_segs, dim=0)

    out = F.zeros((2, F.shape(y)[0] * k), F.dtype(x_segs), F.context(x_segs))

    # if use cosine distance, normalize input points first
    # thus we can use euclidean distance to find knn equivalently.
    if dist == "cosine":
        l2_norm = lambda v: F.sqrt(F.sum(v * v, dim=1, keepdims=True))
        x = x / (l2_norm(x) + 1e-5)
        y = y / (l2_norm(y) + 1e-5)

    _CAPI_DGLKNN(
        F.to_dgl_nd(x),
        F.to_dgl_nd(x_offset),
        F.to_dgl_nd(y),
        F.to_dgl_nd(y_offset),
        k,
        F.zerocopy_to_dgl_ndarray_for_write(out),
        algorithm,
    )
    return out


def to_bidirected(g, copy_ndata=False, readonly=None):
    r"""Convert the graph to a bi-directional simple graph and return.

    For an input graph :math:`G`, return a new graph :math:`G'` such that an edge
    :math:`(u, v)\in G'` exists if and only if there exists an edge
    :math:`(v, u)\in G`. The resulting graph :math:`G'` is a simple graph,
    meaning there is no parallel edge.

    The operation only works for edges whose two endpoints belong to the same node type.
    DGL will raise error if the input graph is heterogeneous and contains edges
    with different types of endpoints.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    copy_ndata: bool, optional
        If True, the node features of the bidirected graph are copied from the
        original graph. If False, the bidirected graph will not have any node features.
        (Default: False)
    readonly : bool
        **DEPRECATED**.

    Returns
    -------
    DGLGraph
        The bidirected graph

    Notes
    -----
    If :attr:`copy_ndata` is True, the resulting graph will share the node feature
    tensors with the input graph. Hence, users should try to avoid in-place operations
    which will be visible to both graphs.

    This function discards the batch information. Please use
    :func:`dgl.DGLGraph.set_batch_num_nodes`
    and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
    to maintain the information.

    Examples
    --------
    The following examples use PyTorch backend.

    >>> import dgl
    >>> import torch as th
    >>> g = dgl.graph((th.tensor([0, 1, 2]), th.tensor([1, 2, 0])))
    >>> bg1 = dgl.to_bidirected(g)
    >>> bg1.edges()
    (tensor([0, 1, 2, 1, 2, 0]), tensor([1, 2, 0, 0, 1, 2]))

    The graph already have i->j and j->i

    >>> g = dgl.graph((th.tensor([0, 1, 2, 0]), th.tensor([1, 2, 0, 2])))
    >>> bg1 = dgl.to_bidirected(g)
    >>> bg1.edges()
    (tensor([0, 1, 2, 1, 2, 0]), tensor([1, 2, 0, 0, 1, 2]))

    **Heterogeneous graphs with Multiple Edge Types**

    >>> g = dgl.heterograph({
    ...     ('user', 'wins', 'user'): (th.tensor([0, 2, 0, 2]), th.tensor([1, 1, 2, 0])),
    ...     ('user', 'follows', 'user'): (th.tensor([1, 2, 1]), th.tensor([2, 1, 1]))
    ... })
    >>> bg1 = dgl.to_bidirected(g)
    >>> bg1.edges(etype='wins')
    (tensor([0, 0, 1, 1, 2, 2]), tensor([1, 2, 0, 2, 0, 1]))
    >>> bg1.edges(etype='follows')
    (tensor([1, 1, 2]), tensor([1, 2, 1]))
    """
    if readonly is not None:
        dgl_warning(
            "Parameter readonly is deprecated"
            "There will be no difference between readonly and non-readonly DGLGraph"
        )

    for c_etype in g.canonical_etypes:
        if c_etype[0] != c_etype[2]:
            assert False, (
                "to_bidirected is not well defined for "
                "unidirectional bipartite graphs"
                ", but {} is unidirectional bipartite".format(c_etype)
            )

    g = add_reverse_edges(g, copy_ndata=copy_ndata, copy_edata=False)
    g = to_simple(
        g, return_counts=None, copy_ndata=copy_ndata, copy_edata=False
    )
    return g


def add_reverse_edges(
    g,
    readonly=None,
    copy_ndata=True,
    copy_edata=False,
    ignore_bipartite=False,
    exclude_self=True,
):
    r"""Add a reversed edge for each edge in the input graph and return a new graph.

    For a graph with edges :math:`(i_1, j_1), \cdots, (i_n, j_n)`, this
    function creates a new graph with edges
    :math:`(i_1, j_1), \cdots, (i_n, j_n), (j_1, i_1), \cdots, (j_n, i_n)`.

    The returned graph may have duplicate edges. To create a bidirected graph without
    duplicate edges, use :func:`to_bidirected`.

    The operation only works for edges whose two endpoints belong to the same node type.
    DGL will raise error if the input graph is heterogeneous and contains edges
    with different types of endpoints. If :attr:`ignore_bipartite` is true, DGL will
    ignore those edges instead.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    readonly : bool, default to be True
        Deprecated. There will be no difference between readonly and non-readonly
    copy_ndata: bool, optional
        If True, the node features of the new graph are copied from
        the original graph. If False, the new graph will not have any
        node features.

        (Default: True)
    copy_edata: bool, optional
        If True, the features of the reversed edges will be identical to
        the original ones.

        If False, the new graph will not have any edge features.

        (Default: False)
    ignore_bipartite: bool, optional
        If True, unidirectional bipartite graphs are ignored and
        no error is raised. If False, an error will be raised if
        an edge type of the input heterogeneous graph is for a unidirectional
        bipartite graph.
    exclude_self: bool, optional
        If True, it does not add reverse edges for self-loops, which is likely
        meaningless in most cases.

    Returns
    -------
    DGLGraph
        The graph with reversed edges added.

    Notes
    -----
    If :attr:`copy_ndata` is True, the resulting graph will share the node feature
    tensors with the input graph. Hence, users should try to avoid in-place operations
    which will be visible to both graphs. On the contrary, the two graphs do not share
    the same edge feature storage.

    This function discards the batch information. Please use
    :func:`dgl.DGLGraph.set_batch_num_nodes`
    and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
    to maintain the information.

    Examples
    --------
    **Homogeneous graphs**

    >>> g = dgl.graph((th.tensor([0, 0]), th.tensor([0, 1])))
    >>> bg1 = dgl.add_reverse_edges(g)
    >>> bg1.edges()
    (tensor([0, 0, 0, 1]), tensor([0, 1, 0, 0]))

    **Heterogeneous graphs**

    >>> g = dgl.heterograph({
    >>>     ('user', 'wins', 'user'): (th.tensor([0, 2, 0, 2, 2]), th.tensor([1, 1, 2, 1, 0])),
    >>>     ('user', 'plays', 'game'): (th.tensor([1, 2, 1]), th.tensor([2, 1, 1])),
    >>>     ('user', 'follows', 'user'): (th.tensor([1, 2, 1), th.tensor([0, 0, 0]))
    >>> })
    >>> g.nodes['game'].data['hv'] = th.ones(3, 1)
    >>> g.edges['wins'].data['h'] = th.tensor([0, 1, 2, 3, 4])

    The :func:`add_reverse_edges` operation is applied to the edge type
    ``('user', 'wins', 'user')`` and the edge type ``('user', 'follows', 'user')``.
    The edge type ``('user', 'plays', 'game')`` is ignored.  Both the node features and
    edge features are shared.

    >>> bg = dgl.add_reverse_edges(g, copy_ndata=True,
                               copy_edata=True, ignore_bipartite=True)
    >>> bg.edges(('user', 'wins', 'user'))
    (tensor([0, 2, 0, 2, 2, 1, 1, 2, 1, 0]), tensor([1, 1, 2, 1, 0, 0, 2, 0, 2, 2]))
    >>> bg.edges(('user', 'follows', 'user'))
    (tensor([1, 2, 1, 0, 0, 0]), tensor([0, 0, 0, 1, 2, 1]))
    >>> bg.edges(('user', 'plays', 'game'))
    (th.tensor([1, 2, 1]), th.tensor([2, 1, 1]))
    >>> bg.nodes['game'].data['hv']
    tensor([0, 0, 0])
    >>> bg.edges[('user', 'wins', 'user')].data['h']
    th.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    """
    if readonly is not None:
        dgl_warning(
            "Parameter readonly is deprecated"
            "There will be no difference between readonly and non-readonly DGLGraph"
        )

    # get node cnt for each ntype
    num_nodes_dict = {}
    for ntype in g.ntypes:
        num_nodes_dict[ntype] = g.num_nodes(ntype)

    canonical_etypes = g.canonical_etypes
    num_nodes_dict = {ntype: g.num_nodes(ntype) for ntype in g.ntypes}
    subgs = {}
    rev_eids = {}

    def add_for_etype(etype):
        u, v = g.edges(form="uv", order="eid", etype=etype)
        rev_u, rev_v = v, u
        eid = F.copy_to(F.arange(0, g.num_edges(etype)), g.device)
        if exclude_self:
            self_loop_mask = F.equal(rev_u, rev_v)
            non_self_loop_mask = F.logical_not(self_loop_mask)
            rev_u = F.boolean_mask(rev_u, non_self_loop_mask)
            rev_v = F.boolean_mask(rev_v, non_self_loop_mask)
            non_self_loop_eid = F.boolean_mask(eid, non_self_loop_mask)
            rev_eids[etype] = F.cat([eid, non_self_loop_eid], 0)
        else:
            rev_eids[etype] = F.cat([eid, eid], 0)
        subgs[etype] = (F.cat([u, rev_u], dim=0), F.cat([v, rev_v], dim=0))

    # fast path
    if ignore_bipartite is False:
        for c_etype in canonical_etypes:
            if c_etype[0] != c_etype[2]:
                assert False, (
                    "add_reverse_edges is not well defined for "
                    "unidirectional bipartite graphs"
                    ", but {} is unidirectional bipartite".format(c_etype)
                )
            add_for_etype(c_etype)

        new_g = convert.heterograph(subgs, num_nodes_dict=num_nodes_dict)
    else:
        for c_etype in canonical_etypes:
            if c_etype[0] != c_etype[2]:
                u, v = g.edges(form="uv", order="eid", etype=c_etype)
                subgs[c_etype] = (u, v)
            else:
                add_for_etype(c_etype)

        new_g = convert.heterograph(subgs, num_nodes_dict=num_nodes_dict)

    # handle features
    if copy_ndata:
        node_frames = utils.extract_node_subframes(g, None)
        utils.set_new_frames(new_g, node_frames=node_frames)

    if copy_edata:
        # find indices
        eids = []
        for c_etype in canonical_etypes:
            if c_etype[0] != c_etype[2]:
                eids.append(
                    F.copy_to(F.arange(0, g.num_edges(c_etype)), new_g.device)
                )
            else:
                eids.append(rev_eids[c_etype])

        edge_frames = utils.extract_edge_subframes(g, eids)
        utils.set_new_frames(new_g, edge_frames=edge_frames)

    return new_g


def line_graph(g, backtracking=True, shared=False):
    """Return the line graph of this graph.

    The line graph ``L(G)`` of a given graph ``G`` is defined as another graph where
    the nodes in ``L(G)`` correspond to the edges in ``G``.  For any pair of edges ``(u, v)``
    and ``(v, w)`` in ``G``, the corresponding node of edge ``(u, v)`` in ``L(G)`` will
    have an edge connecting to the corresponding node of edge ``(v, w)``.

    Parameters
    ----------
    g : DGLGraph
        Input graph.  Must be homogeneous.
    backtracking : bool, optional
        If False, the line graph node corresponding to edge ``(u, v)`` will not have
        an edge connecting to the line graph node corresponding to edge ``(v, u)``.

        Default: True.
    shared : bool, optional
        Whether to copy the edge features of the original graph as the node features
        of the result line graph.

    Returns
    -------
    G : DGLGraph
        The line graph of this graph.

    Notes
    -----
    * If :attr:`shared` is True, the node features of the resulting graph share the same
      storage with the edge features of the input graph. Hence, users should try to
      avoid in-place operations which will be visible to both graphs.
    * The function supports input graph on GPU but copies it to CPU during computation.
    * This function discards the batch information. Please use
      :func:`dgl.DGLGraph.set_batch_num_nodes`
      and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
      to maintain the information.

    Examples
    --------
    Assume that the graph has the following adjacency matrix: ::

       A = [[0, 0, 1],
            [1, 0, 1],
            [1, 1, 0]]

    >>> g = dgl.graph(([0, 1, 1, 2, 2],[2, 0, 2, 0, 1]), 'user', 'follows')
    >>> lg = g.line_graph()
    >>> lg
    Graph(num_nodes=5, num_edges=8,
    ndata_schemes={}
    edata_schemes={})
    >>> lg.edges()
    (tensor([0, 0, 1, 2, 2, 3, 4, 4]), tensor([3, 4, 0, 3, 4, 0, 1, 2]))
    >>> lg = g.line_graph(backtracking=False)
    >>> lg
    Graph(num_nodes=5, num_edges=4,
    ndata_schemes={}
    edata_schemes={})
    >>> lg.edges()
    (tensor([0, 1, 2, 4]), tensor([4, 0, 3, 1]))
    """
    assert g.is_homogeneous, "only homogeneous graph is supported"

    dev = g.device
    lg = DGLGraph(
        _CAPI_DGLHeteroLineGraph(g._graph.copy_to(nd.cpu()), backtracking)
    )
    lg = lg.to(dev)
    if shared:
        new_frames = utils.extract_edge_subframes(g, None)
        utils.set_new_frames(lg, node_frames=new_frames)

    return lg


DGLGraph.line_graph = utils.alias_func(line_graph)


def khop_adj(g, k):
    """Return the matrix of :math:`A^k` where :math:`A` is the adjacency matrix of the graph
    :math:`g`.

    The returned matrix is a 32-bit float dense matrix on CPU. The graph must be homogeneous.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    k : int
        The :math:`k` in :math:`A^k`.

    Returns
    -------
    Tensor
        The returned tensor.

    Examples
    --------
    >>> import dgl
    >>> g = dgl.graph(([0,1,2,3,4,0,1,2,3,4], [0,1,2,3,4,1,2,3,4,0]))
    >>> dgl.khop_adj(g, 1)
    tensor([[1., 1., 0., 0., 0.],
            [0., 1., 1., 0., 0.],
            [0., 0., 1., 1., 0.],
            [0., 0., 0., 1., 1.],
            [1., 0., 0., 0., 1.]])
    >>> dgl.khop_adj(g, 3)
    tensor([[1., 3., 3., 1., 0.],
            [0., 1., 3., 3., 1.],
            [1., 0., 1., 3., 3.],
            [3., 1., 0., 1., 3.],
            [3., 3., 1., 0., 1.]])
    """
    assert g.is_homogeneous, "only homogeneous graph is supported"
    adj_k = (
        g.adj_external(transpose=False, scipy_fmt=g.formats()["created"][0])
        ** k
    )
    return F.tensor(adj_k.todense().astype(np.float32))


def khop_graph(g, k, copy_ndata=True):
    """Return the graph whose edges connect the :attr:`k`-hop neighbors of the original graph.

    More specifically, an edge from node ``u`` and node ``v`` exists in the new graph if
    and only if a path with length :attr:`k` exists from node ``u`` to node ``v`` in the
    original graph.

    The adjacency matrix of the returned graph is :math:`A^k`
    (where :math:`A` is the adjacency matrix of :math:`g`).

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    k : int
        The :math:`k` in `k`-hop graph.
    copy_ndata: bool, optional
        If True, the node features of the new graph are copied from the
        original graph.

        If False, the new graph will not have any node features.

        (Default: True)

    Returns
    -------
    DGLGraph
        The returned graph.

    Notes
    -----
    If :attr:`copy_ndata` is True, the resulting graph will share the node feature
    tensors with the input graph. Hence, users should try to avoid in-place operations
    which will be visible to both graphs.

    This function discards the batch information. Please use
    :func:`dgl.DGLGraph.set_batch_num_nodes`
    and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
    to maintain the information.

    Examples
    --------

    Below gives an easy example:

    >>> import dgl
    >>> g = dgl.graph(([0, 1], [1, 2]))
    >>> g_2 = dgl.transforms.khop_graph(g, 2)
    >>> print(g_2.edges())
    (tensor([0]), tensor([2]))

    A more complicated example:

    >>> import dgl
    >>> g = dgl.graph(([0,1,2,3,4,0,1,2,3,4], [0,1,2,3,4,1,2,3,4,0]))
    >>> dgl.khop_graph(g, 1)
    DGLGraph(num_nodes=5, num_edges=10,
             ndata_schemes={}
             edata_schemes={})
    >>> dgl.khop_graph(g, 3)
    DGLGraph(num_nodes=5, num_edges=40,
             ndata_schemes={}
             edata_schemes={})
    """
    assert g.is_homogeneous, "only homogeneous graph is supported"
    n = g.num_nodes()
    adj_k = (
        g.adj_external(transpose=False, scipy_fmt=g.formats()["created"][0])
        ** k
    )
    adj_k = adj_k.tocoo()
    multiplicity = adj_k.data
    row = np.repeat(adj_k.row, multiplicity)
    col = np.repeat(adj_k.col, multiplicity)
    # TODO(zihao): we should support creating multi-graph from scipy sparse matrix
    # in the future.
    new_g = convert.graph(
        (row, col), num_nodes=n, idtype=g.idtype, device=g.device
    )

    # handle ndata
    if copy_ndata:
        node_frames = utils.extract_node_subframes(g, None)
        utils.set_new_frames(new_g, node_frames=node_frames)

    return new_g


def reverse(
    g, copy_ndata=True, copy_edata=False, *, share_ndata=None, share_edata=None
):
    r"""Return a new graph with every edges being the reverse ones in the input graph.

    The reverse (also called converse, transpose) of a graph with edges
    :math:`(i_1, j_1), (i_2, j_2), \cdots` of type ``(U, E, V)`` is a new graph with edges
    :math:`(j_1, i_1), (j_2, i_2), \cdots` of type ``(V, E, U)``.

    The returned graph shares the data structure with the original graph, i.e. dgl.reverse
    will not create extra storage for the reversed graph.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    copy_ndata: bool, optional
        If True, the node features of the reversed graph are copied from the
        original graph. If False, the reversed graph will not have any node features.
        (Default: True)
    copy_edata: bool, optional
        If True, the edge features of the reversed graph are copied from the
        original graph. If False, the reversed graph will not have any edge features.
        (Default: False)

    Return
    ------
    DGLGraph
        The reversed graph.

    Notes
    -----
    If :attr:`copy_ndata` or :attr:`copy_edata` is True,
    the resulting graph will share the node or edge feature
    tensors with the input graph. Hence, users should try to avoid in-place operations
    which will be visible to both graphs.

    This function discards the batch information. Please use
    :func:`dgl.DGLGraph.set_batch_num_nodes`
    and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
    to maintain the information.

    Examples
    --------
    **Homogeneous graphs**

    Create a graph to reverse.

    >>> import dgl
    >>> import torch as th
    >>> g = dgl.graph((th.tensor([0, 1, 2]), th.tensor([1, 2, 0])))
    >>> g.ndata['h'] = th.tensor([[0.], [1.], [2.]])
    >>> g.edata['h'] = th.tensor([[3.], [4.], [5.]])

    Reverse the graph.

    >>> rg = dgl.reverse(g, copy_edata=True)
    >>> rg.ndata['h']
    tensor([[0.],
            [1.],
            [2.]])

    The i-th edge in the reversed graph corresponds to the i-th edge in the
    original graph. When :attr:`copy_edata` is True, they have the same features.

    >>> rg.edges()
    (tensor([1, 2, 0]), tensor([0, 1, 2]))
    >>> rg.edata['h']
    tensor([[3.],
            [4.],
            [5.]])

    **Heterogenenous graphs**

    >>> g = dgl.heterograph({
    ...     ('user', 'follows', 'user'): (th.tensor([0, 2]), th.tensor([1, 2])),
    ...     ('user', 'plays', 'game'): (th.tensor([1, 2, 1]), th.tensor([2, 1, 1]))
    ... })
    >>> g.nodes['game'].data['hv'] = th.ones(3, 1)
    >>> g.edges['plays'].data['he'] = th.zeros(3, 1)

    The resulting graph will have edge types
    ``('user', 'follows', 'user)`` and ``('game', 'plays', 'user')``.

    >>> rg = dgl.reverse(g, copy_ndata=True)
    >>> rg
    Graph(num_nodes={'game': 3, 'user': 3},
          num_edges={('user', 'follows', 'user'): 2, ('game', 'plays', 'user'): 3},
          metagraph=[('user', 'user'), ('game', 'user')])
    >>> rg.edges(etype='follows')
    (tensor([1, 2]), tensor([0, 2]))
    >>> rg.edges(etype='plays')
    (tensor([2, 1, 1]), tensor([1, 2, 1]))
    >>> rg.nodes['game'].data['hv']
    tensor([[1.],
            [1.],
            [1.]])
    >>> rg.edges['plays'].data
    {}
    """
    if share_ndata is not None:
        dgl_warning("share_ndata argument has been renamed to copy_ndata.")
        copy_ndata = share_ndata
    if share_edata is not None:
        dgl_warning("share_edata argument has been renamed to copy_edata.")
        copy_edata = share_edata
    if g.is_block:
        # TODO(0.5 release, xiangsx) need to handle BLOCK
        # currently reversing a block results in undefined behavior
        raise DGLError("Reversing a block graph is not supported.")
    gidx = g._graph.reverse()
    new_g = DGLGraph(gidx, g.ntypes, g.etypes)

    # handle ndata
    if copy_ndata:
        # for each ntype
        for ntype in g.ntypes:
            new_g.nodes[ntype].data.update(g.nodes[ntype].data)

    # handle edata
    if copy_edata:
        # for each etype
        for utype, etype, vtype in g.canonical_etypes:
            new_g.edges[vtype, etype, utype].data.update(
                g.edges[utype, etype, vtype].data
            )

    return new_g


DGLGraph.reverse = utils.alias_func(reverse)


def to_simple_graph(g):
    """Convert the graph to a simple graph with no multi-edge.

    DEPRECATED: renamed to dgl.to_simple

    Parameters
    ----------
    g : DGLGraph
        The input graph.

    Returns
    -------
    DGLGraph
        A simple graph.

    Notes
    -----

    This function discards the batch information. Please use
    :func:`dgl.DGLGraph.set_batch_num_nodes`
    and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
    to maintain the information.
    """
    dgl_warning("dgl.to_simple_graph is renamed to dgl.to_simple in v0.5.")
    return to_simple(g)


def laplacian_lambda_max(g):
    """Return the largest eigenvalue of the normalized symmetric Laplacian of a graph.

    If the graph is batched from multiple graphs, return the list of the largest eigenvalue
    for each graph instead.

    Parameters
    ----------
    g : DGLGraph
        The input graph, it must be a bi-directed homogeneous graph, i.e., every edge
        should have an accompanied reverse edge in the graph.
        The graph can be batched from multiple graphs.

    Returns
    -------
    list[float]
        A list where the i-th item indicates the largest eigenvalue
        of i-th graph in :attr:`g`.

        In the case where the function takes a single graph, it will return a list
        consisting of a single element.

    Examples
    --------
    >>> import dgl
    >>> g = dgl.graph(([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], [1, 2, 3, 4, 0, 4, 0, 1, 2, 3]))
    >>> dgl.laplacian_lambda_max(g)
    [1.809016994374948]
    """
    g_arr = batch.unbatch(g)
    rst = []
    for g_i in g_arr:
        n = g_i.num_nodes()
        adj = g_i.adj_external(
            transpose=True, scipy_fmt=g_i.formats()["created"][0]
        ).astype(float)
        norm = sparse.diags(
            F.asnumpy(g_i.in_degrees()).clip(1) ** -0.5, dtype=float
        )
        laplacian = sparse.eye(n) - norm * adj * norm
        rst.append(
            scipy.sparse.linalg.eigs(
                laplacian, 1, which="LM", return_eigenvectors=False
            )[0].real
        )
    return rst


def metapath_reachable_graph(g, metapath):
    """Return a graph where the successors of any node ``u`` are nodes reachable from ``u`` by
    the given metapath.

    If the beginning node type ``s`` and ending node type ``t`` are the same, it will return
    a homogeneous graph with node type ``s = t``.  Otherwise, a unidirectional bipartite graph
    with source node type ``s`` and destination node type ``t`` is returned.

    In both cases, two nodes ``u`` and ``v`` will be connected with an edge ``(u, v)`` if
    there exists one path matching the metapath from ``u`` to ``v``.

    The result graph keeps the node set of type ``s`` and ``t`` in the original graph even if
    they might have no neighbor.

    The features of the source/destination node type in the original graph would be copied to
    the new graph.

    Parameters
    ----------
    g : DGLGraph
        The input graph
    metapath : list[str or tuple of str]
        Metapath in the form of a list of edge types

    Returns
    -------
    DGLGraph
        A homogeneous or unidirectional bipartite graph. It will be on CPU regardless of
        whether the input graph is on CPU or GPU.

    Notes
    -----

    This function discards the batch information. Please use
    :func:`dgl.DGLGraph.set_batch_num_nodes`
    and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
    to maintain the information.

    Examples
    --------
    >>> g = dgl.heterograph({
    ...     ('A', 'AB', 'B'): ([0, 1, 2], [1, 2, 3]),
    ...     ('B', 'BA', 'A'): ([1, 2, 3], [0, 1, 2])})
    >>> new_g = dgl.metapath_reachable_graph(g, ['AB', 'BA'])
    >>> new_g.edges(order='eid')
    (tensor([0, 1, 2]), tensor([0, 1, 2]))
    """
    adj = 1
    for etype in metapath:
        adj = adj * g.adj_external(
            etype=etype, scipy_fmt="csr", transpose=False
        )

    adj = (adj != 0).tocsr()
    srctype = g.to_canonical_etype(metapath[0])[0]
    dsttype = g.to_canonical_etype(metapath[-1])[2]
    new_g = convert.heterograph(
        {(srctype, "_E", dsttype): adj.nonzero()},
        {srctype: adj.shape[0], dsttype: adj.shape[1]},
        idtype=g.idtype,
        device=g.device,
    )

    # copy srcnode features
    new_g.nodes[srctype].data.update(g.nodes[srctype].data)
    # copy dstnode features
    if srctype != dsttype:
        new_g.nodes[dsttype].data.update(g.nodes[dsttype].data)

    return new_g


def add_nodes(g, num, data=None, ntype=None):
    r"""Add the given number of nodes to the graph and return a new graph.

    The new nodes will have IDs starting from ``g.num_nodes(ntype)``.

    Parameters
    ----------
    num : int
        The number of nodes to add.
    data : dict[str, Tensor], optional
        Feature data of the added nodes. The keys are feature names
        while the values are feature data.
    ntype : str, optional
        The node type name. Can be omitted if there is
        only one type of nodes in the graph.

    Return
    ------
    DGLGraph
        The graph with newly added nodes.

    Notes
    -----
    * For features in :attr:`g` but not in :attr:`data`,
      DGL assigns zero features for the newly added nodes.
    * For feature in :attr:`data` but not in :attr:`g`, DGL assigns zero features
      for the existing nodes in the graph.
    * This function discards the batch information. Please use
      :func:`dgl.DGLGraph.set_batch_num_nodes`
      and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
      to maintain the information.

    Examples
    --------

    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch

    **Homogeneous Graphs**

    >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
    >>> g.num_nodes()
    3
    >>> g = dgl.add_nodes(g, 2)
    >>> g.num_nodes()
    5

    If the graph has some node features and new nodes are added without
    features, their features will be filled with zeros.

    >>> g.ndata['h'] = torch.ones(5, 1)
    >>> g = dgl.add_nodes(g, 1)
    >>> g.ndata['h']
    tensor([[1.], [1.], [1.], [1.], [1.], [0.]])

    Assign features for the new nodes.

    >>> g = dgl.add_nodes(g, 1, {'h': torch.ones(1, 1), 'w': torch.ones(1, 1)})
    >>> g.ndata['h']
    tensor([[1.], [1.], [1.], [1.], [1.], [0.], [1.]])

    Since :attr:`data` contains new feature fields, the features for existing nodes
    will be filled with zeros.

    >>> g.ndata['w']
    tensor([[0.], [0.], [0.], [0.], [0.], [0.], [1.]])

    **Heterogeneous Graphs**

    >>> g = dgl.heterograph({
    ...     ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
    ...                                 torch.tensor([0, 0, 1, 1])),
    ...     ('developer', 'develops', 'game'): (torch.tensor([0, 1]),
    ...                                         torch.tensor([0, 1]))
    ...     })
    >>> g.num_nodes('user')
    3
    >>> g = dgl.add_nodes(g, 2, ntype='user')
    >>> g.num_nodes('user')
    5

    See Also
    --------
    remove_nodes
    add_edges
    remove_edges
    """
    g = g.clone()
    g.add_nodes(num, data=data, ntype=ntype)
    return g


def add_edges(g, u, v, data=None, etype=None):
    r"""Add the edges to the graph and return a new graph.

    The i-th new edge will be from ``u[i]`` to ``v[i]``.  The IDs of the new
    edges will start from ``g.num_edges(etype)``.

    Parameters
    ----------
    u : int, Tensor or iterable[int]
        Source node IDs, ``u[i]`` gives the source node for the i-th new edge.
    v : int, Tensor or iterable[int]
        Destination node IDs, ``v[i]`` gives the destination node for the i-th new edge.
    data : dict[str, Tensor], optional
        Feature data of the added edges. The keys are feature names
        while the values are feature data.
    etype : str or (str, str, str), optional
        The type names of the edges. The allowed type name formats are:

        * ``(str, str, str)`` for source node type, edge type and destination node type.
        * or one ``str`` edge type name if the name can uniquely identify a
          triplet format in the graph.

        Can be omitted if the graph has only one type of edges.

    Return
    ------
    DGLGraph
        The graph with newly added edges.

    Notes
    -----
    * If the end nodes of the given edges do not exist in :attr:`g`,
      :func:`dgl.add_nodes` is invoked to add those nodes.
      The node features of the new nodes will be filled with zeros.
    * For features in :attr:`g` but not in :attr:`data`,
      DGL assigns zero features for the newly added nodes.
    * For feature in :attr:`data` but not in :attr:`g`, DGL assigns zero features
      for the existing nodes in the graph.
    * This function discards the batch information. Please use
      :func:`dgl.DGLGraph.set_batch_num_nodes`
      and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
      to maintain the information.

    Examples
    --------
    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch

    **Homogeneous Graphs**

    >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
    >>> g.num_edges()
    2
    >>> g = dgl.add_edges(g, torch.tensor([1, 3]), torch.tensor([0, 1]))
    >>> g.num_edges()
    4

    Since ``u`` or ``v`` contains a non-existing node ID, the nodes are
    added implicitly.

    >>> g.num_nodes()
    4

    If the graph has some edge features and new edges are added without
    features, their features will be filled with zeros.

    >>> g.edata['h'] = torch.ones(4, 1)
    >>> g = dgl.add_edges(g, torch.tensor([1]), torch.tensor([1]))
    >>> g.edata['h']
    tensor([[1.], [1.], [1.], [1.], [0.]])

    You can also assign features for the new edges in adding new edges.

    >>> g = dgl.add_edges(g, torch.tensor([0, 0]), torch.tensor([2, 2]),
    ...                   {'h': torch.tensor([[1.], [2.]]), 'w': torch.ones(2, 1)})
    >>> g.edata['h']
    tensor([[1.], [1.], [1.], [1.], [0.], [1.], [2.]])

    Since :attr:`data` contains new feature fields, the features for old edges
    will be filled with zeros.

    >>> g.edata['w']
    tensor([[0.], [0.], [0.], [0.], [0.], [1.], [1.]])

    **Heterogeneous Graphs**

    >>> g = dgl.heterograph({
    ...     ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
    ...                                 torch.tensor([0, 0, 1, 1])),
    ...     ('developer', 'develops', 'game'): (torch.tensor([0, 1]),
    ...                                         torch.tensor([0, 1]))
    ...     })
    >>> g.num_edges('plays')
    4
    >>> g = dgl.add_edges(g, torch.tensor([3]), torch.tensor([3]), etype='plays')
    >>> g.num_edges('plays')
    5

    See Also
    --------
    add_nodes
    remove_nodes
    remove_edges
    """
    g = g.clone()
    g.add_edges(u, v, data=data, etype=etype)
    return g


def remove_edges(g, eids, etype=None, store_ids=False):
    r"""Remove the specified edges and return a new graph.

    Also delete the features of the edges. The edges must exist in the graph.
    The resulting graph has the same number of the nodes as the input one,
    even if some nodes become isolated after the the edge removal.

    Parameters
    ----------
    eids : int, Tensor, iterable[int]
        The IDs of the edges to remove.
    etype : str or (str, str, str), optional
        The type names of the edges. The allowed type name formats are:

        * ``(str, str, str)`` for source node type, edge type and destination node type.
        * or one ``str`` edge type name if the name can uniquely identify a
          triplet format in the graph.

        Can be omitted if the graph has only one type of edges.
    store_ids : bool, optional
        If True, it will store the raw IDs of the extracted nodes and edges in the ``ndata``
        and ``edata`` of the resulting graph under name ``dgl.NID`` and ``dgl.EID``,
        respectively.

    Return
    ------
    DGLGraph
        The graph with edges deleted.

    Notes
    -----
    This function preserves the batch information.

    Examples
    --------
    >>> import dgl
    >>> import torch

    **Homogeneous Graphs**

    >>> g = dgl.graph((torch.tensor([0, 0, 2]), torch.tensor([0, 1, 2])))
    >>> g.edata['he'] = torch.arange(3).float().reshape(-1, 1)
    >>> g = dgl.remove_edges(g, torch.tensor([0, 1]))
    >>> g
    Graph(num_nodes=3, num_edges=1,
        ndata_schemes={}
        edata_schemes={'he': Scheme(shape=(1,), dtype=torch.float32)})
    >>> g.edges('all')
    (tensor([2]), tensor([2]), tensor([0]))
    >>> g.edata['he']
    tensor([[2.]])

    **Heterogeneous Graphs**

    >>> g = dgl.heterograph({
    ...     ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
    ...                                 torch.tensor([0, 0, 1, 1])),
    ...     ('developer', 'develops', 'game'): (torch.tensor([0, 1]),
    ...                                         torch.tensor([0, 1]))
    ...     })
    >>> g = dgl.remove_edges(g, torch.tensor([0, 1]), 'plays')
    >>> g.edges('all', etype='plays')
    (tensor([1, 2]), tensor([1, 1]), tensor([0, 1]))

    See Also
    --------
    add_nodes
    add_edges
    remove_nodes
    """
    g = g.clone()
    g.remove_edges(eids, etype=etype, store_ids=store_ids)
    return g


def remove_nodes(g, nids, ntype=None, store_ids=False):
    r"""Remove the specified nodes and return a new graph.

    Also delete the features. Edges that connect from/to the nodes will be
    removed as well. After the removal, DGL re-labels the remaining nodes and edges
    with IDs from 0.

    Parameters
    ----------
    nids : int, Tensor, iterable[int]
        The nodes to be removed.
    ntype : str, optional
        The type of the nodes to remove. Can be omitted if there is
        only one node type in the graph.
    store_ids : bool, optional
        If True, it will store the raw IDs of the extracted nodes and edges in the ``ndata``
        and ``edata`` of the resulting graph under name ``dgl.NID`` and ``dgl.EID``,
        respectively.

    Return
    ------
    DGLGraph
        The graph with nodes deleted.

    Notes
    -----

    This function discards the batch information. Please use
    :func:`dgl.DGLGraph.set_batch_num_nodes`
    and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
    to maintain the information.

    Examples
    --------

    >>> import dgl
    >>> import torch

    **Homogeneous Graphs**

    >>> g = dgl.graph((torch.tensor([0, 0, 2]), torch.tensor([0, 1, 2])))
    >>> g.ndata['hv'] = torch.arange(3).float().reshape(-1, 1)
    >>> g.edata['he'] = torch.arange(3).float().reshape(-1, 1)
    >>> g = dgl.remove_nodes(g, torch.tensor([0, 1]))
    >>> g
    Graph(num_nodes=1, num_edges=1,
        ndata_schemes={'hv': Scheme(shape=(1,), dtype=torch.float32)}
        edata_schemes={'he': Scheme(shape=(1,), dtype=torch.float32)})
    >>> g.ndata['hv']
    tensor([[2.]])
    >>> g.edata['he']
    tensor([[2.]])

    **Heterogeneous Graphs**

    >>> g = dgl.heterograph({
    ...     ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
    ...                                 torch.tensor([0, 0, 1, 1])),
    ...     ('developer', 'develops', 'game'): (torch.tensor([0, 1]),
    ...                                         torch.tensor([0, 1]))
    ...     })
    >>> g = dgl.remove_nodes(g, torch.tensor([0, 1]), ntype='game')
    >>> g.num_nodes('user')
    3
    >>> g.num_nodes('game')
    0
    >>> g.num_edges('plays')
    0

    See Also
    --------
    add_nodes
    add_edges
    remove_edges
    """
    g = g.clone()
    g.remove_nodes(nids, ntype=ntype, store_ids=store_ids)
    return g


def add_self_loop(g, edge_feat_names=None, fill_data=1.0, etype=None):
    r"""Add self-loops for each node in the graph and return a new graph.

    Parameters
    ----------
    g : DGLGraph
        The graph.
    edge_feat_names : list[str], optional
        The names of the self-loop features to apply `fill_data`. If None, it will apply `fill_data`
        to all self-loop features. Default: None.
    fill_data : int, float or str, optional
        The value to fill the self-loop features. Default: 1.

        * If ``fill_data`` is ``int`` or ``float``, self-loop features will be directly given by
          ``fill_data``.
        * if ``fill_data`` is ``str``, self-loop features will be generated by aggregating the
          features of the incoming edges of the corresponding nodes. The supported aggregation are:
          ``'mean'``, ``'sum'``, ``'max'``, ``'min'``.
    etype : str or (str, str, str), optional
        The type names of the edges. The allowed type name formats are:

        * ``(str, str, str)`` for source node type, edge type and destination node type.
        * or one ``str`` edge type name if the name can uniquely identify a
          triplet format in the graph.

        Can be omitted if the graph has only one type of edges.

    Return
    ------
    DGLGraph
        The graph with self-loops.

    Notes
    -----
    * The function only supports homogeneous graphs or heterogeneous graphs but
      the relation graph specified by the :attr:`etype` argument is homogeneous.
    * The function adds self-loops regardless of whether they already exist or not.
      If one wishes to have exactly one self-loop for every node,
      call :func:`remove_self_loop` before invoking :func:`add_self_loop`.
    * This function discards the batch information. Please use
      :func:`dgl.DGLGraph.set_batch_num_nodes`
      and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
      to maintain the information.

    Examples
    --------
    >>> import dgl
    >>> import torch

    **Homogeneous Graphs**

    >>> g = dgl.graph((torch.tensor([0, 0, 2]), torch.tensor([2, 1, 0])))
    >>> g.ndata['hv'] = torch.arange(3).float().reshape(-1, 1)
    >>> g.edata['he'] = torch.arange(3).float().reshape(-1, 1)
    >>> g = dgl.add_self_loop(g, fill_data='sum')
    >>> g
    Graph(num_nodes=3, num_edges=6,
        ndata_schemes={'hv': Scheme(shape=(1,), dtype=torch.float32)}
        edata_schemes={'he': Scheme(shape=(1,), dtype=torch.float32)})
    >>> g.edata['he']
    tensor([[0.],
            [1.],
            [2.],
            [2.],
            [1.],
            [0.]])

    **Heterogeneous Graphs**

    >>> g = dgl.heterograph({
    ...     ('user', 'follows', 'user'): (torch.tensor([1, 2]),
    ...                                   torch.tensor([0, 1])),
    ...     ('user', 'plays', 'game'): (torch.tensor([0, 1]),
    ...                                 torch.tensor([0, 1]))})
    >>> g = dgl.add_self_loop(g, etype='follows')
    >>> g
    Graph(num_nodes={'user': 3, 'game': 2},
          num_edges={('user', 'plays', 'game'): 2, ('user', 'follows', 'user'): 5},
          metagraph=[('user', 'user'), ('user', 'game')])
    """
    etype = g.to_canonical_etype(etype)
    data = {}
    reduce_funcs = {
        "sum": function.sum,
        "mean": function.mean,
        "max": function.max,
        "min": function.min,
    }

    if edge_feat_names is None:
        edge_feat_names = g.edges[etype].data.keys()

    if etype[0] != etype[2]:
        raise DGLError(
            "add_self_loop does not support unidirectional bipartite graphs: {}."
            "Please make sure the types of head node and tail node are identical."
            "".format(etype)
        )

    for feat_name in edge_feat_names:
        if isinstance(fill_data, (int, float)):
            dtype = g.edges[etype].data[feat_name].dtype
            dshape = g.edges[etype].data[feat_name].shape
            tmp_fill_data = F.copy_to(
                F.astype(F.tensor([fill_data]), dtype), g.device
            )
            if len(dshape) > 1:
                data[feat_name] = (
                    F.zeros(
                        (g.num_nodes(etype[0]), *dshape[1:]), dtype, g.device
                    )
                    + tmp_fill_data
                )
            else:
                data[feat_name] = (
                    F.zeros((g.num_nodes(etype[0]),), dtype, g.device)
                    + tmp_fill_data
                )

        elif isinstance(fill_data, str):
            if fill_data not in reduce_funcs.keys():
                raise DGLError("Unsupported aggregation: {}".format(fill_data))
            reducer = reduce_funcs[fill_data]
            with g.local_scope():
                g.update_all(
                    function.copy_e(feat_name, "h"),
                    reducer("h", "h"),
                    etype=etype,
                )
                data[feat_name] = g.nodes[etype[0]].data["h"]

    nodes = g.nodes(etype[0])
    if len(data):
        new_g = add_edges(g, nodes, nodes, data=data, etype=etype)
    else:
        new_g = add_edges(g, nodes, nodes, etype=etype)
    return new_g


DGLGraph.add_self_loop = utils.alias_func(add_self_loop)


def remove_self_loop(g, etype=None):
    r"""Remove self-loops for each node in the graph and return a new graph.

    Parameters
    ----------
    g : DGLGraph
        The graph.
    etype : str or (str, str, str), optional
        The type names of the edges. The allowed type name formats are:

        * ``(str, str, str)`` for source node type, edge type and destination node type.
        * or one ``str`` edge type name if the name can uniquely identify a
          triplet format in the graph.

        Can be omitted if the graph has only one type of edges.

    Notes
    -----
    If a node has multiple self-loops, remove them all. Do nothing for nodes without
    self-loops.

    This function preserves the batch information.

    Examples
    ---------

    >>> import dgl
    >>> import torch

    **Homogeneous Graphs**

    >>> g = dgl.graph((torch.tensor([0, 0, 0, 1]), torch.tensor([1, 0, 0, 2])))
    >>> g.edata['he'] = torch.arange(4).float().reshape(-1, 1)
    >>> g = dgl.remove_self_loop(g)
    >>> g
    Graph(num_nodes=3, num_edges=2,
        edata_schemes={'he': Scheme(shape=(2,), dtype=torch.float32)})
    >>> g.edata['he']
    tensor([[0.],[3.]])

    **Heterogeneous Graphs**

    >>> g = dgl.heterograph({
    ...     ('user', 'follows', 'user'): (torch.tensor([0, 1, 1, 1, 2]),
    ...                                   torch.tensor([0, 0, 1, 1, 1])),
    ...     ('user', 'plays', 'game'): (torch.tensor([0, 1]),
    ...                                 torch.tensor([0, 1]))
    ...     })
    >>> g = dgl.remove_self_loop(g, etype='follows')
    >>> g.num_nodes('user')
    3
    >>> g.num_nodes('game')
    2
    >>> g.num_edges('follows')
    2
    >>> g.num_edges('plays')
    2

    See Also
    --------
    add_self_loop
    """
    etype = g.to_canonical_etype(etype)
    if etype[0] != etype[2]:
        raise DGLError(
            "remove_self_loop does not support unidirectional bipartite graphs: {}."
            "Please make sure the types of head node and tail node are identical."
            "".format(etype)
        )
    u, v = g.edges(form="uv", order="eid", etype=etype)
    self_loop_eids = F.tensor(F.nonzero_1d(u == v), dtype=F.dtype(u))
    new_g = remove_edges(g, self_loop_eids, etype=etype)
    return new_g


DGLGraph.remove_self_loop = utils.alias_func(remove_self_loop)


def compact_graphs(
    graphs, always_preserve=None, copy_ndata=True, copy_edata=True
):
    """Given a list of graphs with the same set of nodes, find and eliminate the common
    isolated nodes across all graphs.

    This function requires the graphs to have the same set of nodes (i.e. the node types
    must be the same, and the number of nodes of each node type must be the same).  The
    metagraph does not have to be the same.

    It finds all the nodes that have zero in-degree and zero out-degree in all the given
    graphs, and eliminates them from all the graphs.

    Useful for graph sampling where you have a giant graph but you only wish to perform
    message passing on a smaller graph with a (tiny) subset of nodes.

    Parameters
    ----------
    graphs : DGLGraph or list[DGLGraph]
        The graph, or list of graphs.

        All graphs must be on the same devices.

        All graphs must have the same set of nodes.
    always_preserve : Tensor or dict[str, Tensor], optional
        If a dict of node types and node ID tensors is given, the nodes of given
        node types would not be removed, regardless of whether they are isolated.

        If a Tensor is given, DGL assumes that all the graphs have one (same) node type.
    copy_ndata: bool, optional
        If True, the node features of the returned graphs are copied from the
        original graphs.

        If False, the returned graphs will not have any node features.

        (Default: True)
    copy_edata: bool, optional
        If True, the edge features of the reversed graph are copied from the
        original graph.

        If False, the reversed graph will not have any edge features.

        (Default: True)

    Returns
    -------
    DGLGraph or list[DGLGraph]
        The compacted graph or list of compacted graphs.

        Each returned graph would have a feature ``dgl.NID`` containing the mapping
        of node IDs for each type from the compacted graph(s) to the original graph(s).
        Note that the mapping is the same for all the compacted graphs.

        All the returned graphs are on CPU.

    Notes
    -----
    This function currently requires that the same node type of all graphs should have
    the same node type ID, i.e. the node types are *ordered* the same.

    If :attr:`copy_edata` is True, the resulting graph will share the edge feature
    tensors with the input graph. Hence, users should try to avoid in-place operations
    which will be visible to both graphs.

    This function discards the batch information. Please use
    :func:`dgl.DGLGraph.set_batch_num_nodes`
    and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
    to maintain the information.

    Examples
    --------
    The following code constructs a bipartite graph with 20 users and 10 games, but
    only user #1 and #3, as well as game #3 and #5, have connections:

    >>> g = dgl.heterograph({('user', 'plays', 'game'): ([1, 3], [3, 5])},
    >>>                      {'user': 20, 'game': 10})

    The following would compact the graph above to another bipartite graph with only
    two users and two games.

    >>> new_g = dgl.compact_graphs(g)
    >>> new_g.ndata[dgl.NID]
    {'user': tensor([1, 3]), 'game': tensor([3, 5])}

    The mapping tells us that only user #1 and #3 as well as game #3 and #5 are kept.
    Furthermore, the first user and second user in the compacted graph maps to
    user #1 and #3 in the original graph.  Games are similar.

    One can verify that the edge connections are kept the same in the compacted graph.

    >>> new_g.edges(form='all', order='eid', etype='plays')
    (tensor([0, 1]), tensor([0, 1]), tensor([0, 1]))

    When compacting multiple graphs, nodes that do not have any connections in any
    of the given graphs are removed.  So if you compact ``g`` and the following ``g2``
    graphs together:

    >>> g2 = dgl.heterograph({('user', 'plays', 'game'): ([1, 6], [6, 8])},
    >>>                      {'user': 20, 'game': 10})
    >>> new_g, new_g2 = dgl.compact_graphs([g, g2])
    >>> new_g.ndata[dgl.NID]
    {'user': tensor([1, 3, 6]), 'game': tensor([3, 5, 6, 8])}

    Then one can see that user #1 from both graphs, users #3 from the first graph, as
    well as user #6 from the second graph, are kept.  Games are similar.

    Similarly, one can also verify the connections:

    >>> new_g.edges(form='all', order='eid', etype='plays')
    (tensor([0, 1]), tensor([0, 1]), tensor([0, 1]))
    >>> new_g2.edges(form='all', order='eid', etype='plays')
    (tensor([0, 2]), tensor([2, 3]), tensor([0, 1]))
    """
    return_single = False
    if not isinstance(graphs, Iterable):
        graphs = [graphs]
        return_single = True
    if len(graphs) == 0:
        return []
    if graphs[0].is_block:
        raise DGLError("Compacting a block graph is not allowed.")

    # Ensure the node types are ordered the same.
    # TODO(BarclayII): we ideally need to remove this constraint.
    ntypes = graphs[0].ntypes
    idtype = graphs[0].idtype
    device = graphs[0].device
    for g in graphs:
        assert ntypes == g.ntypes, (
            "All graphs should have the same node types in the same order, got %s and %s"
            % ntypes,
            g.ntypes,
        )
        assert (
            idtype == g.idtype
        ), "Expect graph data type to be {}, but got {}".format(
            idtype, g.idtype
        )
        assert device == g.device, (
            "All graphs must be on the same devices."
            "Expect graph device to be {}, but got {}".format(device, g.device)
        )

    # Process the dictionary or tensor of "always preserve" nodes
    if always_preserve is None:
        always_preserve = {}
    elif not isinstance(always_preserve, Mapping):
        if len(ntypes) > 1:
            raise ValueError(
                "Node type must be given if multiple node types exist."
            )
        always_preserve = {ntypes[0]: always_preserve}

    always_preserve = utils.prepare_tensor_dict(
        graphs[0], always_preserve, "always_preserve"
    )
    always_preserve_nd = []
    for ntype in ntypes:
        nodes = always_preserve.get(ntype, None)
        if nodes is None:
            nodes = F.copy_to(F.tensor([], idtype), device)
        always_preserve_nd.append(F.to_dgl_nd(nodes))

    # Compact and construct heterographs
    new_graph_indexes, induced_nodes = _CAPI_DGLCompactGraphs(
        [g._graph for g in graphs], always_preserve_nd
    )
    induced_nodes = [F.from_dgl_nd(nodes) for nodes in induced_nodes]

    new_graphs = [
        DGLGraph(new_graph_index, graph.ntypes, graph.etypes)
        for new_graph_index, graph in zip(new_graph_indexes, graphs)
    ]

    if copy_ndata:
        for g, new_g in zip(graphs, new_graphs):
            node_frames = utils.extract_node_subframes(g, induced_nodes)
            utils.set_new_frames(new_g, node_frames=node_frames)
    if copy_edata:
        for g, new_g in zip(graphs, new_graphs):
            edge_frames = utils.extract_edge_subframes(g, None)
            utils.set_new_frames(new_g, edge_frames=edge_frames)

    if return_single:
        new_graphs = new_graphs[0]

    return new_graphs


def _coalesce_edge_frame(g, edge_maps, counts, aggregator):
    r"""Coalesce edge features of duplicate edges via given aggregator in g.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    edge_maps : List[Tensor]
        The edge mapping corresponding to each edge type in g.
    counts : List[Tensor]
        The number of duplicated edges from the original graph for each edge type.
    aggregator : str
        Indicates how to coalesce edge features, could be ``arbitrary``, ``sum``
        or ``mean``.

    Returns
    -------
    List[Frame]
        The frames corresponding to each edge type.
    """
    if aggregator == "arbitrary":
        eids = []
        for i in range(len(g.canonical_etypes)):
            feat_idx = F.asnumpy(edge_maps[i])
            _, indices = np.unique(feat_idx, return_index=True)
            eids.append(F.zerocopy_from_numpy(indices))

        edge_frames = utils.extract_edge_subframes(g, eids)
    elif aggregator in ["sum", "mean"]:
        edge_frames = []
        for i in range(len(g.canonical_etypes)):
            feat_idx = edge_maps[i]
            _, indices = np.unique(F.asnumpy(feat_idx), return_index=True)
            _num_rows = len(indices)
            _data = {}
            for key, col in g._edge_frames[i]._columns.items():
                data = col.data
                new_data = F.scatter_add(data, feat_idx, _num_rows)
                if aggregator == "mean":
                    norm = F.astype(counts[i], F.dtype(data))
                    norm = F.reshape(
                        norm, (F.shape(norm)[0],) + (1,) * (F.ndim(data) - 1)
                    )
                    new_data /= norm
                _data[key] = new_data

            newf = Frame(data=_data, num_rows=_num_rows)
            edge_frames.append(newf)
    else:
        raise DGLError(
            "Aggregator {} not regonized, cannot coalesce edge feature in the "
            "specified way".format(aggregator)
        )
    return edge_frames


def to_simple(
    g,
    return_counts="count",
    writeback_mapping=False,
    copy_ndata=True,
    copy_edata=False,
    aggregator="arbitrary",
):
    r"""Convert a graph to a simple graph without parallel edges and return.

    For a heterogeneous graph with multiple edge types, DGL treats edges with the same
    edge type and endpoints as parallel edges and removes them.
    Optionally, one can get the the number of parallel edges by specifying the
    :attr:`return_counts` argument. To get the a mapping from the edge IDs in the
    input graph to the edge IDs in the resulting graph, set :attr:`writeback_mapping`
    to true.

    Parameters
    ----------
    g : DGLGraph
        The input graph.  Must be on CPU.
    return_counts : str, optional
        If given, the count of each edge in the original graph
        will be stored as edge features under the name
        ``return_counts``.  The old features with the same name will be replaced.

        (Default: "count")
    writeback_mapping: bool, optional
        If True, return an extra write-back mapping for each edge
        type. The write-back mapping is a tensor recording
        the mapping from the edge IDs in the input graph to
        the edge IDs in the result graph. If the graph is
        heterogeneous, DGL returns a dictionary of edge types and such
        tensors.

        If False, only the simple graph is returned.

        (Default: False)
    copy_ndata: bool, optional
        If True, the node features of the simple graph are copied
        from the original graph.

        If False, the simple graph will not have any node features.

        (Default: True)
    copy_edata: bool, optional
        If True, the edge features of the simple graph are copied
        from the original graph. If there exists duplicate edges between
        two nodes (u, v), the feature of the edge is the aggregation
        of edge feature of duplicate edges.

        If False, the simple graph will not have any edge features.

        (Default: False)
    aggregator: str, optional
        Indicate how to coalesce edge feature of duplicate edges.
        If ``arbitrary``, select one of the duplicate edges' feature.
        If ``sum``, compute the summation of duplicate edges' feature.
        If ``mean``, compute the average of duplicate edges' feature.

        (Default: ``arbitrary``)

    Returns
    -------
    DGLGraph
        The graph.
    tensor or dict of tensor
        The writeback mapping. Only when ``writeback_mapping`` is True.

    Notes
    -----
    If :attr:`copy_ndata` is True, the resulting graph will share the node feature
    tensors with the input graph. Hence, users should try to avoid in-place operations
    which will be visible to both graphs.

    This function discards the batch information. Please use
    :func:`dgl.DGLGraph.set_batch_num_nodes`
    and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
    to maintain the information.

    Examples
    --------
    **Homogeneous Graphs**

    Create a graph for demonstrating to_simple API.
    In the original graph, there are multiple edges between 1 and 2.

    >>> import dgl
    >>> import torch as th
    >>> g = dgl.graph((th.tensor([0, 1, 2, 1]), th.tensor([1, 2, 0, 2])))
    >>> g.ndata['h'] = th.tensor([[0.], [1.], [2.]])
    >>> g.edata['h'] = th.tensor([[3.], [4.], [5.], [6.]])

    Convert the graph to a simple graph. The return counts is
    stored in the edge feature 'cnt' and the writeback mapping
    is returned in a tensor.

    >>> sg, wm = dgl.to_simple(g, return_counts='cnt', writeback_mapping=True)
    >>> sg.ndata['h']
    tensor([[0.],
            [1.],
            [2.]])
    >>> u, v, eid = sg.edges(form='all')
    >>> u
    tensor([0, 1, 2])
    >>> v
    tensor([1, 2, 0])
    >>> eid
    tensor([0, 1, 2])
    >>> sg.edata['cnt']
    tensor([1, 2, 1])
    >>> wm
    tensor([0, 1, 2, 1])
    >>> 'h' in g.edata
    False

    **Heterogeneous Graphs**

    >>> g = dgl.heterograph({
    ...     ('user', 'wins', 'user'): (th.tensor([0, 2, 0, 2, 2]), th.tensor([1, 1, 2, 1, 0])),
    ...     ('user', 'plays', 'game'): (th.tensor([1, 2, 1]), th.tensor([2, 1, 1]))
    ... })
    >>> g.nodes['game'].data['hv'] = th.ones(3, 1)
    >>> g.edges['plays'].data['he'] = th.zeros(3, 1)

    The return counts is stored in the default edge feature 'count' for each edge type.

    >>> sg, wm = dgl.to_simple(g, copy_ndata=False, writeback_mapping=True)
    >>> sg
    Graph(num_nodes={'game': 3, 'user': 3},
          num_edges={('user', 'wins', 'user'): 4, ('game', 'plays', 'user'): 3},
          metagraph=[('user', 'user'), ('game', 'user')])
    >>> sg.edges(etype='wins')
    (tensor([0, 2, 0, 2]), tensor([1, 1, 2, 0]))
    >>> wm[('user', 'wins', 'user')]
    tensor([0, 1, 2, 1, 3])
    >>> sg.edges(etype='plays')
    (tensor([2, 1, 1]), tensor([1, 2, 1]))
    >>> wm[('user', 'plays', 'game')]
    tensor([0, 1, 2])
    >>> 'hv' in sg.nodes['game'].data
    False
    >>> 'he' in sg.edges['plays'].data
    False
    >>> sg.edata['count']
    {('user', 'wins', 'user'): tensor([1, 2, 1, 1])
     ('user', 'plays', 'game'): tensor([1, 1, 1])}
    """
    assert g.device == F.cpu(), "the graph must be on CPU"
    if g.is_block:
        raise DGLError("Cannot convert a block graph to a simple graph.")
    simple_graph_index, counts, edge_maps = _CAPI_DGLToSimpleHetero(g._graph)
    simple_graph = DGLGraph(simple_graph_index, g.ntypes, g.etypes)
    counts = [F.from_dgl_nd(count) for count in counts]
    edge_maps = [F.from_dgl_nd(edge_map) for edge_map in edge_maps]

    if copy_ndata:
        node_frames = utils.extract_node_subframes(g, None)
        utils.set_new_frames(simple_graph, node_frames=node_frames)
    if copy_edata:
        new_edge_frames = _coalesce_edge_frame(g, edge_maps, counts, aggregator)
        utils.set_new_frames(simple_graph, edge_frames=new_edge_frames)

    if return_counts is not None:
        for count, canonical_etype in zip(counts, g.canonical_etypes):
            simple_graph.edges[canonical_etype].data[return_counts] = count

    if writeback_mapping:
        # single edge type
        if len(edge_maps) == 1:
            return simple_graph, edge_maps[0]
        # multiple edge type
        else:
            wb_map = {}
            for edge_map, canonical_etype in zip(edge_maps, g.canonical_etypes):
                wb_map[canonical_etype] = edge_map
            return simple_graph, wb_map

    return simple_graph


DGLGraph.to_simple = utils.alias_func(to_simple)


def _unitgraph_less_than_int32(g):
    """Check if a graph with only one edge type has more than 2 ** 31 - 1
    nodes or edges.
    """
    num_edges = g.num_edges()
    num_nodes = max(g.num_nodes(g.ntypes[0]), g.num_nodes(g.ntypes[-1]))
    return max(num_nodes, num_edges) <= (1 << 31) - 1


def adj_product_graph(A, B, weight_name, etype="_E"):
    r"""Create a weighted graph whose adjacency matrix is the product of
    the adjacency matrices of the given two graphs.

    Namely, given two weighted graphs :attr:`A` and :attr:`B`, whose rows
    represent source nodes and columns represent destination nodes, this function
    returns a new graph whose weighted adjacency matrix is
    :math:`\mathrm{adj}(A) \times \mathrm{adj}(B)`.

    The two graphs must be simple graphs, and must have only one edge type.
    Moreover, the number of nodes of the destination node type of :attr:`A` must
    be the same as the number of nodes of the source node type of :attr:`B`.

    The source node type of the returned graph will be the same as the source
    node type of graph :attr:`A`.  The destination node type of the returned
    graph will be the same as the destination node type of graph :attr:`B`.
    If the two node types are the same, the returned graph will be homogeneous.
    Otherwise, it will be a bipartite graph.

    Unlike ``scipy``, if an edge in the result graph has zero weight, it will
    not be removed from the graph.

    Notes
    -----
    This function works on both CPU and GPU.  For GPU, the number of nodes and
    edges must be less than the maximum of ``int32`` (i.e. ``2 ** 31 - 1``) due
    to restriction of cuSPARSE.

    The edge weights returned by this function is differentiable w.r.t. the
    input edge weights.

    If the graph format is restricted, both graphs must have CSR available.

    Parameters
    ----------
    A : DGLGraph
        The graph as left operand.
    B : DGLGraph
        The graph as right operand.
    weight_name : str
        The feature name of edge weight of both graphs.

        The corresponding edge feature must be scalar.
    etype : str, optional
        The edge type of the returned graph.

    Returns
    -------
    DGLGraph
        The new graph.  The edge weight of the returned graph will have the
        same feature name as :attr:`weight_name`.

    Examples
    --------
    The following shows weighted adjacency matrix multiplication between two
    bipartite graphs.  You can also perform this between two homogeneous
    graphs, or one homogeneous graph and one bipartite graph, as long as the
    numbers of nodes of the same type match.

    >>> A = dgl.heterograph({
    ...     ('A', 'AB', 'B'): ([2, 2, 0, 2, 0, 1], [2, 1, 0, 0, 2, 2])},
    ...     num_nodes_dict={'A': 3, 'B': 4})
    >>> B = dgl.heterograph({
    ...     ('B', 'BA', 'A'): ([0, 3, 2, 1, 3, 3], [1, 2, 0, 2, 1, 0])},
    ...     num_nodes_dict={'A': 3, 'B': 4})

    If your graph is a multigraph, you will need to call :func:`dgl.to_simple`
    to convert it into a simple graph first.

    >>> A = dgl.to_simple(A)
    >>> B = dgl.to_simple(B)

    Initialize learnable edge weights.

    >>> A.edata['w'] = torch.randn(6).requires_grad_()
    >>> B.edata['w'] = torch.randn(6).requires_grad_()

    Take the product.

    >>> C = dgl.adj_product_graph(A, B, 'w')
    >>> C.edges()
    (tensor([0, 0, 1, 2, 2, 2]), tensor([0, 1, 0, 0, 2, 1]))

    >>> C.edata['w']
    tensor([0.6906, 0.2002, 0.0591, 0.3672, 0.1066, 0.1328],
           grad_fn=<CSRMMBackward>)

    Note that this function is differentiable:

    >>> C.edata['w'].sum().backward()
    >>> A.edata['w'].grad
    tensor([0.7153, 0.2775, 0.7141, 0.7141, 0.7153, 0.7153])

    >>> B.edata['w'].grad
    tensor([0.4664, 0.0000, 1.5614, 0.3840, 0.0000, 0.0000])

    If the source node type of the left operand is the same as the destination
    node type of the right operand, this function returns a homogeneous graph:

    >>> C.ntypes
    ['A']

    Otherwise, it returns a bipartite graph instead:

    >>> A = dgl.heterograph({
    ...     ('A', 'AB', 'B'): ([2, 2, 0, 2, 0, 1], [2, 1, 0, 0, 2, 2])},
    ...     num_nodes_dict={'A': 3, 'B': 4})
    >>> B = dgl.heterograph({
    ...     ('B', 'BC', 'C'): ([0, 3, 2, 1, 3, 3], [1, 2, 0, 2, 1, 0])},
    ...     num_nodes_dict={'C': 3, 'B': 4})
    >>> A.edata['w'] = torch.randn(6).requires_grad_()
    >>> B.edata['w'] = torch.randn(6).requires_grad_()
    >>> C = dgl.adj_product_graph(A, B, 'w')
    >>> C.ntypes
    ['A', 'C']
    """
    srctype, _, _ = A.canonical_etypes[0]
    _, _, dsttype = B.canonical_etypes[0]
    num_vtypes = 1 if srctype == dsttype else 2
    ntypes = [srctype] if num_vtypes == 1 else [srctype, dsttype]

    if A.device != F.cpu():
        if not (
            _unitgraph_less_than_int32(A) and _unitgraph_less_than_int32(B)
        ):
            raise ValueError(
                "For GPU graphs the number of nodes and edges must be less than 2 ** 31 - 1."
            )

    C_gidx, C_weights = F.csrmm(
        A._graph,
        A.edata[weight_name],
        B._graph,
        B.edata[weight_name],
        num_vtypes,
    )
    num_nodes_dict = {
        srctype: A.num_nodes(srctype),
        dsttype: B.num_nodes(dsttype),
    }
    C_metagraph, ntypes, etypes, _ = create_metagraph_index(
        ntypes, [(srctype, etype, dsttype)]
    )
    num_nodes_per_type = [num_nodes_dict[ntype] for ntype in ntypes]
    C_gidx = create_heterograph_from_relations(
        C_metagraph, [C_gidx], utils.toindex(num_nodes_per_type)
    )

    C = DGLGraph(C_gidx, ntypes, etypes)
    C.edata[weight_name] = C_weights
    return C


def adj_sum_graph(graphs, weight_name):
    r"""Create a weighted graph whose adjacency matrix is the sum of the
    adjacency matrices of the given graphs, whose rows represent source nodes
    and columns represent destination nodes.

    All the graphs must be simple graphs, and must have only one edge type.
    They also must have the same metagraph, i.e. have the same source node type
    and the same destination node type.  Moreover, the number of nodes for every
    graph must also be the same.

    The metagraph of the returned graph will be the same as the input graphs.

    Unlike ``scipy``, if an edge in the result graph has zero weight, it will
    not be removed from the graph.

    Notes
    -----
    This function works on both CPU and GPU.  For GPU, the number of nodes and
    edges must be less than the maximum of ``int32`` (i.e. ``2 ** 31 - 1``) due
    to restriction of cuSPARSE.

    The edge weights returned by this function is differentiable w.r.t. the
    input edge weights.

    If the graph format is restricted, both graphs must have CSR available.

    Parameters
    ----------
    graphs : list[DGLGraph]
        The list of graphs.  Must have at least one element.
    weight_name : str
        The feature name of edge weight of both graphs.

        The corresponding edge feature must be scalar.

    Returns
    -------
    DGLGraph
        The new graph.  The edge weight of the returned graph will have the
        same feature name as :attr:`weight_name`.

    Examples
    --------
    The following shows weighted adjacency matrix summation between two
    bipartite graphs.  You can also perform this between homogeneous graphs.

    >>> A = dgl.heterograph(
    ...     {('A', 'AB', 'B'): ([2, 2, 0, 2, 0, 1], [2, 1, 0, 0, 2, 2])},
    ...     num_nodes_dict={'A': 3, 'B': 4})
    >>> B = dgl.heterograph(
    ...     {('A', 'AB', 'B'): ([1, 2, 0, 2, 1, 0], [0, 3, 2, 1, 3, 3])},
    ...     num_nodes_dict={'A': 3, 'B': 4})
    >>> A.edata['w'] = torch.randn(6).requires_grad_()
    >>> B.edata['w'] = torch.randn(6).requires_grad_()

    If your graph is a multigraph, call :func:`dgl.to_simple`
    to convert it into a simple graph first.

    >>> A = dgl.to_simple(A)
    >>> B = dgl.to_simple(B)

    Initialize learnable edge weights.

    >>> A.edata['w'] = torch.randn(6).requires_grad_()
    >>> B.edata['w'] = torch.randn(6).requires_grad_()

    Take the sum.

    >>> C = dgl.adj_sum_graph([A, B], 'w')
    >>> C.edges()
    (tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2]),
     tensor([0, 2, 3, 2, 0, 3, 0, 1, 2, 3]))

    Note that this function is differentiable:

    >>> C.edata['w'].sum().backward()
    >>> A.edata['w'].grad
    tensor([1., 1., 1., 1., 1., 1.])

    >>> B.edata['w'].grad
    tensor([1., 1., 1., 1., 1., 1.])
    """
    if len(graphs) == 0:
        raise ValueError("The list of graphs must not be empty.")

    if graphs[0].device != F.cpu():
        if not all(_unitgraph_less_than_int32(A) for A in graphs):
            raise ValueError(
                "For GPU graphs the number of nodes and edges must be less than 2 ** 31 - 1."
            )
    metagraph = graphs[0]._graph.metagraph
    num_nodes = utils.toindex(
        [
            graphs[0]._graph.num_nodes(i)
            for i in range(graphs[0]._graph.number_of_ntypes())
        ]
    )
    weights = [A.edata[weight_name] for A in graphs]
    gidxs = [A._graph for A in graphs]
    C_gidx, C_weights = F.csrsum(gidxs, weights)
    C_gidx = create_heterograph_from_relations(metagraph, [C_gidx], num_nodes)

    C = DGLGraph(C_gidx, graphs[0].ntypes, graphs[0].etypes)
    C.edata[weight_name] = C_weights
    return C


def sort_csr_by_tag(g, tag, tag_offset_name="_TAG_OFFSET", tag_type="node"):
    r"""Return a new graph whose CSR matrix is sorted by the given tag.

    Sort the internal CSR matrix of the graph so that the adjacency list of each node
    , which contains the out-edges, is sorted by the tag of the out-neighbors.
    After sorting, edges sharing the same tag will be arranged in a consecutive range in
    a node's adjacency list. Following is an example:

        Consider a graph as follows::

            0 -> 0, 1, 2, 3, 4
            1 -> 0, 1, 2

        Given node tags ``[1, 1, 0, 2, 0]``, each node's adjacency list
        will be sorted as follows::

            0 -> 2, 4, 0, 1, 3
            1 -> 2, 0, 1

        Given edge tags ``[1, 1, 0, 2, 0, 1, 1, 0]`` has the same effect
        as above node tags.

    The function will also returns the starting offsets of the tag
    segments in a tensor of shape :math:`(N, max\_tag+2)`. For node ``i``,
    its out-edges connecting to node tag ``j`` is stored between
    ``tag_offsets[i][j]`` ~ ``tag_offsets[i][j+1]``. Since the offsets
    can be viewed node data, we store it in the
    ``ndata`` of the returned graph. Users can specify the
    ndata name by the :attr:`tag_pos_name` argument.

    Note that the function will not change the edge ID neither
    how the edge features are stored. The input graph must
    allow CSR format. The graph must be on CPU.

    If the input graph is heterogenous, it must have only one edge
    type and two node types (i.e., source and destination node types).
    In this case, the provided node tags are for the destination nodes,
    and the tag offsets are stored in the source node data.

    The sorted graph and the calculated tag offsets are needed by
    certain operators that consider node tags. See
    :func:`~dgl.sampling.sample_neighbors_biased` for an example.

    Parameters
    ------------
    g : DGLGraph
        The input graph.
    tag : Tensor
        Integer tensor of shape :math:`(N,)`, :math:`N` being the number
        of (destination) nodes or edges.
    tag_offset_name : str
        The name of the node feature to store tag offsets.
    tag_type : str
        Tag type which could be ``node`` or ``edge``.

    Returns
    -------
    g_sorted : DGLGraph
        A new graph whose CSR is sorted. The node/edge features of the
        input graph is shallow-copied over.

        - ``g_sorted.ndata[tag_offset_name]`` : Tensor of shape :math:`(N, max\_tag + 2)`.
        - If ``g`` is heterogeneous, get from ``g_sorted.srcdata``.

    Examples
    -----------

    ``tag_type`` is ``node``.

    >>> import dgl
    >>> import torch

    >>> g = dgl.graph(([0,0,0,0,0,1,1,1],[0,1,2,3,4,0,1,2]))
    >>> g.adj_external(scipy_fmt='csr').nonzero()
    (array([0, 0, 0, 0, 0, 1, 1, 1], dtype=int32),
     array([0, 1, 2, 3, 4, 0, 1, 2], dtype=int32))
    >>> tag = torch.IntTensor([1,1,0,2,0])
    >>> g_sorted = dgl.sort_csr_by_tag(g, tag)
    >>> g_sorted.adj_external(scipy_fmt='csr').nonzero()
    (array([0, 0, 0, 0, 0, 1, 1, 1], dtype=int32),
     array([2, 4, 0, 1, 3, 2, 0, 1], dtype=int32))
    >>> g_sorted.ndata['_TAG_OFFSET']
    tensor([[0, 2, 4, 5],
            [0, 1, 3, 3],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]])

    ``tag_type`` is ``edge``.

    >>> g = dgl.graph(([0,0,0,0,0,1,1,1],[0,1,2,3,4,0,1,2]))
    >>> g.edges()
    (tensor([0, 0, 0, 0, 0, 1, 1, 1]), tensor([0, 1, 2, 3, 4, 0, 1, 2]))
    >>> tag = torch.tensor([1, 1, 0, 2, 0, 1, 1, 0])
    >>> g_sorted = dgl.sort_csr_by_tag(g, tag, tag_type='edge')
    >>> g_sorted.adj_external(scipy_fmt='csr').nonzero()
    (array([0, 0, 0, 0, 0, 1, 1, 1], dtype=int32), array([2, 4, 0, 1, 3, 2, 0, 1], dtype=int32))
    >>> g_sorted.srcdata['_TAG_OFFSET']
    tensor([[0, 2, 4, 5],
            [0, 1, 3, 3],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]])

    See Also
    --------
    dgl.sampling.sample_neighbors_biased
    """
    if len(g.etypes) > 1:
        raise DGLError("Only support homograph and bipartite graph")
    assert tag_type in [
        "node",
        "edge",
    ], "tag_type should be either 'node' or 'edge'."
    if tag_type == "node":
        _, dst = g.edges()
        tag = F.gather_row(tag, F.tensor(dst))
    assert len(tag) == g.num_edges()
    num_tags = int(F.asnumpy(F.max(tag, 0))) + 1
    tag_arr = F.zerocopy_to_dgl_ndarray(tag)
    new_g = g.clone()
    new_g._graph, tag_pos_arr = _CAPI_DGLHeteroSortOutEdges(
        g._graph, tag_arr, num_tags
    )
    new_g.srcdata[tag_offset_name] = F.from_dgl_nd(tag_pos_arr)
    return new_g


def sort_csc_by_tag(g, tag, tag_offset_name="_TAG_OFFSET", tag_type="node"):
    r"""Return a new graph whose CSC matrix is sorted by the given tag.

    Sort the internal CSC matrix of the graph so that the adjacency list of each node
    , which contains the in-edges, is sorted by the tag of the in-neighbors.
    After sorting, edges sharing the same tag will be arranged in a consecutive range in
    a node's adjacency list. Following is an example:


        Consider a graph as follows::

            0 <- 0, 1, 2, 3, 4
            1 <- 0, 1, 2

        Given node tags ``[1, 1, 0, 2, 0]``, each node's adjacency list
        will be sorted as follows::

            0 <- 2, 4, 0, 1, 3
            1 <- 2, 0, 1

        Given edge tags ``[1, 1, 0, 2, 0, 1, 1, 0]`` has the same effect
        as above node tags.

    The function will also return the starting offsets of the tag
    segments in a tensor of shape :math:`(N, max\_tag+2)`. For a node ``i``,
    its in-edges connecting to node tag ``j`` is stored between
    ``tag_offsets[i][j]`` ~ ``tag_offsets[i][j+1]``. Since the offsets
    can be viewed node data, we store it in the
    ``ndata`` of the returned graph. Users can specify the
    ndata name by the ``tag_pos_name`` argument.

    Note that the function will not change the edge ID neither
    how the edge features are stored. The input graph must
    allow CSC format. The graph must be on CPU.

    If the input graph is heterogenous, it must have only one edge
    type and two node types (i.e., source and destination node types).
    In this case, the provided node tags are for the source nodes,
    and the tag offsets are stored in the destination node data.

    The sorted graph and the calculated tag offsets are needed by
    certain operators that consider node tags. See :func:`~dgl.sampling.sample_neighbors_biased`
    for an example.

    Parameters
    ------------
    g : DGLGraph
        The input graph.
    tag : Tensor
        Integer tensor of shape :math:`(N,)`, :math:`N` being the number
        of (source) nodes or edges.
    tag_offset_name : str
        The name of the node feature to store tag offsets.
    tag_type : str
        Tag type which could be ``node`` or ``edge``.

    Returns
    -------
    g_sorted : DGLGraph
        A new graph whose CSC matrix is sorted. The node/edge features of the
        input graph is shallow-copied over.

        - ``g_sorted.ndata[tag_offset_name]`` : Tensor of shape :math:`(N, max\_tag + 2)`.
        - If ``g`` is heterogeneous, get from ``g_sorted.dstdata``.

    Examples
    -----------

    ``tag_type`` is ``node``.

    >>> import dgl
    >>> import torch
    >>> g = dgl.graph(([0,1,2,3,4,0,1,2],[0,0,0,0,0,1,1,1]))
    >>> g.adj_external(scipy_fmt='csr', transpose=True).nonzero()
    (array([0, 0, 0, 0, 0, 1, 1, 1], dtype=int32),
     array([0, 1, 2, 3, 4, 0, 1, 2], dtype=int32)))
    >>> tag = torch.IntTensor([1,1,0,2,0])
    >>> g_sorted = dgl.sort_csc_by_tag(g, tag)
    >>> g_sorted.adj_external(scipy_fmt='csr', transpose=True).nonzero()
    (array([0, 0, 0, 0, 0, 1, 1, 1], dtype=int32),
     array([2, 4, 0, 1, 3, 2, 0, 1], dtype=int32))
    >>> g_sorted.ndata['_TAG_OFFSET']
    tensor([[0, 2, 4, 5],
            [0, 1, 3, 3],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]])

    ``tag_type`` is ``edge``.

    >>> g = dgl.graph(([0,1,2,3,4,0,1,2],[0,0,0,0,0,1,1,1]))
    >>> tag = torch.tensor([1, 1, 0, 2, 0, 1, 1, 0])
    >>> g_sorted = dgl.sort_csc_by_tag(g, tag, tag_type='edge')
    >>> g_sorted.adj_external(scipy_fmt='csr', transpose=True).nonzero()
    (array([0, 0, 0, 0, 0, 1, 1, 1], dtype=int32), array([2, 4, 0, 1, 3, 2, 0, 1], dtype=int32))
    >>> g_sorted.dstdata['_TAG_OFFSET']
    tensor([[0, 2, 4, 5],
            [0, 1, 3, 3],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]])

    See Also
    --------
    dgl.sampling.sample_neighbors_biased
    """
    if len(g.etypes) > 1:
        raise DGLError("Only support homograph and bipartite graph")
    assert tag_type in [
        "node",
        "edge",
    ], "tag_type should be either 'node' or 'edge'."
    if tag_type == "node":
        src, _ = g.edges()
        tag = F.gather_row(tag, F.tensor(src))
    assert len(tag) == g.num_edges()
    num_tags = int(F.asnumpy(F.max(tag, 0))) + 1
    tag_arr = F.zerocopy_to_dgl_ndarray(tag)
    new_g = g.clone()
    new_g._graph, tag_pos_arr = _CAPI_DGLHeteroSortInEdges(
        g._graph, tag_arr, num_tags
    )
    new_g.dstdata[tag_offset_name] = F.from_dgl_nd(tag_pos_arr)
    return new_g


def reorder_graph(
    g,
    node_permute_algo=None,
    edge_permute_algo="src",
    store_ids=True,
    permute_config=None,
):
    r"""Return a new graph with nodes and edges re-ordered/re-labeled
    according to the specified permute algorithm.

    Support homogeneous graph only for the moment.

    The re-ordering has two 2 steps: first re-order nodes and then re-order edges.

    For node permutation, users can re-order by the :attr:`node_permute_algo`
    argument. For edge permutation, user can re-arrange edges according to their
    source nodes or destination nodes by the :attr:`edge_permute_algo` argument.
    Some of the permutation algorithms are only implemented in CPU, so if the
    input graph is on GPU, it will be copied to CPU first. The storage order of
    the node and edge features in the graph are permuted accordingly.

    Parameters
    ----------
    g : DGLGraph
        The homogeneous graph.
    node_permute_algo: str, optional
        The permutation algorithm to re-order nodes. If given, the options are ``rcmk`` or
        ``metis`` or ``custom``.

        * ``None``: Keep the current node order.
        * ``rcmk``: Use the `Reverse Cuthill–McKee <https://docs.scipy.org/doc/scipy/reference/
          generated/scipy.sparse.csgraph.reverse_cuthill_mckee.html#
          scipy-sparse-csgraph-reverse-cuthill-mckee>`__ from ``scipy`` to generate nodes
          permutation.
        * ``metis``: Use the :func:`~dgl.metis_partition_assignment` function
          to partition the input graph, which gives a cluster assignment of each node.
          DGL then sorts the assignment array so the new node order will put nodes of
          the same cluster together. Please note that the generated nodes permutation
          of ``metis`` is non-deterministic due to algorithm's nature.
        * ``custom``: Reorder the graph according to the user-provided node permutation
          array (provided in :attr:`permute_config`).
    edge_permute_algo: str, optional
        The permutation algorithm to reorder edges. Options are ``src`` or ``dst`` or
        ``custom``. ``src`` is the default value.

        * ``src``: Edges are arranged according to their source nodes.
        * ``dst``: Edges are arranged according to their destination nodes.
        * ``custom``: Edges are arranged according to the user-provided edge permutation
          array (provided in :attr:`permute_config`).
    store_ids: bool, optional
        If True, DGL will store the original node and edge IDs in the ndata and edata
        of the resulting graph under name ``dgl.NID`` and ``dgl.EID``, respectively.
    permute_config: dict, optional
        Additional key-value config data for the specified permutation algorithm.

        * For ``rcmk``, this argument is not required.
        * For ``metis``, users should specify the number of partitions ``k`` (e.g.,
          ``permute_config={'k':10}`` to partition the graph to 10 clusters).
        * For ``custom`` node reordering, users should provide a node permutation
          array ``nodes_perm``. The array must be an integer list or a tensor with
          the same device of the input graph.
        * For ``custom`` edge reordering, users should provide an edge permutation
          array ``edges_perm``. The array must be an integer list or a tensor with
          the same device of the input graph.

    Returns
    -------
    DGLGraph
        The re-ordered graph.

    Examples
    --------
    >>> import dgl
    >>> import torch
    >>> g = dgl.graph((torch.tensor([0, 1, 2, 3, 4]), torch.tensor([2, 2, 3, 2, 3])))
    >>> g.ndata['h'] = torch.arange(g.num_nodes() * 2).view(g.num_nodes(), 2)
    >>> g.edata['w'] = torch.arange(g.num_edges() * 1).view(g.num_edges(), 1)
    >>> g.ndata
    {'h': tensor([[0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9]])}
    >>> g.edata
    {'w': tensor([[0],
            [1],
            [2],
            [3],
            [4]])}

    Reorder according to ``'rcmk'`` permute algorithm.

    >>> rg = dgl.reorder_graph(g, node_permute_algo='rcmk')
    >>> rg.ndata
    {'h': tensor([[8, 9],
            [6, 7],
            [2, 3],
            [4, 5],
            [0, 1]]), '_ID': tensor([4, 3, 1, 2, 0])}
    >>> rg.edata
    {'w': tensor([[4],
            [3],
            [1],
            [2],
            [0]]), '_ID': tensor([4, 3, 1, 2, 0])}

    Reorder according to ``'metis'`` permute algorithm.

    >>> rg = dgl.reorder_graph(g, node_permute_algo='metis', permute_config={'k':2})
    >>> rg.ndata
    {'h': tensor([[4, 5],
            [2, 3],
            [0, 1],
            [8, 9],
            [6, 7]]), '_ID': tensor([2, 1, 0, 4, 3])}
    >>> rg.edata
    {'w': tensor([[2],
            [1],
            [0],
            [4],
            [3]]), '_ID': tensor([2, 1, 0, 4, 3])}

    Reorder according to ``'custom'`` permute algorithm with user-provided nodes_perm.

    >>> rg = dgl.reorder_graph(g, node_permute_algo='custom',
    ...                        permute_config={'nodes_perm': [3, 2, 0, 4, 1]})
    >>> rg.ndata
    {'h': tensor([[6, 7],
            [4, 5],
            [0, 1],
            [8, 9],
            [2, 3]]), '_ID': tensor([3, 2, 0, 4, 1])}
    >>> rg.edata
    {'w': tensor([[3],
            [2],
            [0],
            [4],
            [1]]), '_ID': tensor([3, 2, 0, 4, 1])}

    Reorder nodes according to ``'rcmk'`` and reorder edges according to ``dst``
    edge permute algorithm.

    >>> rg = dgl.reorder_graph(g, node_permute_algo='rcmk', edge_permute_algo='dst')
    >>> print(rg.ndata)
    {'h': tensor([[8, 9],
            [6, 7],
            [2, 3],
            [4, 5],
            [0, 1]]), '_ID': tensor([4, 3, 1, 2, 0])}
    >>> print(rg.edata)
    {'w': tensor([[4],
            [2],
            [3],
            [1],
            [0]]), '_ID': tensor([4, 2, 3, 1, 0])}

    Nodes are not reordered but edges are reordered according to ``'custom'`` permute
    algorithm with user-provided edges_perm.

    >>> rg = dgl.reorder_graph(g, edge_permute_algo='custom',
    ...                        permute_config={'edges_perm': [1, 2, 3, 4, 0]})
    >>> print(rg.ndata)
    {'h': tensor([[0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9]]), '_ID': tensor([0, 1, 2, 3, 4])}
    >>> print(rg.edata)
    {'w': tensor([[1],
            [2],
            [3],
            [4],
            [0]]), '_ID': tensor([1, 2, 3, 4, 0])}
    """
    # sanity checks
    if not g.is_homogeneous:
        raise DGLError("Only homogeneous graphs are supported.")
    expected_node_algo = ["rcmk", "metis", "custom"]
    if (
        node_permute_algo is not None
        and node_permute_algo not in expected_node_algo
    ):
        raise DGLError(
            "Unexpected node_permute_algo is specified: {}. Expected algos: {}".format(
                node_permute_algo, expected_node_algo
            )
        )
    expected_edge_algo = ["src", "dst", "custom"]
    if edge_permute_algo not in expected_edge_algo:
        raise DGLError(
            "Unexpected edge_permute_algo is specified: {}. Expected algos: {}".format(
                edge_permute_algo, expected_edge_algo
            )
        )

    g.edata["__orig__"] = F.arange(0, g.num_edges(), g.idtype, g.device)

    # reorder nodes
    if node_permute_algo == "rcmk":
        nodes_perm = rcmk_perm(g)
        rg = subgraph.node_subgraph(g, nodes_perm, store_ids=False)
    elif node_permute_algo == "metis":
        if permute_config is None or "k" not in permute_config:
            raise DGLError(
                "Partition parts 'k' is required for metis. Please specify in permute_config."
            )
        nodes_perm = metis_perm(g, permute_config["k"])
        rg = subgraph.node_subgraph(g, nodes_perm, store_ids=False)
    elif node_permute_algo == "custom":
        if permute_config is None or "nodes_perm" not in permute_config:
            raise DGLError(
                "node_permute_algo is specified as custom, but no 'nodes_perm' is specified in \
                    permute_config."
            )
        nodes_perm = permute_config["nodes_perm"]
        if len(nodes_perm) != g.num_nodes():
            raise DGLError(
                "Length of 'nodes_perm' ({}) does not \
                    match graph num_nodes ({}).".format(
                    len(nodes_perm), g.num_nodes()
                )
            )
        rg = subgraph.node_subgraph(g, nodes_perm, store_ids=False)
    else:
        nodes_perm = F.arange(0, g.num_nodes(), g.idtype, g.device)
        rg = g.clone()

    if store_ids:
        rg.ndata[NID] = F.copy_to(F.tensor(nodes_perm, g.idtype), g.device)

    g.edata.pop("__orig__")

    # reorder edges
    if edge_permute_algo == "src":
        edges_perm = np.argsort(F.asnumpy(rg.edges()[0]))
        rg = subgraph.edge_subgraph(
            rg, edges_perm, relabel_nodes=False, store_ids=False
        )
    elif edge_permute_algo == "dst":
        edges_perm = np.argsort(F.asnumpy(rg.edges()[1]))
        rg = subgraph.edge_subgraph(
            rg, edges_perm, relabel_nodes=False, store_ids=False
        )
    elif edge_permute_algo == "custom":
        if permute_config is None or "edges_perm" not in permute_config:
            raise DGLError(
                "edge_permute_algo is specified as custom, but no 'edges_perm' is specified in \
                    permute_config."
            )
        edges_perm = permute_config["edges_perm"]
        # First revert the edge reorder caused by node reorder and then
        # apply user-provided edge permutation
        rev_id = F.argsort(rg.edata["__orig__"], 0, False)
        edges_perm = F.astype(
            F.gather_row(rev_id, F.tensor(edges_perm)), rg.idtype
        )
        rg = subgraph.edge_subgraph(
            rg, edges_perm, relabel_nodes=False, store_ids=False
        )

    if store_ids:
        rg.edata[EID] = rg.edata.pop("__orig__")

    return rg


DGLGraph.reorder_graph = utils.alias_func(reorder_graph)


def metis_perm(g, k):
    r"""Return nodes permutation according to ``'metis'`` algorithm.

    For internal use.

    Parameters
    ----------
    g : DGLGraph
        The homogeneous graph.
    k: int
        The partition parts number.

    Returns
    -------
    iterable[int]
        The nodes permutation.
    """
    pids = metis_partition_assignment(
        g if g.device == F.cpu() else g.to(F.cpu()), k
    )
    pids = F.asnumpy(pids)
    return np.argsort(pids).copy()


def rcmk_perm(g):
    r"""Return nodes permutation according to ``'rcmk'`` algorithm.

    For internal use.

    Parameters
    ----------
    g : DGLGraph
        The homogeneous graph.

    Returns
    -------
    iterable[int]
        The nodes permutation.
    """
    fmat = "csr"
    allowed_fmats = sum(g.formats().values(), [])
    if fmat not in allowed_fmats:
        g = g.formats(allowed_fmats + [fmat])
    csr_adj = g.adj_external(scipy_fmt=fmat)
    perm = sparse.csgraph.reverse_cuthill_mckee(csr_adj)
    return perm.copy()


def norm_by_dst(g, etype=None):
    r"""Calculate normalization coefficient per edge based on destination node degree.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    etype : str or (str, str, str), optional
        The type of the edges to calculate. The allowed edge type formats are:

        * ``(str, str, str)`` for source node type, edge type and destination node type.
        * or one ``str`` edge type name if the name can uniquely identify a
          triplet format in the graph.

        It can be omitted if the graph has a single edge type.

    Returns
    -------
    1D Tensor
        The normalization coefficient of the edges.

    Examples
    --------

    >>> import dgl
    >>> g = dgl.graph(([0, 1, 1], [1, 1, 2]))
    >>> print(dgl.norm_by_dst(g))
    tensor([0.5000, 0.5000, 1.0000])
    """
    _, v, _ = g.edges(form="all", etype=etype)
    _, inv_index, count = F.unique(v, return_inverse=True, return_counts=True)
    deg = F.astype(count[inv_index], F.float32)
    norm = 1.0 / deg
    norm = F.replace_inf_with_zero(norm)

    return norm


def radius_graph(
    x,
    r,
    p=2,
    self_loop=False,
    compute_mode="donot_use_mm_for_euclid_dist",
    get_distances=False,
):
    r"""Construct a graph from a set of points with neighbors within given distance.

    The function transforms the coordinates/features of a point set
    into a bidirected homogeneous graph. The coordinates of the point
    set is specified as a matrix whose rows correspond to points and
    columns correspond to coordinate/feature dimensions.

    The nodes of the returned graph correspond to the points, where the neighbors
    of each point are within given distance.

    The function requires the PyTorch backend.

    Parameters
    ----------
    x : Tensor
        The point coordinates. It can be either on CPU or GPU.
        Device of the point coordinates specifies device of the radius graph and
        ``x[i]`` corresponds to the i-th node in the radius graph.
    r : float
        Radius of the neighbors.
    p : float, optional
        Power parameter for the Minkowski metric. When :attr:`p = 1` it is the
        equivalent of Manhattan distance (L1 norm) and Euclidean distance
        (L2 norm) for :attr:`p = 2`.

        (default: 2)
    self_loop : bool, optional
        Whether the radius graph will contain self-loops.

        (default: False)
    compute_mode : str, optional
        ``use_mm_for_euclid_dist_if_necessary`` - will use matrix multiplication
        approach to calculate euclidean distance (p = 2) if P > 25 or R > 25
        ``use_mm_for_euclid_dist`` - will always use matrix multiplication
        approach to calculate euclidean distance (p = 2)
        ``donot_use_mm_for_euclid_dist`` - will never use matrix multiplication
        approach to calculate euclidean distance (p = 2).

        (default: donot_use_mm_for_euclid_dist)
    get_distances : bool, optional
        Whether to return the distances for the corresponding edges in the
        radius graph.

        (default: False)

    Returns
    -------
    DGLGraph
        The constructed graph. The node IDs are in the same order as :attr:`x`.
    torch.Tensor, optional
        The distances for the edges in the constructed graph. The distances are
        in the same order as edge IDs.

    Examples
    --------

    The following examples use PyTorch backend.

    >>> import dgl
    >>> import torch

    >>> x = torch.tensor([[0.0, 0.0, 1.0],
    ...                   [1.0, 0.5, 0.5],
    ...                   [0.5, 0.2, 0.2],
    ...                   [0.3, 0.2, 0.4]])
    >>> r_g = dgl.radius_graph(x, 0.75)  # Each node has neighbors within 0.75 distance
    >>> r_g.edges()
    (tensor([0, 1, 2, 2, 3, 3]), tensor([3, 2, 1, 3, 0, 2]))

    When :attr:`get_distances` is True, function returns the radius graph and
    distances for the corresponding edges.

    >>> x = torch.tensor([[0.0, 0.0, 1.0],
    ...                   [1.0, 0.5, 0.5],
    ...                   [0.5, 0.2, 0.2],
    ...                   [0.3, 0.2, 0.4]])
    >>> r_g, dist = dgl.radius_graph(x, 0.75, get_distances=True)
    >>> r_g.edges()
    (tensor([0, 1, 2, 2, 3, 3]), tensor([3, 2, 1, 3, 0, 2]))
    >>> dist
    tensor([[0.7000],
            [0.6557],
            [0.6557],
            [0.2828],
            [0.7000],
            [0.2828]])
    """
    # check invalid r
    if r <= 0:
        raise DGLError("Invalid r value. expect r > 0, got r = {}".format(r))

    # check empty point set
    if F.shape(x)[0] == 0:
        raise DGLError("Find empty point set")

    distances = th.cdist(x, x, p=p, compute_mode=compute_mode)

    if not self_loop:
        distances.fill_diagonal_(r + 1)

    edges = th.nonzero(distances <= r, as_tuple=True)

    g = convert.graph(edges, num_nodes=x.shape[0], device=x.device)

    if get_distances:
        distances = distances[edges].unsqueeze(-1)

        return g, distances

    return g


def random_walk_pe(g, k, eweight_name=None):
    r"""Random Walk Positional Encoding, as introduced in
    `Graph Neural Networks with Learnable Structural and Positional Representations
    <https://arxiv.org/abs/2110.07875>`__

    This function computes the random walk positional encodings as landing probabilities
    from 1-step to k-step, starting from each node to itself.

    Parameters
    ----------
    g : DGLGraph
        The input graph. Must be homogeneous.
    k : int
        The number of random walk steps. The paper found the best value to be 16 and 20
        for two experiments.
    eweight_name : str, optional
        The name to retrieve the edge weights. Default: None, not using the edge weights.

    Returns
    -------
    Tensor
        The random walk positional encodings of shape :math:`(N, k)`, where :math:`N` is the
        number of nodes in the input graph.

    Example
    -------
    >>> import dgl
    >>> g = dgl.graph(([0,1,1], [1,1,0]))
    >>> dgl.random_walk_pe(g, 2)
    tensor([[0.0000, 0.5000],
            [0.5000, 0.7500]])
    """
    N = g.num_nodes()  # number of nodes
    M = g.num_edges()  # number of edges
    A = g.adj_external(scipy_fmt="csr")  # adjacency matrix
    if eweight_name is not None:
        # add edge weights if required
        W = sparse.csr_matrix(
            (g.edata[eweight_name].squeeze(), g.find_edges(list(range(M)))),
            shape=(N, N),
        )
        A = A.multiply(W)
    # 1-step transition probability
    if Version(scipy.__version__) < Version("1.11.0"):
        RW = np.array(A / (A.sum(1) + 1e-30))
    else:
        # Sparse matrix divided by a dense array returns a sparse matrix in
        # scipy since 1.11.0.
        RW = (A / (A.sum(1) + 1e-30)).toarray()

    # Iterate for k steps
    PE = [F.astype(F.tensor(RW.diagonal()), F.float32)]
    RW_power = RW
    for _ in range(k - 1):
        RW_power = RW_power @ RW
        PE.append(F.astype(F.tensor(RW_power.diagonal()), F.float32))
    PE = F.stack(PE, dim=-1)

    return PE


def lap_pe(g, k, padding=False, return_eigval=False):
    r"""Laplacian Positional Encoding, as introduced in
    `Benchmarking Graph Neural Networks
    <https://arxiv.org/abs/2003.00982>`__

    This function computes the laplacian positional encodings as the
    k smallest non-trivial eigenvectors.

    Parameters
    ----------
    g : DGLGraph
        The input graph. Must be homogeneous and bidirected.
    k : int
        Number of smallest non-trivial eigenvectors to use for positional
        encoding.
    padding : bool, optional
        If False, raise an exception when k>=n. Otherwise, add zero paddings
        in the end of eigenvectors and 'nan' paddings in the end of eigenvalues
        when k>=n. Default: False. n is the number of nodes in the given graph.
    return_eigval : bool, optional
        If True, return laplacian eigenvalues together with eigenvectors.
        Otherwise, return laplacian eigenvectors only.
        Default: False.

    Returns
    -------
    Tensor or (Tensor, Tensor)
        Return the laplacian positional encodings of shape :math:`(N, k)`,
        where :math:`N` is the number of nodes in the input graph, when
        :attr:`return_eigval` is False. The eigenvalues of shape :math:`N` is
        additionally returned as the second element when :attr:`return_eigval`
        is True.

    Example
    -------
    >>> import dgl
    >>> g = dgl.graph(([0,1,2,3,1,2,3,0], [1,2,3,0,0,1,2,3]))
    >>> dgl.lap_pe(g, 2)
    tensor([[ 7.0711e-01, -6.4921e-17],
            [ 3.0483e-16, -7.0711e-01],
            [-7.0711e-01, -2.4910e-16],
            [ 9.9288e-17,  7.0711e-01]])
    >>> dgl.lap_pe(g, 5, padding=True)
    tensor([[ 7.0711e-01, -6.4921e-17,  5.0000e-01,  0.0000e+00,  0.0000e+00],
            [ 3.0483e-16, -7.0711e-01, -5.0000e-01,  0.0000e+00,  0.0000e+00],
            [-7.0711e-01, -2.4910e-16,  5.0000e-01,  0.0000e+00,  0.0000e+00],
            [ 9.9288e-17,  7.0711e-01, -5.0000e-01,  0.0000e+00,  0.0000e+00]])
    >>> dgl.lap_pe(g, 5, padding=True, return_eigval=True)
    (tensor([[-7.0711e-01,  6.4921e-17, -5.0000e-01,  0.0000e+00,  0.0000e+00],
             [-3.0483e-16,  7.0711e-01,  5.0000e-01,  0.0000e+00,  0.0000e+00],
             [ 7.0711e-01,  2.4910e-16, -5.0000e-01,  0.0000e+00,  0.0000e+00],
             [-9.9288e-17, -7.0711e-01,  5.0000e-01,  0.0000e+00,  0.0000e+00]]),
     tensor([1., 1., 2., nan, nan]))
    """
    # check for the "k < n" constraint
    n = g.num_nodes()
    if not padding and n <= k:
        assert (
            "the number of eigenvectors k must be smaller than the number of "
            + f"nodes n, {k} and {n} detected."
        )

    # get laplacian matrix as I - D^-0.5 * A * D^-0.5
    A = g.adj_external(scipy_fmt="csr")  # adjacency matrix
    N = sparse.diags(
        F.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float
    )  # D^-1/2
    L = sparse.eye(g.num_nodes()) - N * A * N

    # select eigenvectors with smaller eigenvalues O(n + klogk)
    EigVal, EigVec = np.linalg.eig(L.toarray())
    max_freqs = min(n - 1, k)
    kpartition_indices = np.argpartition(EigVal, max_freqs)[: max_freqs + 1]
    topk_eigvals = EigVal[kpartition_indices]
    topk_indices = kpartition_indices[topk_eigvals.argsort()][1:]
    topk_EigVec = EigVec[:, topk_indices]
    eigvals = F.tensor(EigVal[topk_indices], dtype=F.float32)

    # get random flip signs
    rand_sign = 2 * (np.random.rand(max_freqs) > 0.5) - 1.0
    PE = F.astype(F.tensor(rand_sign * topk_EigVec), F.float32)

    # add paddings
    if n <= k:
        temp_EigVec = F.zeros(
            [n, k - n + 1], dtype=F.float32, ctx=F.context(PE)
        )
        PE = F.cat([PE, temp_EigVec], dim=1)
        temp_EigVal = F.tensor(np.full(k - n + 1, np.nan), F.float32)
        eigvals = F.cat([eigvals, temp_EigVal], dim=0)

    if return_eigval:
        return PE, eigvals
    return PE


def laplacian_pe(g, k, padding=False, return_eigval=False):
    r"""Alias of `dgl.lap_pe`."""
    dgl_warning("dgl.laplacian_pe will be deprecated. Use dgl.lap_pe please.")
    return lap_pe(g, k, padding, return_eigval)


def to_bfloat16(g):
    r"""Cast this graph to use bfloat16 for any
    floating-point edge and node feature data.

    A shallow copy is returned so that the original graph is not modified.
    Feature tensors that are not floating-point will not be modified.

    Returns
    -------
    DGLGraph
        Clone of graph with the feature data converted to float16.
    """
    ret = copy.copy(g)
    ret._edge_frames = [frame.bfloat16() for frame in ret._edge_frames]
    ret._node_frames = [frame.bfloat16() for frame in ret._node_frames]
    return ret


def to_half(g):
    r"""Cast this graph to use float16 (half-precision) for any
    floating-point edge and node feature data.

    A shallow copy is returned so that the original graph is not modified.
    Feature tensors that are not floating-point will not be modified.

    Returns
    -------
    DGLGraph
        Clone of graph with the feature data converted to float16.
    """
    ret = copy.copy(g)
    ret._edge_frames = [frame.half() for frame in ret._edge_frames]
    ret._node_frames = [frame.half() for frame in ret._node_frames]
    return ret


def to_float(g):
    r"""Cast this graph to use float32 (single-precision) for any
    floating-point edge and node feature data.

    A shallow copy is returned so that the original graph is not modified.
    Feature tensors that are not floating-point will not be modified.

    Returns
    -------
    DGLGraph
        Clone of graph with the feature data converted to float32.
    """
    ret = copy.copy(g)
    ret._edge_frames = [frame.float() for frame in ret._edge_frames]
    ret._node_frames = [frame.float() for frame in ret._node_frames]
    return ret


def to_double(g):
    r"""Cast this graph to use float64 (double-precision) for any
    floating-point edge and node feature data.

    A shallow copy is returned so that the original graph is not modified.
    Feature tensors that are not floating-point will not be modified.

    Returns
    -------
    DGLGraph
        Clone of graph with the feature data converted to float64.
    """
    ret = copy.copy(g)
    ret._edge_frames = [frame.double() for frame in ret._edge_frames]
    ret._node_frames = [frame.double() for frame in ret._node_frames]
    return ret


def double_radius_node_labeling(g, src, dst):
    r"""Double Radius Node Labeling, as introduced in `Link Prediction
    Based on Graph Neural Networks <https://arxiv.org/abs/1802.09691>`__.

    This function computes the double radius node labeling for each node to mark
    nodes' different roles in an enclosing subgraph, given a target link.

    The node labels of source :math:`s` and destination :math:`t` are set to 1 and
    those of unreachable nodes from source or destination are set to 0. The labels
    of other nodes :math:`l` are defined according to the following hash function:

    :math:`l = 1 + min(d_s, d_t) + (d//2)[(d//2) + (d%2) - 1]`

    where :math:`d_s` and :math:`d_t` denote the shortest distance to the source and
    the target, respectively. :math:`d = d_s + d_t`.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    src : int
        The source node ID of the target link.
    dst : int
        The destination node ID of the target link.

    Returns
    -------
    Tensor
        Labels of all nodes. The tensor is of shape :math:`(N,)`, where
        :math:`N` is the number of nodes in the input graph.

    Example
    -------
    >>> import dgl

    >>> g = dgl.graph(([0,0,0,0,1,1,2,4], [1,2,3,6,3,4,4,5]))
    >>> dgl.double_radius_node_labeling(g, 0, 1)
    tensor([1, 1, 3, 2, 3, 7, 0])
    """
    adj = g.adj_external(scipy_fmt="csr")
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    # distance to the source node
    ds = sparse.csgraph.shortest_path(
        adj_wo_dst, directed=False, unweighted=True, indices=src
    )
    ds = np.insert(ds, dst, 0, axis=0)
    # distance to the destination node
    dt = sparse.csgraph.shortest_path(
        adj_wo_src, directed=False, unweighted=True, indices=dst - 1
    )
    dt = np.insert(dt, src, 0, axis=0)

    d = ds + dt
    # suppress invalid value (nan) warnings
    with np.errstate(invalid="ignore"):
        z = 1 + np.stack([ds, dt]).min(axis=0) + d // 2 * (d // 2 + d % 2 - 1)
    z[src] = 1
    z[dst] = 1
    z[np.isnan(z)] = 0  # unreachable nodes

    return F.tensor(z, F.int64)


def shortest_dist(g, root=None, return_paths=False):
    r"""Compute shortest distance and paths on the given graph.

    Only unweighted cases are supported. Only directed paths (in which the
    edges are all oriented in the same direction) are considered effective.

    Parameters
    ----------
    g : DGLGraph
        The input graph. Must be homogeneous.
    root : int, optional
        Given a root node ID, it returns the shortest distance and paths
        (optional) between the root node and all the nodes. If None, it returns
        the results for all node pairs. Default: None.
    return_paths : bool, optional
        If True, it returns the shortest paths corresponding to the shortest
        distances. Default: False.

    Returns
    -------
    dist : Tensor
        The shortest distance tensor.

        * If :attr:`root` is a node ID, it is a tensor of shape :math:`(N,)`,
          where :math:`N` is the number of nodes. :attr:`dist[j]` gives the
          shortest distance from :attr:`root` to node :attr:`j`.
        * Otherwise, it is a tensor of shape :math:`(N, N)`. :attr:`dist[i][j]`
          gives the shortest distance from node :attr:`i` to node :attr:`j`.
        * The distance values of unreachable node pairs are filled with -1.
    paths : Tensor, optional
        The shortest path tensor. It is only returned when :attr:`return_paths`
        is True.

        * If :attr:`root` is a node ID, it is a tensor of shape :math:`(N, L)`,
          where :math:`L` is the length of the longest path. :attr:`path[j]` is
          the shortest path from node :attr:`root` to node :attr:`j`.
        * Otherwise, it is a tensor of shape :math:`(N, N, L)`.
          :attr:`path[i][j]` is the shortest path from node :attr:`i` to node
          :attr:`j`.
        * Each path is a vector that consists of edge IDs with paddings of -1
          at the end.
        * Shortest path between a node and itself is a vector filled with -1's.

    Example
    -------
    >>> import dgl

    >>> g = dgl.graph(([0, 1, 1, 2], [2, 0, 3, 3]))
    >>> dgl.shortest_dist(g, root=0)
    tensor([ 0,  -1,  1, 2])
    >>> dist, paths = dgl.shortest_dist(g, root=None, return_paths=True)
    >>> print(dist)
    tensor([[ 0, -1,  1,  2],
            [ 1,  0,  2,  1],
            [-1, -1,  0,  1],
            [-1, -1, -1,  0]])
    >>> print(paths)
    tensor([[[-1, -1],
             [-1, -1],
             [ 0, -1],
             [ 0,  3]],
    <BLANKLINE>
            [[ 1, -1],
             [-1, -1],
             [ 1,  0],
             [ 2, -1]],
    <BLANKLINE>
            [[-1, -1],
             [-1, -1],
             [-1, -1],
             [ 3, -1]],
    <BLANKLINE>
            [[-1, -1],
             [-1, -1],
             [-1, -1],
             [-1, -1]]])
    """
    if root is None:
        dist, pred = sparse.csgraph.shortest_path(
            g.adj_external(scipy_fmt="csr"),
            return_predecessors=True,
            unweighted=True,
            directed=True,
        )
    else:
        dist, pred = sparse.csgraph.dijkstra(
            g.adj_external(scipy_fmt="csr"),
            directed=True,
            indices=root,
            return_predecessors=True,
            unweighted=True,
        )
    dist[np.isinf(dist)] = -1

    if not return_paths:
        return F.copy_to(F.tensor(dist, dtype=F.int64), g.device)

    def _get_nodes(pred, i, j):
        r"""return node IDs of a path from i to j given predecessors"""
        if i == j:
            return []
        prev = pred[j]
        nodes = [j, prev]
        while prev != i:
            prev = pred[prev]
            nodes.append(prev)
        nodes.reverse()

        return nodes

    # construct paths with given predecessors
    max_len = int(dist[~np.isinf(dist)].max())
    N = g.num_nodes()
    roots = list(range(N)) if root is None else [root]
    paths = np.ones([len(roots), N, max_len], dtype=np.int64) * -1
    masks, u, v = [], [], []
    for i in roots:
        pred_ = pred[i] if root is None else pred
        masks_i = np.zeros([N, max_len], dtype=bool)
        for j in range(N):
            if pred_[j] < 0:
                continue
            nodes = _get_nodes(pred_, i, j)
            u.extend(nodes[:-1])
            v.extend(nodes[1:])
            if nodes:
                masks_i[j, : len(nodes) - 1] = True
        masks.append(masks_i)
    masks = np.stack(masks, axis=0)

    u, v = np.array(u), np.array(v)
    edge_ids = g.edge_ids(u, v)
    paths[masks] = F.asnumpy(edge_ids)
    if root is not None:
        paths = paths[0]

    return F.copy_to(F.tensor(dist, dtype=F.int64), g.device), F.copy_to(
        F.tensor(paths, dtype=F.int64), g.device
    )


def svd_pe(g, k, padding=False, random_flip=True):
    r"""SVD-based Positional Encoding, as introduced in
    `Global Self-Attention as a Replacement for Graph Convolution
    <https://arxiv.org/pdf/2108.03348.pdf>`__

    This function computes the largest :math:`k` singular values and
    corresponding left and right singular vectors to form positional encodings.

    Parameters
    ----------
    g : DGLGraph
        A DGLGraph to be encoded, which must be a homogeneous one.
    k : int
        Number of largest singular values and corresponding singular vectors
        used for positional encoding.
    padding : bool, optional
        If False, raise an error when :math:`k > N`,
        where :math:`N` is the number of nodes in :attr:`g`.
        If True, add zero paddings in the end of encoding vectors when
        :math:`k > N`.
        Default : False.
    random_flip : bool, optional
        If True, randomly flip the signs of encoding vectors.
        Proposed to be activated during training for better generalization.
        Default : True.

    Returns
    -------
    Tensor
        Return SVD-based positional encodings of shape :math:`(N, 2k)`.

    Example
    -------
    >>> import dgl

    >>> g = dgl.graph(([0,1,2,3,4,2,3,1,4,0], [2,3,1,4,0,0,1,2,3,4]))
    >>> dgl.svd_pe(g, k=2, padding=False, random_flip=True)
    tensor([[-6.3246e-01, -1.1373e-07, -6.3246e-01,  0.0000e+00],
            [-6.3246e-01,  7.6512e-01, -6.3246e-01, -7.6512e-01],
            [ 6.3246e-01,  4.7287e-01,  6.3246e-01, -4.7287e-01],
            [-6.3246e-01, -7.6512e-01, -6.3246e-01,  7.6512e-01],
            [ 6.3246e-01, -4.7287e-01,  6.3246e-01,  4.7287e-01]])
    """
    n = g.num_nodes()
    if not padding and n < k:
        raise ValueError(
            "The number of singular values k must be no greater than the "
            "number of nodes n, but " + f"got {k} and {n} respectively."
        )
    a = g.adj_external(ctx=g.device, scipy_fmt="coo").toarray()
    u, d, vh = scipy.linalg.svd(a)
    v = vh.transpose()
    m = min(n, k)
    topm_u = u[:, 0:m]
    topm_v = v[:, 0:m]
    topm_sqrt_d = sparse.diags(np.sqrt(d[0:m]))
    encoding = np.concatenate(
        ((topm_u @ topm_sqrt_d), (topm_v @ topm_sqrt_d)), axis=1
    )
    # randomly flip row vectors
    if random_flip:
        rand_sign = 2 * (np.random.rand(n) > 0.5) - 1
        flipped_encoding = F.tensor(
            rand_sign[:, np.newaxis] * encoding, dtype=F.float32
        )
    else:
        flipped_encoding = F.tensor(encoding, dtype=F.float32)

    if n < k:
        zero_padding = F.zeros(
            [n, 2 * (k - n)], dtype=F.float32, ctx=F.context(flipped_encoding)
        )
        flipped_encoding = F.cat([flipped_encoding, zero_padding], dim=1)

    return flipped_encoding


_init_api("dgl.transform", __name__)
