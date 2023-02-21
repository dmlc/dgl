"""Modules that transforms between graphs and between graph and tensors."""
import torch.nn as nn

from ...transforms import knn_graph, radius_graph, segmented_knn_graph


def pairwise_squared_distance(x):
    """
    x : (n_samples, n_points, dims)
    return : (n_samples, n_points, n_points)
    """
    x2s = (x * x).sum(-1, keepdim=True)
    return x2s + x2s.transpose(-1, -2) - 2 * x @ x.transpose(-1, -2)


class KNNGraph(nn.Module):
    r"""Layer that transforms one point set into a graph, or a batch of
    point sets with the same number of points into a batched union of those graphs.

    The KNNGraph is implemented in the following steps:

    1. Compute an NxN matrix of pairwise distance for all points.
    2. Pick the k points with the smallest distance for each point as their k-nearest neighbors.
    3. Construct a graph with edges to each point as a node from its k-nearest neighbors.

    The overall computational complexity is :math:`O(N^2(logN + D)`.

    If a batch of point sets is provided, the point :math:`j` in point
    set :math:`i` is mapped to graph node ID: :math:`i \times M + j`, where
    :math:`M` is the number of nodes in each point set.

    The predecessors of each node are the k-nearest neighbors of the
    corresponding point.

    Parameters
    ----------
    k : int
        The number of neighbors.

    Notes
    -----
    The nearest neighbors found for a node include the node itself.

    Examples
    --------
    The following example uses PyTorch backend.

    >>> import torch
    >>> from dgl.nn.pytorch.factory import KNNGraph
    >>>
    >>> kg = KNNGraph(2)
    >>> x = torch.tensor([[0,1],
                          [1,2],
                          [1,3],
                          [100, 101],
                          [101, 102],
                          [50, 50]])
    >>> g = kg(x)
    >>> print(g.edges())
        (tensor([0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5]),
         tensor([0, 0, 1, 2, 1, 2, 5, 3, 4, 3, 4, 5]))
    """

    def __init__(self, k):
        super(KNNGraph, self).__init__()
        self.k = k

    # pylint: disable=invalid-name
    def forward(
        self,
        x,
        algorithm="bruteforce-blas",
        dist="euclidean",
        exclude_self=False,
    ):
        r"""

        Forward computation.

        Parameters
        ----------
        x : Tensor
            :math:`(M, D)` or :math:`(N, M, D)` where :math:`N` means the
            number of point sets, :math:`M` means the number of points in
            each point set, and :math:`D` means the size of features.
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

            * 'nn-descent' is a approximate approach from paper
              `Efficient k-nearest neighbor graph construction for generic similarity
              measures <https://www.cs.princeton.edu/cass/papers/www11.pdf>`_. This method
              will search for nearest neighbor candidates in "neighbors' neighbors".

            (default: 'bruteforce-blas')
        dist : str, optional
            The distance metric used to compute distance between points. It can be the following
            metrics:
            * 'euclidean': Use Euclidean distance (L2 norm)
              :math:`\sqrt{\sum_{i} (x_{i} - y_{i})^{2}}`.
            * 'cosine': Use cosine distance.
            (default: 'euclidean')
        exclude_self : bool, optional
            If True, the output graph will not contain self loop edges, and each node will not
            be counted as one of its own k neighbors.  If False, the output graph will contain
            self loop edges, and a node will be counted as one of its own k neighbors.

        Returns
        -------
        DGLGraph
            A DGLGraph without features.
        """
        return knn_graph(
            x, self.k, algorithm=algorithm, dist=dist, exclude_self=exclude_self
        )


class SegmentedKNNGraph(nn.Module):
    r"""Layer that transforms one point set into a graph, or a batch of
    point sets with different number of points into a batched union of those graphs.

    If a batch of point sets is provided, then the point :math:`j` in the point
    set :math:`i` is mapped to graph node ID:
    :math:`\sum_{p<i} |V_p| + j`, where :math:`|V_p|` means the number of
    points in the point set :math:`p`.

    The predecessors of each node are the k-nearest neighbors of the
    corresponding point.

    Parameters
    ----------
    k : int
        The number of neighbors.

    Notes
    -----
    The nearest neighbors found for a node include the node itself.

    Examples
    --------
    The following example uses PyTorch backend.

    >>> import torch
    >>> from dgl.nn.pytorch.factory import SegmentedKNNGraph
    >>>
    >>> kg = SegmentedKNNGraph(2)
    >>> x = torch.tensor([[0,1],
    ...                   [1,2],
    ...                   [1,3],
    ...                   [100, 101],
    ...                   [101, 102],
    ...                   [50, 50],
    ...                   [24,25],
    ...                   [25,24]])
    >>> g = kg(x, [3,3,2])
    >>> print(g.edges())
    (tensor([0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7]),
     tensor([0, 0, 1, 2, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 6, 7]))
    >>>

    """

    def __init__(self, k):
        super(SegmentedKNNGraph, self).__init__()
        self.k = k

    # pylint: disable=invalid-name
    def forward(
        self,
        x,
        segs,
        algorithm="bruteforce-blas",
        dist="euclidean",
        exclude_self=False,
    ):
        r"""Forward computation.

        Parameters
        ----------
        x : Tensor
            :math:`(M, D)` where :math:`M` means the total number of points
            in all point sets, and :math:`D` means the size of features.
        segs : iterable of int
            :math:`(N)` integers where :math:`N` means the number of point
            sets.  The number of elements must sum up to :math:`M`. And any
            :math:`N` should :math:`\ge k`
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

            * 'nn-descent' is a approximate approach from paper
              `Efficient k-nearest neighbor graph construction for generic similarity
              measures <https://www.cs.princeton.edu/cass/papers/www11.pdf>`_. This method
              will search for nearest neighbor candidates in "neighbors' neighbors".

            (default: 'bruteforce-blas')
        dist : str, optional
            The distance metric used to compute distance between points. It can be the following
            metrics:
            * 'euclidean': Use Euclidean distance (L2 norm)
              :math:`\sqrt{\sum_{i} (x_{i} - y_{i})^{2}}`.
            * 'cosine': Use cosine distance.
            (default: 'euclidean')
        exclude_self : bool, optional
            If True, the output graph will not contain self loop edges, and each node will not
            be counted as one of its own k neighbors.  If False, the output graph will contain
            self loop edges, and a node will be counted as one of its own k neighbors.

        Returns
        -------
        DGLGraph
            A batched DGLGraph without features.
        """

        return segmented_knn_graph(
            x,
            self.k,
            segs,
            algorithm=algorithm,
            dist=dist,
            exclude_self=exclude_self,
        )


class RadiusGraph(nn.Module):
    r"""Layer that transforms one point set into a bidirected graph with
    neighbors within given distance.

    The RadiusGraph is implemented in the following steps:

    1. Compute an NxN matrix of pairwise distance for all points.
    2. Pick the points within distance to each point as their neighbors.
    3. Construct a graph with edges to each point as a node from its neighbors.

    The nodes of the returned graph correspond to the points, where the neighbors
    of each point are within given distance.

    Parameters
    ----------
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

    Examples
    --------
    The following examples uses PyTorch backend.

    >>> import dgl
    >>> from dgl.nn.pytorch.factory import RadiusGraph

    >>> x = torch.tensor([[0.0, 0.0, 1.0],
    ...                   [1.0, 0.5, 0.5],
    ...                   [0.5, 0.2, 0.2],
    ...                   [0.3, 0.2, 0.4]])
    >>> rg = RadiusGraph(0.75)
    >>> g = rg(x)  # Each node has neighbors within 0.75 distance
    >>> g.edges()
    (tensor([0, 1, 2, 2, 3, 3]), tensor([3, 2, 1, 3, 0, 2]))

    When :attr:`get_distances` is True, forward pass returns the radius graph and
    distances for the corresponding edges.

    >>> x = torch.tensor([[0.0, 0.0, 1.0],
    ...                   [1.0, 0.5, 0.5],
    ...                   [0.5, 0.2, 0.2],
    ...                   [0.3, 0.2, 0.4]])
    >>> rg = RadiusGraph(0.75)
    >>> g, dist = rg(x, get_distances=True)
    >>> g.edges()
    (tensor([0, 1, 2, 2, 3, 3]), tensor([3, 2, 1, 3, 0, 2]))
    >>> dist
    tensor([[0.7000],
            [0.6557],
            [0.6557],
            [0.2828],
            [0.7000],
            [0.2828]])
    """

    # pylint: disable=invalid-name
    def __init__(
        self,
        r,
        p=2,
        self_loop=False,
        compute_mode="donot_use_mm_for_euclid_dist",
    ):
        super(RadiusGraph, self).__init__()
        self.r = r
        self.p = p
        self.self_loop = self_loop
        self.compute_mode = compute_mode

    # pylint: disable=invalid-name
    def forward(self, x, get_distances=False):
        r"""
        Forward computation.

        Parameters
        ----------
        x : Tensor
            The point coordinates. :math:`(N, D)` where :math:`N` means the
            number of points in the point set, and :math:`D` means the size of
            the features. It can be either on CPU or GPU. Device of the point
            coordinates specifies device of the radius graph.
        get_distances : bool, optional
            Whether to return the distances for the corresponding edges in the
            radius graph.

            (default: False)

        Returns
        -------
        DGLGraph
            The constructed graph. The node IDs are in the same order as :attr:`x`.
        torch.Tensor, optional
            The distances for the edges in the constructed graph. The distances
            are in the same order as edge IDs.
        """
        return radius_graph(
            x, self.r, self.p, self.self_loop, self.compute_mode, get_distances
        )
