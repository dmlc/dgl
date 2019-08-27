"""Modules that transforms between graphs and between graph and tensors."""
import torch as th
import torch.nn as nn
import scipy.sparse as ssp
from ...transform import nearest_neighbor_graph, segmented_nearest_neighbor_graph
from ...graph import DGLGraph

def pairwise_squared_distance(x):
    '''
    x : (n_samples, n_points, dims)
    return : (n_samples, n_points, n_points)
    '''
    x2s = (x * x).sum(-1, keepdim=True)
    return x2s + x2s.transpose(-1, -2) - 2 * x @ x.transpose(-1, -2)


class NearestNeighborGraph(nn.Module):
    r"""Layer that transforms one point set into a graph, or a batch of
    point sets with the same number of points into a union of those graphs.

    If a batch of point set is provided, then the point :math:`j` in point
    set :math:`i` is mapped to graph node ID :math:`i \times M + j`, where
    :math:`M` is the number of nodes in each point set.

    The predecessors of each node are the k-nearest neighbors of the
    corresponding point.

    Parameters
    ----------
    k : int
        The number of neighbors

    Inputs
    ------
    input : Tensor
        :math:`(M, D)` or :math:`(N, M, D)` where :math:`N` means the
        number of point sets, :math:`M` means the number of points in
        each point set, and :math:`D` means the size of features.

    Outputs
    -------
    - A DGLGraph with no features.
    """
    def __init__(self, k):
        super(NearestNeighborGraph, self).__init__()
        self.k = k

    def forward(self, input):
        return nearest_neighbor_graph(input, self.k)


class SegmentedNearestNeighborGraph(nn.Module):
    r"""Layer that transforms one point set into a graph, or a batch of
    point sets with different number of points into a union of those graphs.

    If a batch of point set is provided, then the point :math:`j` in point
    set :math:`i` is mapped to graph node ID
    :math:`\sum_{p<i} |V_p| + j`, where :math:`|V_p|` means the number of
    points in point set :math:`p`.

    The predecessors of each node are the k-nearest neighbors of the
    corresponding point.

    Parameters
    ----------
    k : int
        The number of neighbors

    Inputs
    ------
    input : Tensor
        :math:`(M, D)` where :math:`M` means the total number of points
        in all point sets.
    segs : Tensor
        :math:`(N)` integer tensors where :math:`N` means the number of
        point sets.  The elements must sum up to :math:`M`.

    Outputs
    -------
    - A DGLGraph with no features.
    """
    def __init__(self, k):
        super(SegmentedNearestNeighborGraph, self).__init__()
        self.k = k

    def forward(self, input, segs):
        return segmented_nearest_neighbor_graph(input, self.k, segs)
