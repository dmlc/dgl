"""Modules that transforms between graphs and between graph and tensors."""

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
    K : int
        The number of neighbors

    Inputs
    ------
    h : Tensor
        :math:`(M, D)` or :math:`(N, M, D)` where :math:`N` means the
        number of point sets, :math:`M` means the number of points in
        each point set, and :math:`D` means the size of features.

    Outputs
    -------
    - A DGLGraph with no features.
    """
    def __init__(self, K):
        super(NearestNeighborGraph, self).__init__()
        self.K = K

    def forward(self, h):
        return nearest_neighbor_graph(h, self.K)


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
    K : int
        The number of neighbors

    Inputs
    ------
    h : Tensor
        :math:`(M, D)` where :math:`M` means the total number of points
        in all point sets.
    segs : Tensor
        :math:`(N)` integer tensors where :math:`N` means the number of
        point sets.  The elements must sum up to :math:`M`.

    Outputs
    -------
    - A DGLGraph with no features.
    """
    def __init__(self, K):
        super(NearestNeighborGraph, self).__init__()
        self.K = K

    def forward(self, h, segs):
        n_total_points, n_dims = h.shape

        with torch.no_grad():
            hs = h.split(segs)
            dst = [
                pairwise_squared_distance(h_g).topk(self.K, dim=1, largest=False)[1] +
                segs[i - 1] if i > 0 else 0
                for i, h_g in enumerate(hs)]
            dst = torch.cat(dst, 0)
            src = torch.arange(n_total_points).unsqueeze(1).expand(n_total_points, self.K)

            dst = dst.flatten()
            src = src.flatten()
            adj = ssp.csr_matrix((torch.ones_like(dst).numpy(), (dst.numpy(), src.numpy())))

            g = dgl.DGLGraph(adj, readonly=True)
            return g
