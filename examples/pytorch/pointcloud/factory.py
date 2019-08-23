import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

def pairwise_squared_distance(x):
    '''
    x : (n_samples, n_points, dims)
    - : (n_samples, n_points, n_points)
    '''
    x2s = (x * x).sum(-1, keepdim=True)
    return x2s + x2s.transpose(-1, -2) - 2 * x @ x.transpose(-1, -2)

class NearestNeighborGraph(nn.Module):
    def __init__(self, K):
        super(NearestNeighborGraph, self).__init__()
        self.K = K

    def forward(self, h):
        '''
        h : (n_samples, n_points, dims)
        segs : (n_samples,) LongTensor, sum to n_total_points
        - : BatchedDGLGraph, 'x' contains the coordinates
        '''
        n_samples, n_points, n_dims = h.shape
        gs = []

        with torch.no_grad():
            d = pairwise_squared_distance(h)
            _, k_indices = d.topk(self.K, dim=2, largest=False)
            k_indices = k_indices.cpu()

        src = (torch.zeros_like(k_indices[0]) + torch.arange(n_points)[:, None]).flatten()

        for i in range(n_samples):
            dst = k_indices[i].flatten()
            g = dgl.DGLGraph()
            g.add_nodes(h.shape[1])
            g.add_edges(dst, src)   # node receive message from nearest neighbors
            g.readonly()
            gs.append(g)

        gs = dgl.batch(gs)
        return gs
