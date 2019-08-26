import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import scipy.sparse as ssp

def pairwise_squared_distance(x):
    '''
    x : (n_samples, n_points, dims)
    return : (n_samples, n_points, n_points)
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
        return : DGLGraph, 'x' contains the coordinates
        '''
        n_samples, n_points, n_dims = h.shape
        gs = []

        with torch.no_grad():
            d = pairwise_squared_distance(h)
            _, k_indices = d.topk(self.K, dim=2, largest=False)
            dst = k_indices.cpu()

        src = torch.zeros_like(dst) + torch.arange(n_points)[None, :, None]

        per_sample_offset = (torch.arange(n_samples) * n_points)[:, None, None]
        dst += per_sample_offset
        src += per_sample_offset
        dst = dst.flatten()
        src = src.flatten()
        adj = ssp.csr_matrix((torch.ones_like(dst).numpy(), (dst.numpy(), src.numpy())))

        g = dgl.DGLGraph(adj, readonly=True)

        return g
