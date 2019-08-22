import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

class NearestNeighborGraph(nn.Module):
    def __init__(self, k):
        super(NearestNeighborGrpah, self).__init__()
        self.k = k

    def forward(self, h, segs):
        '''
        h : (n_total_points, dims)
        segs : (n_samples,) LongTensor, sum to n_total_points
        - : BatchedDGLGraph, 'x' contains the coordinates
        '''
        start = 0
        gs = []
        for i in range(len(segs)):
            end = start + segs[i]
            h_i = h[start:end]
            d = h_i[:, None] - h_i[None, :]
            d = (d * d).sum(2)
            k_indices = d.topk(self.K)
            dst = (k_indices + start).flatten()
            src = (torch.zeros_like(k_indices) + torch.arange(start, end)[:, None]).flatten()
            g = dgl.DGLGraph()
            g.add_nodes(segs[i])
            g.add_edges(src, dst)
            g.readonly()
            gs.append(g)
        gs = dgl.batch(gs)
        gs.ndata['x'] = h
        return gs
