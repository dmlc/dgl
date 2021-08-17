import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.geometry import farthest_point_sampler

'''
Part of the code are adapted from
https://github.com/yanx27/Pointnet_Pointnet2_pytorch
'''


def square_distance(src, dst):
    '''
    Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
    '''
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    '''
    Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
    '''
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(
        device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class KNearNeighbors(nn.Module):
    '''
    Find the k nearest neighbors
    '''

    def __init__(self, n_neighbor):
        super(KNearNeighbors, self).__init__()
        self.n_neighbor = n_neighbor

    def forward(self, pos, centroids):
        '''
        Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
        '''
        device = pos.device
        B, N, _ = pos.shape
        center_pos = index_points(pos, centroids)
        _, S, _ = center_pos.shape
        group_idx = torch.arange(N, dtype=torch.long).to(
            device).view(1, 1, N).repeat([B, S, 1])
        sqrdists = square_distance(center_pos, pos)
        group_idx = group_idx.sort(dim=-1)[0][:, :, :self.n_neighbor]
        group_first = group_idx[:, :, 0].view(
            B, S, 1).repeat([1, 1, self.n_neighbor])
        return group_idx


class KNNGraphBuilder(nn.Module):
    '''
    Build NN graph
    '''

    def __init__(self, n_neighbor):
        super(KNNGraphBuilder, self).__init__()
        self.n_neighbor = n_neighbor
        self.knn = KNearNeighbors(n_neighbor)

    def forward(self, pos, centroids, feat=None):
        dev = pos.device
        group_idx = self.knn(pos, centroids)
        B, N, _ = pos.shape
        glist = []
        for i in range(B):
            center = torch.zeros((N)).to(dev)
            center[centroids[i]] = 1
            src = group_idx[i].contiguous().view(-1)
            dst = centroids[i].view(-1, 1).repeat(1, self.n_neighbor).view(-1)

            unified = torch.cat([src, dst])
            uniq, inv_idx = torch.unique(unified, return_inverse=True)
            src_idx = inv_idx[:src.shape[0]]
            dst_idx = inv_idx[src.shape[0]:]

            g = dgl.graph((src_idx, dst_idx))
            g.ndata['pos'] = pos[i][uniq]
            g.ndata['center'] = center[uniq]
            if feat is not None:
                g.ndata['feat'] = feat[i][uniq]
            glist.append(g)
        bg = dgl.batch(glist)
        return bg


class RelativePositionMessage(nn.Module):
    '''
    Compute the input feature from neighbors
    '''

    def __init__(self, n_neighbor):
        super(RelativePositionMessage, self).__init__()
        self.n_neighbor = n_neighbor

    def forward(self, edges):
        pos = edges.src['pos'] - edges.dst['pos']
        if 'feat' in edges.src:
            res = torch.cat([pos, edges.src['feat']], 1)
        else:
            res = pos
        return {'agg_feat': res}


class KNNConv(nn.Module):
    '''
    Feature aggregation
    '''

    def __init__(self, sizes, batch_size):
        super(KNNConv, self).__init__()
        self.batch_size = batch_size
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        for i in range(1, len(sizes)):
            self.conv.append(nn.Conv2d(sizes[i-1], sizes[i], 1))
            self.bn.append(nn.BatchNorm2d(sizes[i]))

    def forward(self, nodes):
        shape = nodes.mailbox['agg_feat'].shape
        h = nodes.mailbox['agg_feat'].view(
            self.batch_size, -1, shape[1], shape[2]).permute(0, 3, 2, 1)
        for conv, bn in zip(self.conv, self.bn):
            h = conv(h)
            h = bn(h)
            h = F.relu(h)
        h = torch.max(h, 2)[0]
        feat_dim = h.shape[1]
        h = h.permute(0, 2, 1).reshape(-1, feat_dim)
        return {'new_feat': h}

    def group_all(self, pos, feat):
        '''
        Feature aggregation and pooling for the non-sampling layer
        '''
        if feat is not None:
            h = torch.cat([pos, feat], 2)
        else:
            h = pos
        B, N, D = h.shape
        _, _, C = pos.shape
        new_pos = torch.zeros(B, 1, C)
        h = h.permute(0, 2, 1).view(B, -1, N, 1)
        for conv, bn in zip(self.conv, self.bn):
            h = conv(h)
            h = bn(h)
            h = F.relu(h)
        h = torch.max(h[:, :, :, 0], 2)[0]  # [B,D]
        return new_pos, h


class TransitionDown(nn.Module):
    """
    The Transition Down Module
    """

    def __init__(self, n_points, batch_size, mlp_sizes, n_neighbors=64):
        super(TransitionDown, self).__init__()
        self.n_points = n_points
        self.frnn_graph = KNNGraphBuilder(n_neighbors)
        self.message = RelativePositionMessage(n_neighbors)
        self.conv = KNNConv(mlp_sizes, batch_size)
        self.batch_size = batch_size

    def forward(self, pos, feat):
        centroids = farthest_point_sampler(pos, self.n_points)
        g = self.frnn_graph(pos, centroids, feat)
        g.update_all(self.message, self.conv)

        mask = g.ndata['center'] == 1
        pos_dim = g.ndata['pos'].shape[-1]
        feat_dim = g.ndata['new_feat'].shape[-1]
        pos_res = g.ndata['pos'][mask].view(self.batch_size, -1, pos_dim)
        feat_res = g.ndata['new_feat'][mask].view(
            self.batch_size, -1, feat_dim)
        return pos_res, feat_res


# class TransitionUp(nn.Module):
#     def __init__(self, dim1, dim2, dim_out):
#         class SwapAxes(nn.Module):
#             def __init__(self):
#                 super().__init__()

#             def forward(self, x):
#                 return x.transpose(1, 2)

#         super().__init__()
#         self.fc1 = nn.Sequential(
#             nn.Linear(dim1, dim_out),
#             SwapAxes(),
#             nn.BatchNorm1d(dim_out),  # TODO
#             SwapAxes(),
#             nn.ReLU(),
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(dim2, dim_out),
#             SwapAxes(),
#             nn.BatchNorm1d(dim_out),  # TODO
#             SwapAxes(),
#             nn.ReLU(),
#         )
#         self.fp = KNNConv([], -1)

#     def forward(self, xyz1, points1, xyz2, points2):
#         feats1 = self.fc1(points1)
#         feats2 = self.fc2(points2)
#         feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(
#             1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
#         return feats1 + feats2
