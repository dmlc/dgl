import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.geometry import farthest_point_sampler

"""
Part of the code are adapted from
https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""


def square_distance(src, dst):
    """
    Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


class KNearNeighbors(nn.Module):
    """
    Find the k nearest neighbors
    """

    def __init__(self, n_neighbor):
        super(KNearNeighbors, self).__init__()
        self.n_neighbor = n_neighbor

    def forward(self, pos, centroids):
        """
        Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
        """
        center_pos = index_points(pos, centroids)
        sqrdists = square_distance(center_pos, pos)
        group_idx = sqrdists.argsort(dim=-1)[:, :, : self.n_neighbor]
        return group_idx


class KNNGraphBuilder(nn.Module):
    """
    Build NN graph
    """

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
            dst = (
                centroids[i]
                .view(-1, 1)
                .repeat(
                    1, min(self.n_neighbor, src.shape[0] // centroids.shape[1])
                )
                .view(-1)
            )

            unified = torch.cat([src, dst])
            uniq, inv_idx = torch.unique(unified, return_inverse=True)
            src_idx = inv_idx[: src.shape[0]]
            dst_idx = inv_idx[src.shape[0] :]

            g = dgl.graph((src_idx, dst_idx))
            g.ndata["pos"] = pos[i][uniq]
            g.ndata["center"] = center[uniq]
            if feat is not None:
                g.ndata["feat"] = feat[i][uniq]
            glist.append(g)
        bg = dgl.batch(glist)
        return bg


class KNNMessage(nn.Module):
    """
    Compute the input feature from neighbors
    """

    def __init__(self, n_neighbor):
        super(KNNMessage, self).__init__()
        self.n_neighbor = n_neighbor

    def forward(self, edges):
        norm = edges.src["feat"] - edges.dst["feat"]
        if "feat" in edges.src:
            res = torch.cat([norm, edges.src["feat"]], 1)
        else:
            res = norm
        return {"agg_feat": res}


class KNNConv(nn.Module):
    """
    Feature aggregation
    """

    def __init__(self, sizes):
        super(KNNConv, self).__init__()
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        for i in range(1, len(sizes)):
            self.conv.append(nn.Conv2d(sizes[i - 1], sizes[i], 1))
            self.bn.append(nn.BatchNorm2d(sizes[i]))

    def forward(self, nodes):
        shape = nodes.mailbox["agg_feat"].shape
        h = (
            nodes.mailbox["agg_feat"]
            .view(shape[0], -1, shape[1], shape[2])
            .permute(0, 3, 2, 1)
        )
        for conv, bn in zip(self.conv, self.bn):
            h = conv(h)
            h = bn(h)
            h = F.relu(h)
        h = torch.max(h, 2)[0]
        feat_dim = h.shape[1]
        h = h.permute(0, 2, 1).reshape(-1, feat_dim)
        return {"new_feat": h}


class TransitionDown(nn.Module):
    """
    The Transition Down Module
    """

    def __init__(self, in_channels, out_channels, n_neighbor=64):
        super(TransitionDown, self).__init__()
        self.frnn_graph = KNNGraphBuilder(n_neighbor)
        self.message = KNNMessage(n_neighbor)
        self.conv = KNNConv([in_channels, out_channels, out_channels])

    def forward(self, pos, feat, n_point):
        batch_size = pos.shape[0]
        centroids = farthest_point_sampler(pos, n_point)
        g = self.frnn_graph(pos, centroids, feat)
        g.update_all(self.message, self.conv)

        mask = g.ndata["center"] == 1
        pos_dim = g.ndata["pos"].shape[-1]
        feat_dim = g.ndata["new_feat"].shape[-1]
        pos_res = g.ndata["pos"][mask].view(batch_size, -1, pos_dim)
        feat_res = g.ndata["new_feat"][mask].view(batch_size, -1, feat_dim)
        return pos_res, feat_res
