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


class RelativePositionMessage(nn.Module):
    """
    Compute the input feature from neighbors
    """

    def __init__(self, n_neighbor):
        super(RelativePositionMessage, self).__init__()
        self.n_neighbor = n_neighbor

    def forward(self, edges):
        pos = edges.src["pos"] - edges.dst["pos"]
        if "feat" in edges.src:
            res = torch.cat([pos, edges.src["feat"]], 1)
        else:
            res = pos
        return {"agg_feat": res}


class KNNConv(nn.Module):
    """
    Feature aggregation
    """

    def __init__(self, sizes, batch_size):
        super(KNNConv, self).__init__()
        self.batch_size = batch_size
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        for i in range(1, len(sizes)):
            self.conv.append(nn.Conv2d(sizes[i - 1], sizes[i], 1))
            self.bn.append(nn.BatchNorm2d(sizes[i]))

    def forward(self, nodes):
        shape = nodes.mailbox["agg_feat"].shape
        h = (
            nodes.mailbox["agg_feat"]
            .view(self.batch_size, -1, shape[1], shape[2])
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

    def group_all(self, pos, feat):
        """
        Feature aggregation and pooling for the non-sampling layer
        """
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

        mask = g.ndata["center"] == 1
        pos_dim = g.ndata["pos"].shape[-1]
        feat_dim = g.ndata["new_feat"].shape[-1]
        pos_res = g.ndata["pos"][mask].view(self.batch_size, -1, pos_dim)
        feat_res = g.ndata["new_feat"][mask].view(self.batch_size, -1, feat_dim)
        return pos_res, feat_res


class FeaturePropagation(nn.Module):
    """
    The FeaturePropagation Layer
    """

    def __init__(self, input_dims, sizes):
        super(FeaturePropagation, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        sizes = [input_dims] + sizes
        for i in range(1, len(sizes)):
            self.convs.append(nn.Conv1d(sizes[i - 1], sizes[i], 1))
            self.bns.append(nn.BatchNorm1d(sizes[i]))

    def forward(self, x1, x2, feat1, feat2):
        """
        Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
            Input:
                x1: input points position data, [B, N, C]
                x2: sampled input points position data, [B, S, C]
                feat1: input points data, [B, N, D]
                feat2: input points data, [B, S, D]
            Return:
                new_feat: upsampled points data, [B, D', N]
        """
        B, N, C = x1.shape
        _, S, _ = x2.shape

        if S == 1:
            interpolated_feat = feat2.repeat(1, N, 1)
        else:
            dists = square_distance(x1, x2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feat = torch.sum(
                index_points(feat2, idx) * weight.view(B, N, 3, 1), dim=2
            )

        if feat1 is not None:
            new_feat = torch.cat([feat1, interpolated_feat], dim=-1)
        else:
            new_feat = interpolated_feat

        new_feat = new_feat.permute(0, 2, 1)  # [B, D, S]
        for i, conv in enumerate(self.convs):
            bn = self.bns[i]
            new_feat = F.relu(bn(conv(new_feat)))
        return new_feat


class SwapAxes(nn.Module):
    def __init__(self, dim1=1, dim2=2):
        super(SwapAxes, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class TransitionUp(nn.Module):
    """
    The Transition Up Module
    """

    def __init__(self, dim1, dim2, dim_out):
        super(TransitionUp, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = FeaturePropagation(-1, [])

    def forward(self, pos1, feat1, pos2, feat2):
        h1 = self.fc1(feat1)
        h2 = self.fc2(feat2)
        h1 = self.fp(pos2, pos1, None, h1).transpose(1, 2)
        return h1 + h2
