import torch
import torch.nn as nn
import torch.nn.functional as F
from basic import (
    BiConv2d,
    BiLinearLSR,
    FixedRadiusNNGraph,
    RelativePositionMessage,
)
from dgl.geometry import farthest_point_sampler


class BiPointNetConv(nn.Module):
    """
    Feature aggregation
    """

    def __init__(self, sizes, batch_size):
        super(BiPointNetConv, self).__init__()
        self.batch_size = batch_size
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        for i in range(1, len(sizes)):
            self.conv.append(BiConv2d(sizes[i - 1], sizes[i], 1))
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


class BiSAModule(nn.Module):
    """
    The Set Abstraction Layer
    """

    def __init__(
        self,
        npoints,
        batch_size,
        radius,
        mlp_sizes,
        n_neighbor=64,
        group_all=False,
    ):
        super(BiSAModule, self).__init__()
        self.group_all = group_all
        if not group_all:
            self.npoints = npoints
            self.frnn_graph = FixedRadiusNNGraph(radius, n_neighbor)
        self.message = RelativePositionMessage(n_neighbor)
        self.conv = BiPointNetConv(mlp_sizes, batch_size)
        self.batch_size = batch_size

    def forward(self, pos, feat):
        if self.group_all:
            return self.conv.group_all(pos, feat)

        centroids = farthest_point_sampler(pos, self.npoints)
        g = self.frnn_graph(pos, centroids, feat)
        g.update_all(self.message, self.conv)

        mask = g.ndata["center"] == 1
        pos_dim = g.ndata["pos"].shape[-1]
        feat_dim = g.ndata["new_feat"].shape[-1]
        pos_res = g.ndata["pos"][mask].view(self.batch_size, -1, pos_dim)
        feat_res = g.ndata["new_feat"][mask].view(self.batch_size, -1, feat_dim)
        return pos_res, feat_res


class BiPointNet2SSGCls(nn.Module):
    def __init__(
        self, output_classes, batch_size, input_dims=3, dropout_prob=0.4
    ):
        super(BiPointNet2SSGCls, self).__init__()
        self.input_dims = input_dims

        self.sa_module1 = BiSAModule(
            512, batch_size, 0.2, [input_dims, 64, 64, 128]
        )
        self.sa_module2 = BiSAModule(
            128, batch_size, 0.4, [128 + 3, 128, 128, 256]
        )
        self.sa_module3 = BiSAModule(
            None, batch_size, None, [256 + 3, 256, 512, 1024], group_all=True
        )

        self.mlp1 = BiLinearLSR(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(dropout_prob)

        self.mlp2 = BiLinearLSR(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(dropout_prob)

        self.mlp_out = BiLinearLSR(256, output_classes)

    def forward(self, x):
        if x.shape[-1] > 3:
            pos = x[:, :, :3]
            feat = x[:, :, 3:]
        else:
            pos = x
            feat = None
        pos, feat = self.sa_module1(pos, feat)
        pos, feat = self.sa_module2(pos, feat)
        _, h = self.sa_module3(pos, feat)

        h = self.mlp1(h)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.drop1(h)
        h = self.mlp2(h)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.drop2(h)

        out = self.mlp_out(h)
        return out
