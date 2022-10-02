import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Hardtanh, BatchNorm1d as BN
from torch.nn.modules.utils import _single
from torch.autograd import Function
from torch.nn import Parameter
import math
import numpy as np
import dgl
import dgl.function as fn
from dgl.geometry import farthest_point_sampler # dgl.geometry.pytorch -> dgl.geometry


class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output
        grad_input[input[0].gt(1)] = 0
        grad_input[input[0].lt(-1)] = 0
        return grad_input


class BiLinearLSR(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=False, binary_act=True):
        super(BiLinearLSR, self).__init__(in_features, out_features, bias=bias)
        self.binary_act = binary_act

        # must register a nn.Parameter placeholder for model loading
        # self.register_parameter('scale', None) doesn't register None into state_dict
        # so it leads to unexpected key error when loading saved model
        # hence, init scale with Parameter
        # however, Parameter(None) actually has size [0], not [] as a scalar
        # hence, init it using the following trick
        self.register_parameter('scale', Parameter(torch.Tensor([0.0]).squeeze()))

    def reset_scale(self, input):
        bw = self.weight
        ba = input
        bw = bw - bw.mean()
        self.scale = Parameter((F.linear(ba, bw).std() / F.linear(torch.sign(ba), torch.sign(bw)).std()).float().to(ba.device))
        # corner case when ba is all 0.0
        if torch.isnan(self.scale):
            self.scale = Parameter((bw.std() / torch.sign(bw).std()).float().to(ba.device))

    def forward(self, input):
        bw = self.weight
        ba = input
        bw = bw - bw.mean()

        if self.scale.item() == 0.0:
            self.reset_scale(input)

        bw = BinaryQuantize().apply(bw)
        bw = bw * self.scale
        if self.binary_act:
            ba = BinaryQuantize().apply(ba)
        output = F.linear(ba, bw)
        return output


class BiLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, binary_act=True):
        super(BiLinear, self).__init__(in_features, out_features, bias=True)
        self.binary_act = binary_act
        self.output_ = None

    def forward(self, input):
        bw = self.weight
        ba = input
        bw = BinaryQuantize().apply(bw)
        if self.binary_act:
            ba = BinaryQuantize().apply(ba)
        output = F.linear(ba, bw, self.bias)
        self.output_ = output
        return output

class BiConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(BiConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)

    def forward(self, input):

        bw = self.weight
        ba = input
        bw = bw - bw.mean()
        bw = BinaryQuantize().apply(bw)
        ba = BinaryQuantize().apply(ba)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(ba, expanded_padding, mode='circular'),
                            bw, self.bias, self.stride,
                            _single(0), self.dilation, self.groups)    
        return F.conv2d(ba, bw, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

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
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

class FixedRadiusNearNeighbors(nn.Module):
    '''
    Ball Query - Find the neighbors with-in a fixed radius
    '''
    def __init__(self, radius, n_neighbor):
        super(FixedRadiusNearNeighbors, self).__init__()
        self.radius = radius
        self.n_neighbor = n_neighbor

    def forward(self, pos, centroids):
        '''
        Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
        '''
        device = pos.device
        B, N, _ = pos.shape
        center_pos = index_points(pos, centroids)
        _, S, _ = center_pos.shape
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        sqrdists = square_distance(center_pos, pos)
        group_idx[sqrdists > self.radius ** 2] = N
        group_idx = group_idx.sort(dim=-1)[0][:, :, :self.n_neighbor]
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, self.n_neighbor])
        mask = group_idx == N
        group_idx[mask] = group_first[mask]
        return group_idx

class FixedRadiusNNGraph(nn.Module):
    '''
    Build NN graph
    '''
    def __init__(self, radius, n_neighbor):
        super(FixedRadiusNNGraph, self).__init__()
        self.radius = radius
        self.n_neighbor = n_neighbor
        self.frnn = FixedRadiusNearNeighbors(radius, n_neighbor)

    def forward(self, pos, centroids, feat=None):
        dev = pos.device
        group_idx = self.frnn(pos, centroids)
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

class PointNetConv(nn.Module):
    '''
    Feature aggregation
    '''
    def __init__(self, sizes, batch_size):
        super(PointNetConv, self).__init__()
        self.batch_size = batch_size
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        for i in range(1, len(sizes)):
            self.conv.append(BiConv1d(sizes[i-1], sizes[i], 1))
            self.bn.append(nn.BatchNorm2d(sizes[i]))

    def forward(self, nodes):
        shape = nodes.mailbox['agg_feat'].shape
        h = nodes.mailbox['agg_feat'].view(self.batch_size, -1, shape[1], shape[2]).permute(0, 3, 2, 1)
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

class SAModule(nn.Module):
    """
    The Set Abstraction Layer
    """
    def __init__(self, npoints, batch_size, radius, mlp_sizes, n_neighbor=64,
                 group_all=False):
        super(SAModule, self).__init__()
        self.group_all = group_all
        if not group_all:
            self.npoints = npoints
            self.frnn_graph = FixedRadiusNNGraph(radius, n_neighbor)
        self.message = RelativePositionMessage(n_neighbor)
        self.conv = PointNetConv(mlp_sizes, batch_size)
        self.batch_size = batch_size

    def forward(self, pos, feat):
        if self.group_all:
            return self.conv.group_all(pos, feat)

        centroids = farthest_point_sampler(pos, self.npoints)
        g = self.frnn_graph(pos, centroids, feat)
        g.update_all(self.message, self.conv)

        mask = g.ndata['center'] == 1
        pos_dim = g.ndata['pos'].shape[-1]
        feat_dim = g.ndata['new_feat'].shape[-1]
        pos_res = g.ndata['pos'][mask].view(self.batch_size, -1, pos_dim)
        feat_res = g.ndata['new_feat'][mask].view(self.batch_size, -1, feat_dim)
        return pos_res, feat_res

class SAMSGModule(nn.Module):
    """
    The Set Abstraction Multi-Scale grouping Layer
    """
    def __init__(self, npoints, batch_size, radius_list, n_neighbor_list, mlp_sizes_list):
        super(SAMSGModule, self).__init__()
        self.batch_size = batch_size
        self.group_size = len(radius_list)

        self.npoints = npoints
        self.frnn_graph_list = nn.ModuleList()
        self.message_list = nn.ModuleList()
        self.conv_list = nn.ModuleList()
        for i in range(self.group_size):
            self.frnn_graph_list.append(FixedRadiusNNGraph(radius_list[i],
                                                           n_neighbor_list[i]))
            self.message_list.append(RelativePositionMessage(n_neighbor_list[i]))
            self.conv_list.append(PointNetConv(mlp_sizes_list[i], batch_size))

    def forward(self, pos, feat):
        centroids = farthest_point_sampler(pos, self.npoints)
        feat_res_list = []

        for i in range(self.group_size):
            g = self.frnn_graph_list[i](pos, centroids, feat)
            g.update_all(self.message_list[i], self.conv_list[i])
            mask = g.ndata['center'] == 1
            pos_dim = g.ndata['pos'].shape[-1]
            feat_dim = g.ndata['new_feat'].shape[-1]
            if i == 0:
                pos_res = g.ndata['pos'][mask].view(self.batch_size, -1, pos_dim)
            feat_res = g.ndata['new_feat'][mask].view(self.batch_size, -1, feat_dim)
            feat_res_list.append(feat_res)

        feat_res = torch.cat(feat_res_list, 2)
        return pos_res, feat_res

class PointNet2FP(nn.Module):
    """
    The Feature Propagation Layer
    """
    def __init__(self, input_dims, sizes):
        super(PointNet2FP, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        sizes = [input_dims] + sizes
        for i in range(1, len(sizes)):
            self.convs.append(nn.Conv1d(sizes[i-1], sizes[i], 1))
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
            interpolated_feat = torch.sum(index_points(feat2, idx) * weight.view(B, N, 3, 1), dim=2)

        if feat1 is not None:
            new_feat = torch.cat([feat1, interpolated_feat], dim=-1)
        else:
            new_feat = interpolated_feat

        new_feat = new_feat.permute(0, 2, 1)  # [B, D, S]
        for i, conv in enumerate(self.convs):
            bn = self.bns[i]
            new_feat = F.relu(bn(conv(new_feat)))
        return new_feat


class PointNet2SSGCls(nn.Module):
    def __init__(self, output_classes, batch_size, input_dims=3, dropout_prob=0.4):
        super(PointNet2SSGCls, self).__init__()
        self.input_dims = input_dims

        self.sa_module1 = SAModule(512, batch_size, 0.2, [input_dims, 64, 64, 128])
        self.sa_module2 = SAModule(128, batch_size, 0.4, [128 + 3, 128, 128, 256])
        self.sa_module3 = SAModule(None, batch_size, None, [256 + 3, 256, 512, 1024],
                                   group_all=True)

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

class PointNet2MSGCls(nn.Module):
    def __init__(self, output_classes, batch_size, input_dims=3, dropout_prob=0.4):
        super(PointNet2MSGCls, self).__init__()
        self.input_dims = input_dims

        self.sa_msg_module1 = SAMSGModule(512, batch_size, [0.1, 0.2, 0.4], [16, 32, 128],
                                          [[input_dims, 32, 32, 64], [input_dims, 64, 64, 128],
                                           [input_dims, 64, 96, 128]])
        self.sa_msg_module2 = SAMSGModule(128, batch_size, [0.2, 0.4, 0.8], [32, 64, 128],
                                          [[320 + 3, 64, 64, 128], [320 + 3, 128, 128, 256],
                                           [320 + 3, 128, 128, 256]])
        self.sa_module3 = SAModule(None, batch_size, None, [640 + 3, 256, 512, 1024],
                                   group_all=True)

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
        pos, feat = self.sa_msg_module1(pos, feat)
        pos, feat = self.sa_msg_module2(pos, feat)
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
