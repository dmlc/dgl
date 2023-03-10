import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import Parameter
from torch.nn.modules.utils import _single


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
        self.register_parameter(
            "scale", Parameter(torch.Tensor([0.0]).squeeze())
        )

    def reset_scale(self, input):
        bw = self.weight
        ba = input
        bw = bw - bw.mean()
        self.scale = Parameter(
            (
                F.linear(ba, bw).std()
                / F.linear(torch.sign(ba), torch.sign(bw)).std()
            )
            .float()
            .to(ba.device)
        )
        # corner case when ba is all 0.0
        if torch.isnan(self.scale):
            self.scale = Parameter(
                (bw.std() / torch.sign(bw).std()).float().to(ba.device)
            )

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
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super(BiConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )

    def forward(self, input):
        bw = self.weight
        ba = input
        bw = bw - bw.mean()
        bw = BinaryQuantize().apply(bw)
        ba = BinaryQuantize().apply(ba)

        if self.padding_mode == "circular":
            expanded_padding = (
                (self.padding[0] + 1) // 2,
                self.padding[0] // 2,
            )
            return F.conv2d(
                F.pad(ba, expanded_padding, mode="circular"),
                bw,
                self.bias,
                self.stride,
                _single(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            ba,
            bw,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


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


class FixedRadiusNearNeighbors(nn.Module):
    """
    Ball Query - Find the neighbors with-in a fixed radius
    """

    def __init__(self, radius, n_neighbor):
        super(FixedRadiusNearNeighbors, self).__init__()
        self.radius = radius
        self.n_neighbor = n_neighbor

    def forward(self, pos, centroids):
        """
        Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
        """
        device = pos.device
        B, N, _ = pos.shape
        center_pos = index_points(pos, centroids)
        _, S, _ = center_pos.shape
        group_idx = (
            torch.arange(N, dtype=torch.long)
            .to(device)
            .view(1, 1, N)
            .repeat([B, S, 1])
        )
        sqrdists = square_distance(center_pos, pos)
        group_idx[sqrdists > self.radius**2] = N
        group_idx = group_idx.sort(dim=-1)[0][:, :, : self.n_neighbor]
        group_first = (
            group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, self.n_neighbor])
        )
        mask = group_idx == N
        group_idx[mask] = group_first[mask]
        return group_idx


class FixedRadiusNNGraph(nn.Module):
    """
    Build NN graph
    """

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
