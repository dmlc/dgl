import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Hardtanh, BatchNorm1d as BN
from torch.nn.modules.utils import _single
from torch.autograd import Function
from torch.nn import Parameter
import math
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
import numpy as np


activations = {
    'ReLU': ReLU,
    'Hardtanh': Hardtanh
}


class MeanShift(torch.nn.Module):

    def __init__(self, channels):
        super(MeanShift, self).__init__()
        self.register_buffer('median', torch.zeros((1, channels)))
        self.register_buffer("num_track", torch.LongTensor([0]))

    def forward(self, x):
        if self.training:
            median = torch.sort(x, dim=0)[0][x.shape[0] // 2].view(1, -1)
            self.median.mul_(self.num_track)
            self.median.add_(median)
            self.median.div_(self.num_track + 1)
            self.num_track.add_(1)
            self.median.detach_()
            self.num_track.detach_()
            x = x - self.median
        else:
            x = x - self.median
        return x


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


class BinaryQuantizeIdentity(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output
        return grad_input


class BinaryQuantizeIRNet(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None


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


class BiLinearXNOR(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, binary_act=True):
        super(BiLinearXNOR, self).__init__(in_features, out_features, bias=bias)
        self.binary_act = binary_act

    def forward(self, input):
        bw = self.weight
        ba = input
        bw = bw - bw.mean(-1).view(-1, 1)
        sw = bw.abs().mean(-1).view(-1, 1).detach()
        bw = BinaryQuantize().apply(bw)
        bw = bw * sw
        if self.binary_act:
            sa = ba.abs().mean(-1).view(-1, 1).detach()
            ba = BinaryQuantize().apply(ba)
            ba = ba * sa
        output = F.linear(ba, bw, self.bias)
        return output


class BiLinearBiReal(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, binary_act=True):
        super(BiLinearBiReal, self).__init__(in_features, out_features, bias=bias)
        self.binary_act = binary_act

    def forward(self, input):
        x = input
        out_forward = torch.sign(input)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        input = out_forward.detach() - out3.detach() + out3
        real_weights = self.weight
        scaling_factor = torch.mean(abs(real_weights),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        # y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)
        output = F.linear(input, binary_weights)
        return output


class BiLinearIRNet(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, binary_act=True):
        super(BiLinearIRNet, self).__init__(in_features, out_features, bias=bias)
        self.k = Parameter(torch.tensor([10], requires_grad=False).float().cuda())
        self.t = Parameter(torch.tensor([0.1], requires_grad=False).float().cuda())
        self.binary_act = binary_act

    def forward(self, input):
        bw = self.weight
        ba = input
        bw = bw - bw.mean(-1).view(-1, 1)
        bw = bw / bw.std(-1).view(-1, 1)
        sw = torch.pow(torch.tensor([2] * bw.size(0)).cuda().float(), (torch.log(bw.abs().mean(-1)) / math.log(2)).round().float()).view(-1, 1).detach()
        k, t = self.k, self.t
        bw = BinaryQuantizeIRNet().apply(bw, k, t)
        bw = bw * sw
        if self.binary_act:
            ba = BinaryQuantizeIRNet().apply(ba, k, t)
        output = F.linear(ba, bw, self.bias)
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


biLinears = {
    'BiLinear': BiLinear,
    'BiLinearXNOR': BiLinearXNOR,
    'BiLinearABC': BiLinearXNOR,
    'BiLinearIRNet': BiLinearIRNet,
    'BiLinearLSR': BiLinearLSR
}


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


def BiMLP(channels, batch_norm=True, activation=ReLU, BiLinear=BiLinear):
    return Seq(*[
        Seq(BiLinear(channels[i - 1], channels[i]), activation(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


def BwMLP(channels, batch_norm=True, activation='ReLU', bilinear='BiLinear'):
    return Seq(*[
        Seq(biLinears[bilinear](channels[i - 1], channels[i], binary_act=False), activations[activation](), BN(channels[i]))
        for i in range(1, len(channels))
    ])


def FirstBiMLP(channels, batch_norm=True, activation='ReLU', bilinear='BiLinear'):
    part1 = [Seq(Lin(channels[0], channels[1]), activations[activation](), BN(channels[1]))]
    part2 = [
        Seq(biLinears[bilinear](channels[i - 1], channels[i]), activations[activation](), BN(channels[i]))
        for i in range(2, len(channels))
    ]
    obj = part1 + part2
    return Seq(*obj)


class BiConv1d(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(BiConv1d, self).__init__(
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
            return F.conv1d(F.pad(ba, expanded_padding, mode='circular'),
                            bw, self.bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv1d(ba, bw, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class BiConv1dXNOR(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(BiConv1dXNOR, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)

    def forward(self, input):

        bw = self.weight
        ba = input
        bw = bw - bw.mean()
        sw = bw.abs().view(bw.size(0), bw.size(1), -1).mean(-1).view(bw.size(0), bw.size(1), 1).detach()
        sa = ba.abs().view(ba.size(0), ba.size(1), -1).mean(-1).view(ba.size(0), ba.size(1), 1).detach()
        bw = BinaryQuantize().apply(bw)
        ba = BinaryQuantize().apply(ba)
        bw = bw * sw
        ba = ba * sa

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv1d(F.pad(ba, expanded_padding, mode='circular'),
                            bw, self.bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv1d(ba, bw, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class BiConv1dLSR(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=False, padding_mode='zeros'):
        super(BiConv1dLSR, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)
        self.register_parameter('scale', None)

    def reset_scale(self, input):
        bw = self.weight
        ba = input
        bw = bw - bw.mean()
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            self.scale = Parameter((F.conv1d(F.pad(ba, expanded_padding, mode='circular'),\
                                  bw, self.bias, self.stride,\
                                  _single(0), self.dilation, self.groups).std() / \
                F.conv1d(torch.sign(F.pad(ba, expanded_padding, mode='circular')),\
                         torch.sign(bw), self.bias, self.stride,\
                         _single(0), self.dilation, self.groups).std()).float().to(ba.device))
        else:
            self.scale = Parameter((F.conv1d(ba, bw, self.bias, self.stride, self.padding, self.dilation, self.groups).std() \
                                   / F.conv1d(torch.sign(ba), torch.sign(bw), self.bias, self.stride, self.padding, self.dilation, self.groups).std()).float().to(ba.device))

    def forward(self, input):
        bw = self.weight
        ba = input
        bw = bw - bw.mean()
        if self.scale is None:
            self.reset_scale(input)
        bw = BinaryQuantize().apply(bw)
        ba = BinaryQuantize().apply(ba)
        bw = bw * self.scale

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv1d(F.pad(ba, expanded_padding, mode='circular'),
                            bw, self.bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv1d(ba, bw, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
