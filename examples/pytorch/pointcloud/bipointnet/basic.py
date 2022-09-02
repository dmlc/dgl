import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Hardtanh, BatchNorm1d as BN
from torch.nn.modules.utils import _single
from torch.autograd import Function
from torch.nn import Parameter
import math


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

