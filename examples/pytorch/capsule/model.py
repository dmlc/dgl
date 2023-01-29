import torch
from DGLDigitCapsule import DGLDigitCapsuleLayer
from DGLRoutingLayer import squash
from torch import nn


class Net(nn.Module):
    def __init__(self, device="cpu"):
        super(Net, self).__init__()
        self.device = device
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1),
            nn.ReLU(inplace=True),
        )

        self.primary = PrimaryCapsuleLayer(device=device)
        self.digits = DGLDigitCapsuleLayer(device=device)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_primary_caps = self.primary(out_conv1)
        out_digit_caps = self.digits(out_primary_caps)
        return out_digit_caps

    def margin_loss(self, input, target):
        batch_s = target.size(0)
        one_hot_vec = torch.zeros(batch_s, 10).to(self.device)
        for i in range(batch_s):
            one_hot_vec[i, target[i]] = 1.0
        batch_size = input.size(0)
        v_c = torch.sqrt((input**2).sum(dim=2, keepdim=True))
        zero = torch.zeros(1).to(self.device)
        m_plus = 0.9
        m_minus = 0.1
        loss_lambda = 0.5
        max_left = torch.max(m_plus - v_c, zero).view(batch_size, -1) ** 2
        max_right = torch.max(v_c - m_minus, zero).view(batch_size, -1) ** 2
        t_c = one_hot_vec
        l_c = t_c * max_left + loss_lambda * (1.0 - t_c) * max_right
        l_c = l_c.sum(dim=1)
        return l_c.mean()


class PrimaryCapsuleLayer(nn.Module):
    def __init__(self, in_channel=256, num_unit=8, device="cpu"):
        super(PrimaryCapsuleLayer, self).__init__()
        self.in_channel = in_channel
        self.num_unit = num_unit
        self.deivce = device
        self.conv_units = nn.ModuleList(
            [nn.Conv2d(self.in_channel, 32, 9, 2) for _ in range(self.num_unit)]
        )

    def forward(self, x):
        unit = [self.conv_units[i](x) for i, l in enumerate(self.conv_units)]
        unit = torch.stack(unit, dim=1)
        batch_size = x.size(0)
        unit = unit.view(batch_size, 8, -1)
        return squash(unit, dim=2)
