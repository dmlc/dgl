import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from basic import BiLinear
from torch.autograd import Variable

offset_map = {1024: -3.2041, 2048: -3.4025, 4096: -3.5836}


class Conv1d(nn.Module):
    def __init__(self, inplane, outplane, Linear):
        super().__init__()
        self.lin = Linear(inplane, outplane)

    def forward(self, x):
        B, C, N = x.shape
        x = x.permute(0, 2, 1).contiguous().view(-1, C)
        x = self.lin(x).view(B, N, -1).permute(0, 2, 1).contiguous()
        return x


class EmaMaxPool(nn.Module):
    def __init__(self, kernel_size, affine=True, Linear=BiLinear, use_bn=True):
        super(EmaMaxPool, self).__init__()
        self.kernel_size = kernel_size
        self.bn3 = nn.BatchNorm1d(1024, affine=affine)
        self.use_bn = use_bn

    def forward(self, x):
        batchsize, D, N = x.size()
        if self.use_bn:
            x = torch.max(x, 2, keepdim=True)[0] + offset_map[N]
        else:
            x = torch.max(x, 2, keepdim=True)[0] - 0.3
        return x


class BiPointNetCls(nn.Module):
    def __init__(
        self,
        output_classes,
        input_dims=3,
        conv1_dim=64,
        use_transform=True,
        Linear=BiLinear,
    ):
        super(BiPointNetCls, self).__init__()
        self.input_dims = input_dims
        self.conv1 = nn.ModuleList()
        self.conv1.append(Conv1d(input_dims, conv1_dim, Linear=Linear))
        self.conv1.append(Conv1d(conv1_dim, conv1_dim, Linear=Linear))
        self.conv1.append(Conv1d(conv1_dim, conv1_dim, Linear=Linear))

        self.bn1 = nn.ModuleList()
        self.bn1.append(nn.BatchNorm1d(conv1_dim))
        self.bn1.append(nn.BatchNorm1d(conv1_dim))
        self.bn1.append(nn.BatchNorm1d(conv1_dim))

        self.conv2 = nn.ModuleList()
        self.conv2.append(Conv1d(conv1_dim, conv1_dim * 2, Linear=Linear))
        self.conv2.append(Conv1d(conv1_dim * 2, conv1_dim * 16, Linear=Linear))

        self.bn2 = nn.ModuleList()
        self.bn2.append(nn.BatchNorm1d(conv1_dim * 2))
        self.bn2.append(nn.BatchNorm1d(conv1_dim * 16))

        self.maxpool = EmaMaxPool(conv1_dim * 16, Linear=Linear, use_bn=True)
        self.pool_feat_len = conv1_dim * 16

        self.mlp3 = nn.ModuleList()
        self.mlp3.append(Linear(conv1_dim * 16, conv1_dim * 8))
        self.mlp3.append(Linear(conv1_dim * 8, conv1_dim * 4))

        self.bn3 = nn.ModuleList()
        self.bn3.append(nn.BatchNorm1d(conv1_dim * 8))
        self.bn3.append(nn.BatchNorm1d(conv1_dim * 4))

        self.dropout = nn.Dropout(0.3)
        self.mlp_out = Linear(conv1_dim * 4, output_classes)

        self.use_transform = use_transform
        if use_transform:
            self.transform1 = TransformNet(input_dims)
            self.trans_bn1 = nn.BatchNorm1d(input_dims)
            self.transform2 = TransformNet(conv1_dim)
            self.trans_bn2 = nn.BatchNorm1d(conv1_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        h = x.permute(0, 2, 1)
        if self.use_transform:
            trans = self.transform1(h)
            h = h.transpose(2, 1)
            h = torch.bmm(h, trans)
            h = h.transpose(2, 1)
            h = F.relu(self.trans_bn1(h))

        for conv, bn in zip(self.conv1, self.bn1):
            h = conv(h)
            h = bn(h)
            h = F.relu(h)

        if self.use_transform:
            trans = self.transform2(h)
            h = h.transpose(2, 1)
            h = torch.bmm(h, trans)
            h = h.transpose(2, 1)
            h = F.relu(self.trans_bn2(h))

        for conv, bn in zip(self.conv2, self.bn2):
            h = conv(h)
            h = bn(h)
            h = F.relu(h)

        h = self.maxpool(h).view(-1, self.pool_feat_len)
        for mlp, bn in zip(self.mlp3, self.bn3):
            h = mlp(h)
            h = bn(h)
            h = F.relu(h)

        h = self.dropout(h)
        out = self.mlp_out(h)
        return out


class TransformNet(nn.Module):
    def __init__(self, input_dims=3, conv1_dim=64, Linear=BiLinear):
        super(TransformNet, self).__init__()
        self.conv = nn.ModuleList()
        self.conv.append(Conv1d(input_dims, conv1_dim, Linear=Linear))
        self.conv.append(Conv1d(conv1_dim, conv1_dim * 2, Linear=Linear))
        self.conv.append(Conv1d(conv1_dim * 2, conv1_dim * 16, Linear=Linear))

        self.bn = nn.ModuleList()
        self.bn.append(nn.BatchNorm1d(conv1_dim))
        self.bn.append(nn.BatchNorm1d(conv1_dim * 2))
        self.bn.append(nn.BatchNorm1d(conv1_dim * 16))

        # self.maxpool = nn.MaxPool1d(conv1_dim * 16)
        self.maxpool = EmaMaxPool(conv1_dim * 16, Linear=Linear, use_bn=True)
        self.pool_feat_len = conv1_dim * 16

        self.mlp2 = nn.ModuleList()
        self.mlp2.append(Linear(conv1_dim * 16, conv1_dim * 8))
        self.mlp2.append(Linear(conv1_dim * 8, conv1_dim * 4))

        self.bn2 = nn.ModuleList()
        self.bn2.append(nn.BatchNorm1d(conv1_dim * 8))
        self.bn2.append(nn.BatchNorm1d(conv1_dim * 4))

        self.input_dims = input_dims
        self.mlp_out = Linear(conv1_dim * 4, input_dims * input_dims)

    def forward(self, h):
        batch_size = h.shape[0]
        for conv, bn in zip(self.conv, self.bn):
            h = conv(h)
            h = bn(h)
            h = F.relu(h)

        h = self.maxpool(h).view(-1, self.pool_feat_len)
        for mlp, bn in zip(self.mlp2, self.bn2):
            h = mlp(h)
            h = bn(h)
            h = F.relu(h)

        out = self.mlp_out(h)

        iden = Variable(
            torch.from_numpy(
                np.eye(self.input_dims).flatten().astype(np.float32)
            )
        )
        iden = iden.view(1, self.input_dims * self.input_dims).repeat(
            batch_size, 1
        )
        if out.is_cuda:
            iden = iden.cuda()
        out = out + iden
        out = out.view(-1, self.input_dims, self.input_dims)
        return out
