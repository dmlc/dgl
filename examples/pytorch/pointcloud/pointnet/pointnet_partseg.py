import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PointNetPartSeg(nn.Module):
    def __init__(
        self, output_classes, input_dims=3, num_points=2048, use_transform=True
    ):
        super(PointNetPartSeg, self).__init__()
        self.input_dims = input_dims

        self.conv1 = nn.ModuleList()
        self.conv1.append(nn.Conv1d(input_dims, 64, 1))
        self.conv1.append(nn.Conv1d(64, 128, 1))
        self.conv1.append(nn.Conv1d(128, 128, 1))

        self.bn1 = nn.ModuleList()
        self.bn1.append(nn.BatchNorm1d(64))
        self.bn1.append(nn.BatchNorm1d(128))
        self.bn1.append(nn.BatchNorm1d(128))

        self.conv2 = nn.ModuleList()
        self.conv2.append(nn.Conv1d(128, 512, 1))

        self.bn2 = nn.ModuleList()
        self.bn2.append(nn.BatchNorm1d(512))

        self.conv_max = nn.Conv1d(512, 2048, 1)
        self.bn_max = nn.BatchNorm1d(2048)

        self.maxpool = nn.MaxPool1d(num_points)
        self.pool_feat_len = 2048

        self.conv3 = nn.ModuleList()
        self.conv3.append(nn.Conv1d(2048 + 64 + 128 * 3 + 512 + 16, 256, 1))
        self.conv3.append(nn.Conv1d(256, 256, 1))
        self.conv3.append(nn.Conv1d(256, 128, 1))

        self.bn3 = nn.ModuleList()
        self.bn3.append(nn.BatchNorm1d(256))
        self.bn3.append(nn.BatchNorm1d(256))
        self.bn3.append(nn.BatchNorm1d(128))

        self.conv_out = nn.Conv1d(128, output_classes, 1)

        self.use_transform = use_transform
        if use_transform:
            self.transform1 = TransformNet(self.input_dims)
            self.trans_bn1 = nn.BatchNorm1d(self.input_dims)
            self.transform2 = TransformNet(128)
            self.trans_bn2 = nn.BatchNorm1d(128)

    def forward(self, x, cat_vec=None):
        batch_size = x.shape[0]
        h = x.permute(0, 2, 1)
        num_points = h.shape[2]
        if self.use_transform:
            trans = self.transform1(h)
            h = h.transpose(2, 1)
            h = torch.bmm(h, trans)
            h = h.transpose(2, 1)
            h = F.relu(self.trans_bn1(h))

        mid_feat = []
        for conv, bn in zip(self.conv1, self.bn1):
            h = conv(h)
            h = bn(h)
            h = F.relu(h)
            mid_feat.append(h)

        if self.use_transform:
            trans = self.transform2(h)
            h = h.transpose(2, 1)
            h = torch.bmm(h, trans)
            h = h.transpose(2, 1)
            h = F.relu(self.trans_bn2(h))
            mid_feat.append(h)

        for conv, bn in zip(self.conv2, self.bn2):
            h = conv(h)
            h = bn(h)
            h = F.relu(h)
            mid_feat.append(h)

        h = self.conv_max(h)
        h = self.bn_max(h)
        h = self.maxpool(h).view(batch_size, -1, 1).repeat(1, 1, num_points)
        mid_feat.append(h)
        if cat_vec is not None:
            mid_feat.append(cat_vec)
        h = torch.cat(mid_feat, 1)
        for conv, bn in zip(self.conv3, self.bn3):
            h = conv(h)
            h = bn(h)
            h = F.relu(h)

        out = self.conv_out(h)
        return out


class TransformNet(nn.Module):
    def __init__(self, input_dims=3, num_points=2048):
        super(TransformNet, self).__init__()
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv1d(input_dims, 64, 1))
        self.conv.append(nn.Conv1d(64, 128, 1))
        self.conv.append(nn.Conv1d(128, 1024, 1))

        self.bn = nn.ModuleList()
        self.bn.append(nn.BatchNorm1d(64))
        self.bn.append(nn.BatchNorm1d(128))
        self.bn.append(nn.BatchNorm1d(1024))

        self.maxpool = nn.MaxPool1d(num_points)
        self.pool_feat_len = 1024

        self.mlp2 = nn.ModuleList()
        self.mlp2.append(nn.Linear(1024, 512))
        self.mlp2.append(nn.Linear(512, 256))

        self.bn2 = nn.ModuleList()
        self.bn2.append(nn.BatchNorm1d(512))
        self.bn2.append(nn.BatchNorm1d(256))

        self.input_dims = input_dims
        self.mlp_out = nn.Linear(256, input_dims * input_dims)

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


class PartSegLoss(nn.Module):
    def __init__(self, eps=0.2):
        super(PartSegLoss, self).__init__()
        self.eps = eps
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, y):
        num_classes = logits.shape[1]
        logits = logits.permute(0, 2, 1).contiguous().view(-1, num_classes)
        loss = self.loss(logits, y)
        return loss
