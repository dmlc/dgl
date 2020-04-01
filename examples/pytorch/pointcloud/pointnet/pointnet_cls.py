import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class PointNetCls(nn.Module):
    def __init__(self, output_classes, input_dims=3,
                 dropout_prob=0.5, use_transform=True):
        super(PointNetCls, self).__init__()
        self.input_dims = input_dims
        self.conv1 = nn.ModuleList()
        self.conv1.append(nn.Conv1d(input_dims, 64, 1))
        self.conv1.append(nn.Conv1d(64, 64, 1))
        self.conv1.append(nn.Conv1d(64, 64, 1))

        self.bn1 = nn.ModuleList()
        self.bn1.append(nn.BatchNorm1d(64))
        self.bn1.append(nn.BatchNorm1d(64))
        self.bn1.append(nn.BatchNorm1d(64))

        self.conv2 = nn.ModuleList()
        self.conv2.append(nn.Conv1d(64, 128, 1))
        self.conv2.append(nn.Conv1d(128, 1024, 1))

        self.bn2 = nn.ModuleList()
        self.bn2.append(nn.BatchNorm1d(128))
        self.bn2.append(nn.BatchNorm1d(1024))

        self.maxpool = nn.MaxPool1d(1024)
        self.pool_feat_len = 1024

        self.mlp3 = nn.ModuleList()
        self.mlp3.append(nn.Linear(1024, 512))
        self.mlp3.append(nn.Linear(512, 256))

        self.bn3 = nn.ModuleList()
        self.bn3.append(nn.BatchNorm1d(512))
        self.bn3.append(nn.BatchNorm1d(256))

        self.dropout = nn.Dropout(0.3)
        self.mlp_out = nn.Linear(256, output_classes)

        self.use_transform = use_transform
        if use_transform:
            self.transform1 = TransformNet(3)
            self.trans_bn1 = nn.BatchNorm1d(3)
            self.transform2 = TransformNet(64)
            self.trans_bn2 = nn.BatchNorm1d(64)

    def forward(self, g):
        batch_size = g.batch_size
        h = g.ndata['x'].view(batch_size, -1, self.input_dims).permute(0, 2, 1)
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
    def __init__(self, input_dims=3):
        super(TransformNet, self).__init__()
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv1d(input_dims, 64, 1))
        self.conv.append(nn.Conv1d(64, 128, 1))
        self.conv.append(nn.Conv1d(128, 1024, 1))

        self.bn = nn.ModuleList()
        self.bn.append(nn.BatchNorm1d(64))
        self.bn.append(nn.BatchNorm1d(128))
        self.bn.append(nn.BatchNorm1d(1024))

        self.maxpool = nn.MaxPool1d(1024)
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

        iden = Variable(torch.from_numpy(np.eye(self.input_dims).flatten().astype(np.float32)))
        iden = iden.view(1, self.input_dims * self.input_dims).repeat(batch_size, 1)
        if out.is_cuda:
            iden = iden.cuda()
        out = out + iden
        out = out.view(-1, self.input_dims, self.input_dims)
        return out

def compute_loss(logits, y, eps=0.2):
    num_classes = logits.shape[1]
    one_hot = torch.zeros_like(logits).scatter_(1, y.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (num_classes - 1)
    log_prob = F.log_softmax(logits, 1)
    loss = -(one_hot * log_prob).sum(1).mean()
    return loss

