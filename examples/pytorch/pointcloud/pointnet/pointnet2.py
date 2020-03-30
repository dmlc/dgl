import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

class FarthestPointSampler(nn.Module):
    def __init__(self, npoints):
        super(FarthestPointSampler, self).__init__()
        self.npoints = npoints

    def forward(self, g):
        bs = g.batch_size
        coord = g.ndata['x'].view(bs, -1, 3)
        device = coord.device
        B, N, C =coord.shape
        centroids = torch.zeros(B, self.npoints, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(self.npoints):
            centroids[:, i] = farthest
            centroid = coord[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((coord - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        for i in range(B):
            centroids[i,:] += i*N
        g.ndata['sampled'][centroids.view(-1)] = 1
        return

class EpsBallPoints(nn.Module):
    def __init__(self, radius, nsample):
        super(EpsBallPoints, self).__init__()
        self.radius = radius
        self.nsample = nsample

    def forward(self, g):
        bs = g.batch_size
        coord = g.ndata['x'].view(bs, -1, 3)
        samples = g.ndata['x'][g.ndata['sampled'][:,0] == 1].view(bs, -1, 3)
        device = coord.device
        B, N, C = coord.shape
        _, S, _ = samples.shape
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        sqrdists = square_distance(samples, coord)
        import pdb; pdb.set_trace()
        group_idx[sqrdists > self.radius ** 2] = N
        group_idx = group_idx.sort(dim=-1)[0][:, :, :self.nsample]
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, self.nsample])
        mask = group_idx == N
        group_idx[mask] = group_first[mask]
        return group_idx

class SAMLP(nn.Module):
    def __init__(self, sizes, maxpool=False):
        super(SAMLP, self).__init__()
        self.mlp = nn.ModuleList()
        self.bn = nn.ModuleList()
        for i in range(1, len(sizes)):
            self.mlp.append(nn.Conv1d(sizes[i-1], sizes[i], 1))
            self.bn.append(nn.BatchNorm1d(sizes[i]))
        if maxpool:
            self.maxpool = nn.MaxPool1d(sizes[-1])
        else:
            self.maxpool = None
    
    def forward(self, g):
        batch_size = g.batch_size
        h = g.ndata['x'].view(batch_size, -1, self.input_dims).permute(0, 2, 1)
        for mlp, bn in zip(self.mlp, self.bn):
            h = mlp(h)
            h = bn(h)
            h = F.relu(h)
        if self.maxpool:
            h = self.maxpool(h)
        h = h.view(g.number_of_nodes, -1)
        g.ndata['x'] = torch.cat(g.ndata['x'], h)

class SAModule(nn.Module):
    def __init__(self, npoints, radius, mlp_sizes):
        super(SAModule, self).__init__()
        self.fps = FarthestPointSampler(npoints)
        self.epsball = EpsBallPoints(radius, 64)
        self.mlp = SAMLP(mlp_sizes)
    
    def forward(self, g):
        self.fps(g)
        self.epsball(g)
        self.mlp(g)

class PointNet2(nn.Module):
    def __init__(self, output_classes, input_dims=3,
                 dropout_prob=0.5, use_transform=True):
        super(PointNet2, self).__init__()
        self.input_dims = input_dims

        self.sa_module1 = SAModule(512, 0.2, [3, 64, 64, 128])
        self.sa_module2 = SAModule(128, 0.4, [128 + 3, 128, 128, 256])
        self.sa_module3 = SAMLP([256 + 3, 256, 512, 1024], maxpool=True)

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
        self.sa_module1(g)
        self.sa_module2(g)
        self.sa_module3(g)

        h = self.dropout(h)
        out = self.mlp_out(h)
        return out
