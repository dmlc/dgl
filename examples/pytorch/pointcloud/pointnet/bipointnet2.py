import torch
import torch.nn as nn
from pointnet2 import *
from basic import *
import torch.nn.functional as F

class BiPointNet2SSGCls(nn.Module):
    def __init__(self, output_classes, batch_size, input_dims=3, dropout_prob=0.4):
        super(BiPointNet2SSGCls, self).__init__()
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