import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2 import PointNet2FP, SAModule, SAMSGModule
from torch.autograd import Variable


class PointNet2SSGPartSeg(nn.Module):
    def __init__(self, output_classes, batch_size, input_dims=6):
        super(PointNet2SSGPartSeg, self).__init__()
        # if normal_channel == true, input_dims = 6+3
        self.input_dims = input_dims

        self.sa_module1 = SAModule(
            512, batch_size, 0.2, [input_dims, 64, 64, 128], n_neighbor=32
        )
        self.sa_module2 = SAModule(
            128, batch_size, 0.4, [128 + 3, 128, 128, 256]
        )
        self.sa_module3 = SAModule(
            None, batch_size, None, [256 + 3, 256, 512, 1024], group_all=True
        )

        self.fp3 = PointNet2FP(1280, [256, 256])
        self.fp2 = PointNet2FP(384, [256, 128])
        # if normal_channel == true, 128+16+6+3
        self.fp1 = PointNet2FP(128 + 16 + 6, [128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, output_classes, 1)

    def forward(self, x, cat_vec=None):
        if x.shape[-1] > 3:
            l0_pos = x[:, :, :3]
            l0_feat = x
        else:
            l0_pos = x
            l0_feat = x
        # Set Abstraction layers
        l1_pos, l1_feat = self.sa_module1(l0_pos, l0_feat)  # l1_feat: [B, N, D]
        l2_pos, l2_feat = self.sa_module2(l1_pos, l1_feat)
        l3_pos, l3_feat = self.sa_module3(l2_pos, l2_feat)  # [B, N, C], [B, D]
        # Feature Propagation layers
        l2_feat = self.fp3(
            l2_pos, l3_pos, l2_feat, l3_feat.unsqueeze(1)
        )  # l2_feat: [B, D, N]
        l1_feat = self.fp2(l1_pos, l2_pos, l1_feat, l2_feat.permute(0, 2, 1))
        l0_feat = torch.cat([cat_vec.permute(0, 2, 1), l0_pos, l0_feat], 2)
        l0_feat = self.fp1(l0_pos, l1_pos, l0_feat, l1_feat.permute(0, 2, 1))
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_feat)))
        out = self.drop1(feat)
        out = self.conv2(out)  # [B, output_classes, N]
        return out


class PointNet2MSGPartSeg(nn.Module):
    def __init__(self, output_classes, batch_size, input_dims=6):
        super(PointNet2MSGPartSeg, self).__init__()

        self.sa_msg_module1 = SAMSGModule(
            512,
            batch_size,
            [0.1, 0.2, 0.4],
            [32, 64, 128],
            [
                [input_dims, 32, 32, 64],
                [input_dims, 64, 64, 128],
                [input_dims, 64, 96, 128],
            ],
        )
        self.sa_msg_module2 = SAMSGModule(
            128,
            batch_size,
            [0.4, 0.8],
            [64, 128],
            [
                [128 + 128 + 64 + 3, 128, 128, 256],
                [128 + 128 + 64 + 3, 128, 196, 256],
            ],
        )
        self.sa_module3 = SAModule(
            None, batch_size, None, [512 + 3, 256, 512, 1024], group_all=True
        )

        self.fp3 = PointNet2FP(1536, [256, 256])
        self.fp2 = PointNet2FP(576, [256, 128])
        # if normal_channel == true, 150 + 3
        self.fp1 = PointNet2FP(150, [128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, output_classes, 1)

    def forward(self, x, cat_vec=None):
        if x.shape[-1] > 3:
            l0_pos = x[:, :, :3]
            l0_feat = x
        else:
            l0_pos = x
            l0_feat = x
        # Set Abstraction layers
        l1_pos, l1_feat = self.sa_msg_module1(l0_pos, l0_feat)
        l2_pos, l2_feat = self.sa_msg_module2(l1_pos, l1_feat)
        l3_pos, l3_feat = self.sa_module3(l2_pos, l2_feat)
        # Feature Propagation layers
        l2_feat = self.fp3(l2_pos, l3_pos, l2_feat, l3_feat.unsqueeze(1))
        l1_feat = self.fp2(l1_pos, l2_pos, l1_feat, l2_feat.permute(0, 2, 1))
        l0_feat = torch.cat([cat_vec.permute(0, 2, 1), l0_pos, l0_feat], 2)
        l0_feat = self.fp1(l0_pos, l1_pos, l0_feat, l1_feat.permute(0, 2, 1))
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_feat)))
        out = self.drop1(feat)
        out = self.conv2(out)
        return out
